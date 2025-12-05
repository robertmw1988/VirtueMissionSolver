"""Linear-programming solver for optimal mission selection."""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import pulp # type: ignore

from .bom import BOMEngine, RollupResult, get_bom_engine
from .config import CostWeights, Constraints, EpicResearch, UserConfig
from .mission_data import (
    FTL_SHIPS,
    MissionOption,
    build_mission_inventory,
    compute_research_bonuses,
    filter_inventory_by_level,
)
from .solver_logging import LogLevel, SolverLogger, create_logger

# Egg types that are stored in the shared fuel tank (excludes HUMILITY)
TANK_FUEL_EGGS = frozenset({"INTEGRITY", "CURIOSITY", "KINDNESS", "RESILIENCE"})

# Humility egg is consumed directly from farm, not from tank
HUMILITY_EGG = "HUMILITY"


class SolverState:
    """
    Thread-safe state container for solver caches.
    
    Manages efficiency baselines and configuration hashes to avoid
    redundant recalculations between solver invocations.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._efficiency_baseline_cache: Dict[str, float] = {}
        self._baseline_config_hash: Optional[str] = None
    
    def get_or_compute_baselines(
        self,
        config_hash: str,
        compute_fn,
    ) -> Dict[str, float]:
        """
        Get cached baselines or compute them if config changed.
        
        Parameters
        ----------
        config_hash : str
            Hash of current mission configuration
        compute_fn : callable
            Function to call if baselines need recomputing
        
        Returns
        -------
        Dict[str, float]
            Efficiency baselines
        """
        with self._lock:
            if config_hash != self._baseline_config_hash:
                self._efficiency_baseline_cache = compute_fn()
                self._baseline_config_hash = config_hash
                return self._efficiency_baseline_cache, True  # cache_miss=True
            return self._efficiency_baseline_cache, False  # cache_miss=False
    
    def clear_cache(self) -> None:
        """Clear all cached state."""
        with self._lock:
            self._efficiency_baseline_cache = {}
            self._baseline_config_hash = None


# Module-level singleton
_solver_state: Optional[SolverState] = None
_solver_state_lock = threading.Lock()


def get_solver_state() -> SolverState:
    """Get or create the singleton solver state."""
    global _solver_state
    if _solver_state is None:
        with _solver_state_lock:
            if _solver_state is None:
                _solver_state = SolverState()
    return _solver_state


def clear_solver_cache() -> None:
    """Clear all solver caches. Call when data sources change."""
    state = get_solver_state()
    state.clear_cache()


@dataclass
class FuelUsage:
    """Fuel consumption breakdown by egg type."""
    by_egg: Dict[str, float] = field(default_factory=dict)  # egg_name -> total amount
    
    @property
    def tank_total(self) -> float:
        """Total fuel from tank eggs (excludes Humility)."""
        return sum(amt for egg, amt in self.by_egg.items() if egg in TANK_FUEL_EGGS)
    
    @property
    def humility_total(self) -> float:
        """Total Humility egg fuel (not stored in tank)."""
        return self.by_egg.get(HUMILITY_EGG, 0.0)
    
    def __str__(self) -> str:
        lines = ["Fuel Usage:"]
        for egg, amt in sorted(self.by_egg.items()):
            # Format large numbers with T/B/M suffix
            if amt >= 1e12:
                formatted = f"{amt / 1e12:.2f}T"
            elif amt >= 1e9:
                formatted = f"{amt / 1e9:.2f}B"
            elif amt >= 1e6:
                formatted = f"{amt / 1e6:.2f}M"
            else:
                formatted = f"{amt:,.0f}"
            tank_note = "" if egg == HUMILITY_EGG else " (tank)"
            lines.append(f"  {egg}: {formatted}{tank_note}")
        lines.append(f"  Tank Total: {self.tank_total / 1e12:.2f}T")
        return "\n".join(lines)


@dataclass
class SolverResult:
    status: str
    objective_value: float
    selected_missions: List[Tuple[MissionOption, int]]  # (mission, count)
    total_drops: Dict[str, float]
    total_time_hours: float
    fuel_usage: FuelUsage = field(default_factory=FuelUsage)
    bom_rollup: Optional[RollupResult] = None  # BOM rollup if crafting weights provided
    slack_drops: Dict[str, float] = field(default_factory=dict)  # Unwanted artifact drops
    slack_percentage: float = 0.0  # Percentage of total drops that are slack (0-100)


def _compute_research_bonuses(epic: Dict[str, EpicResearch]) -> Tuple[float, float]:
    """Return (capacity_bonus, ftl_time_reduction) from epic research."""
    capacity_bonus = 0.0
    ftl_time_reduction = 0.0

    zgqc = epic.get("Zero-G Quantum Containment")
    if zgqc:
        # effect is per-level multiplier (e.g. 0.05 = 5%)
        capacity_bonus = zgqc.level * zgqc.effect

    ftl = epic.get("FTL Drive Upgrades")
    if ftl:
        # effect is per-level multiplier (e.g. 0.01 = 1%)
        ftl_time_reduction = ftl.level * ftl.effect

    return capacity_bonus, ftl_time_reduction


def calculate_fuel_usage(
    selected_missions: List[Tuple[MissionOption, int]],
) -> FuelUsage:
    """
    Calculate total fuel usage from a list of selected missions.
    
    Parameters
    ----------
    selected_missions : list of (MissionOption, count) tuples
    
    Returns
    -------
    FuelUsage with breakdown by egg type
    """
    fuel_by_egg: Dict[str, float] = {}
    
    for mission, count in selected_missions:
        for egg, amount in mission.fuel_requirements.items():
            fuel_by_egg[egg] = fuel_by_egg.get(egg, 0.0) + amount * count
    
    return FuelUsage(by_egg=fuel_by_egg)


def get_fuel_coefficients(
    inventory: List[MissionOption],
) -> Dict[str, List[float]]:
    """
    Build coefficient vectors for fuel constraints.
    
    Returns a dict mapping each tank egg type to a list of coefficients,
    where coefficient[i] is the amount of that egg required by mission i.
    Humility egg is excluded as it's not stored in the tank.
    
    Parameters
    ----------
    inventory : list of MissionOption
    
    Returns
    -------
    Dict mapping egg name -> list of fuel amounts per mission
    """
    # Collect all tank fuel egg types used
    all_eggs: set[str] = set()
    for m in inventory:
        for egg in m.fuel_requirements:
            if egg in TANK_FUEL_EGGS:
                all_eggs.add(egg)
    
    # Build coefficient vectors
    coefficients: Dict[str, List[float]] = {egg: [] for egg in all_eggs}
    for m in inventory:
        for egg in all_eggs:
            coefficients[egg].append(float(m.fuel_requirements.get(egg, 0)))
    
    return coefficients


def calculate_fuel_per_artifact(mission: MissionOption, capacity: int) -> float:
    """
    Calculate the total fuel cost per artifact for a mission.
    
    This represents the "opportunity cost" of each artifact slot:
    high-tier missions cost billions per artifact, while low-tier
    missions cost only millions.
    
    Parameters
    ----------
    mission : MissionOption
        The mission to calculate fuel cost for
    capacity : int
        Effective capacity (artifacts returned)
    
    Returns
    -------
    float
        Total fuel (all egg types) divided by capacity
    """
    if capacity <= 0:
        return 0.0
    
    total_fuel = sum(mission.fuel_requirements.values())
    return total_fuel / capacity


def get_tank_fuel(mission: MissionOption) -> float:
    """
    Get total tank fuel for a mission (excludes Humility egg).
    
    Parameters
    ----------
    mission : MissionOption
        The mission to calculate tank fuel for
    
    Returns
    -------
    float
        Sum of fuel requirements for tank eggs only
    """
    return sum(
        amount for egg, amount in mission.fuel_requirements.items()
        if egg in TANK_FUEL_EGGS
    )


def _compute_mission_config_hash(missions: Dict[str, int]) -> str:
    """Compute a hash of the mission configuration for cache invalidation."""
    config_str = str(sorted(missions.items()))
    return hashlib.md5(config_str.encode()).hexdigest()


def calculate_efficiency_baselines(
    inventory: List[MissionOption],
    effective_caps: List[int],
    effective_secs: List[int],
    logger: Optional[SolverLogger] = None,
) -> Dict[str, float]:
    """
    Calculate baseline values for efficiency normalization.
    
    Returns min/max values for fuel per artifact and artifacts per hour
    across all missions. These are used to normalize efficiency ratios to 0-1.
    
    Excludes missions that use ONLY Humility fuel from fuel baseline calculation
    (they have tank_fuel=0, which would skew the minimum).
    
    Parameters
    ----------
    inventory : List[MissionOption]
        All available mission options
    effective_caps : List[int]
        Effective capacities for each mission
    effective_secs : List[int]
        Effective durations in seconds for each mission
    logger : SolverLogger, optional
        Logger for debugging
    
    Returns
    -------
    Dict with keys: min_fuel, max_fuel, min_arts_hr, max_arts_hr
    """
    fuel_values = []  # Tank fuel per artifact (for missions with tank fuel)
    arts_hr_values = []  # Artifacts per hour
    
    for i, m in enumerate(inventory):
        cap = effective_caps[i]
        secs = effective_secs[i]
        
        if cap <= 0 or secs <= 0:
            continue
        
        # Tank fuel per artifact (exclude humility-only missions)
        tank_fuel = get_tank_fuel(m)
        if tank_fuel > 0:  # Only include missions that use tank fuel
            fuel_per_art = tank_fuel / cap
            fuel_values.append(fuel_per_art)
        
        # Artifacts per hour
        hours = secs / 3600.0
        arts_per_hr = cap / hours
        arts_hr_values.append(arts_per_hr)
    
    # Calculate baselines
    baselines = {
        'min_fuel': min(fuel_values) if fuel_values else 0.0,
        'max_fuel': max(fuel_values) if fuel_values else 1.0,
        'min_arts_hr': min(arts_hr_values) if arts_hr_values else 0.0,
        'max_arts_hr': max(arts_hr_values) if arts_hr_values else 1.0,
    }
    
    if logger:
        logger._log(LogLevel.DETAILED, "EFFICIENCY",
                    f"Baselines: fuel=[{baselines['min_fuel']:.2e}, {baselines['max_fuel']:.2e}]/art, "
                    f"arts/hr=[{baselines['min_arts_hr']:.1f}, {baselines['max_arts_hr']:.1f}]")
    
    return baselines


def calculate_targeted_waste(
    mission: MissionOption,
    drop_ratios: Dict[str, float],
    capacity: int,
    mission_artifact_weights: Dict[str, float],
) -> Tuple[float, float]:
    """
    Calculate targeted drops and waste drops for a mission.
    
    Uses mission_artifact_weights to determine which artifacts are:
    - Targeted (weight > 0): counted as targeted
    - Acceptable (weight = 0): neutral, not counted in either
    - Waste (weight < 0): counted as waste
    
    Parameters
    ----------
    mission : MissionOption
        The mission
    drop_ratios : Dict[str, float]
        Drop ratios for each artifact
    capacity : int
        Effective capacity
    mission_artifact_weights : Dict[str, float]
        Weights defining targeted/acceptable/waste categories
    
    Returns
    -------
    Tuple[float, float]
        (targeted_drops, waste_drops)
    """
    targeted = 0.0
    waste = 0.0
    
    for art, ratio in drop_ratios.items():
        if ratio <= 0:
            continue
        
        expected = ratio * capacity
        weight = mission_artifact_weights.get(art, 0.0)
        
        if weight > 0:
            targeted += expected
        elif weight < 0:
            waste += expected
        # weight == 0 is "Acceptable" - neutral, don't count
    
    return targeted, waste


def calculate_efficiency_factors(
    mission_idx: int,
    mission: MissionOption,
    effective_cap: int,
    effective_secs: int,
    drop_ratios: Dict[str, float],
    mission_artifact_weights: Dict[str, float],
    baselines: Dict[str, float],
    weights: CostWeights,
    logger: Optional[SolverLogger] = None,
) -> Dict[str, float]:
    """
    Calculate efficiency factors for a single mission.
    
    Each factor uses the formula: (scale × ratio)^power
    where ratio is normalized to 0-1 range (1 = best, 0 = worst).
    
    If power = 0, the factor evaluates to 1.0 (ignored).
    
    Parameters
    ----------
    mission_idx : int
        Index of mission in inventory (for logging)
    mission : MissionOption
        The mission to evaluate
    effective_cap : int
        Effective capacity for this mission
    effective_secs : int
        Effective duration in seconds
    drop_ratios : Dict[str, float]
        Drop ratios for this mission
    mission_artifact_weights : Dict[str, float]
        Weights for targeted/waste classification
    baselines : Dict[str, float]
        Baseline values from calculate_efficiency_baselines
    weights : CostWeights
        Weight configuration with scale/power parameters
    logger : SolverLogger, optional
        Logger for per-mission factor logging
    
    Returns
    -------
    Dict with keys: fuel_factor, time_factor, waste_factor, combined_factor
    and intermediate values for logging
    """
    result = {
        'fuel_ratio': 0.0,
        'fuel_factor': 1.0,
        'time_ratio': 0.0,
        'time_factor': 1.0,
        'waste_ratio': 0.0,
        'waste_factor': 1.0,
        'combined_factor': 1.0,
    }
    
    # ----- Fuel Efficiency -----
    # Ratio: (max_fuel - mission_fuel) / (max_fuel - min_fuel)
    # High ratio = low fuel usage = good
    tank_fuel = get_tank_fuel(mission)
    min_fuel = baselines['min_fuel']
    max_fuel = baselines['max_fuel']
    
    if effective_cap > 0 and tank_fuel > 0 and max_fuel > min_fuel:
        fuel_per_art = tank_fuel / effective_cap
        fuel_ratio = (max_fuel - fuel_per_art) / (max_fuel - min_fuel)
        fuel_ratio = max(0.0, min(1.0, fuel_ratio))  # Clamp to [0, 1]
        result['fuel_ratio'] = fuel_ratio
        
        if weights.fuel_efficiency_power > 0:
            scaled = weights.fuel_efficiency_scale * fuel_ratio
            result['fuel_factor'] = scaled ** weights.fuel_efficiency_power
        # else: power=0 -> factor=1.0 (already set)
    
    # ----- Time Efficiency -----
    # Ratio: (mission_arts_hr - min_arts_hr) / (max_arts_hr - min_arts_hr)
    # High ratio = more artifacts per hour = good
    min_arts_hr = baselines['min_arts_hr']
    max_arts_hr = baselines['max_arts_hr']
    
    if effective_cap > 0 and effective_secs > 0 and max_arts_hr > min_arts_hr:
        hours = effective_secs / 3600.0
        arts_per_hr = effective_cap / hours
        time_ratio = (arts_per_hr - min_arts_hr) / (max_arts_hr - min_arts_hr)
        time_ratio = max(0.0, min(1.0, time_ratio))  # Clamp to [0, 1]
        result['time_ratio'] = time_ratio
        
        if weights.time_efficiency_power > 0:
            scaled = weights.time_efficiency_scale * time_ratio
            result['time_factor'] = scaled ** weights.time_efficiency_power
        # else: power=0 -> factor=1.0 (already set)
    
    # ----- Waste Efficiency -----
    # Ratio: targeted / (targeted + waste)
    # High ratio = more targeted drops = good
    # Note: "Acceptable" (weight=0) artifacts are neutral and not counted
    targeted, waste = calculate_targeted_waste(
        mission, drop_ratios, effective_cap, mission_artifact_weights
    )
    
    denominator = targeted + waste
    if denominator > 0:
        waste_ratio = targeted / denominator
        result['waste_ratio'] = waste_ratio
        
        if weights.waste_efficiency_power > 0:
            scaled = weights.waste_efficiency_scale * waste_ratio
            result['waste_factor'] = scaled ** weights.waste_efficiency_power
        # else: power=0 -> factor=1.0 (already set)
    else:
        # No targeted or waste - mission only has acceptable artifacts
        result['waste_ratio'] = 1.0  # Treat as fully efficient
    
    # ----- Combined Factor -----
    result['combined_factor'] = (
        result['fuel_factor'] * 
        result['time_factor'] * 
        result['waste_factor']
    )
    
    if logger:
        logger._log(LogLevel.TRACE, "EFFICIENCY",
                    f"  Mission {mission_idx} ({mission.ship} {mission.duration_type}): "
                    f"fuel={result['fuel_ratio']:.3f}^{weights.fuel_efficiency_power}->{result['fuel_factor']:.4f}, "
                    f"time={result['time_ratio']:.3f}^{weights.time_efficiency_power}->{result['time_factor']:.4f}, "
                    f"waste={result['waste_ratio']:.3f}^{weights.waste_efficiency_power}->{result['waste_factor']:.4f}, "
                    f"combined={result['combined_factor']:.4f}")
    
    return result


def solve(
    config: UserConfig,
    num_ships: int = 3,
    verbose: bool = False,
    logger: Optional[SolverLogger] = None,
    log_level: Union[LogLevel, str, int, None] = None,
) -> SolverResult:
    """
    Formulate and solve mission LP.

    Decision variables: x[i] = integer count of how many times to run mission i.
    Objective: maximise weighted artifact gain minus time cost.
    Constraints:
        - total elapsed time <= max_time_hours (mission time / num_ships)
        - fuel usage <= fuel tank capacity
    
    The number of missions is not limited - the solver finds an optimal list
    regardless of length. The concurrent ships count affects elapsed time calculation
    since multiple missions can run in parallel.
    
    Parameters
    ----------
    config : UserConfig
        User configuration with missions, weights, and constraints
    num_ships : int
        Number of concurrent mission slots (affects time calculation)
    verbose : bool
        If True, enables CBC solver output (deprecated, use log_level instead)
    logger : SolverLogger, optional
        Pre-configured logger. If None, one is created based on log_level.
    log_level : LogLevel | str | int, optional
        Logging verbosity. Only used if logger is None.
        - SILENT: No output
        - MINIMAL: Only results
        - SUMMARY: Config overview and key metrics
        - DETAILED: Coefficient tables
        - DEBUG: Full solver state
        - TRACE: Per-artifact calculations
    
    Returns
    -------
    SolverResult
        Solution with selected missions, drops, and metrics
    """
    start_time = time.perf_counter()
    
    # Set up logging
    if logger is None:
        if log_level is not None:
            logger = create_logger(level=log_level)
        elif verbose:
            logger = create_logger(level=LogLevel.SUMMARY)
        else:
            logger = create_logger(level=LogLevel.SILENT)
    
    # Log configuration
    logger.log_config_start(config, num_ships)
    logger.log_cost_weights(config.cost_weights)
    logger.log_epic_research(config.epic_researches)
    logger.log_artifact_weights(config.mission_artifact_weights, "Mission")
    if config.crafted_artifact_weights:
        logger.log_artifact_weights(config.crafted_artifact_weights, "Crafted")
    
    # Build mission inventory filtered by user's unlocked levels
    full_inventory = build_mission_inventory(allowed_ships=config.missions)
    inventory = filter_inventory_by_level(full_inventory, config.missions)
    logger.log_inventory_built(len(full_inventory), len(inventory))

    if not inventory:
        logger._log(LogLevel.MINIMAL, "SOLVER", "No missions available - returning empty result")
        return SolverResult(
            status="No missions available",
            objective_value=0.0,
            selected_missions=[],
            total_drops={},
            total_time_hours=0.0,
        )

    capacity_bonus, ftl_reduction = _compute_research_bonuses(config.epic_researches)
    logger._log(LogLevel.DETAILED, "RESEARCH", 
                f"Research bonuses: capacity={capacity_bonus:.2%}, ftl_reduction={ftl_reduction:.2%}")

    # Pre-compute effective values for each mission
    effective_caps: List[int] = []
    effective_secs: List[int] = []
    drop_ratios_list: List[Dict[str, float]] = []
    for m in inventory:
        mission_level = config.missions.get(m.ship, 0)
        is_ftl = m.ship in FTL_SHIPS
        effective_caps.append(m.effective_capacity(mission_level, capacity_bonus))
        effective_secs.append(m.effective_seconds(ftl_reduction, is_ftl))
        drop_ratios_list.append(m.drop_ratios())

    # Log mission details
    logger.log_mission_details(inventory, effective_caps, effective_secs)

    # Collect all artifact columns present in any mission
    all_artifacts_set: set[str] = set()
    for m in inventory:
        all_artifacts_set.update(m.drop_vector.keys())
    all_artifacts = sorted(all_artifacts_set)
    logger._log(LogLevel.DETAILED, "ARTIFACTS", 
                f"Total unique artifacts across all missions: {len(all_artifacts)}")

    # ----- Calculate Base Ingredient Equivalence for All Artifacts -----
    # For crafting optimization, we value drops by their "base ingredient worth"
    # A T3 artifact that crafts from 49 T1s is worth 49 base ingredients
    # This makes the solver prefer missions that return higher-tier ingredients
    # (more time-efficient for gathering crafting materials)
    
    art_weights = config.mission_artifact_weights.copy()
    
    # Calculate base ingredient equivalence for all artifacts
    # This tells us how many base T1 ingredients each drop is "worth"
    base_equivalence: Dict[str, float] = {}
    ingredient_ratios: Dict[str, Dict[str, float]] = {}
    target_ingredients: Set[str] = set()  # All ingredients needed for any target
    
    if config.crafted_artifact_weights:
        try:
            engine = get_bom_engine()
            
            # Get ingredient ratios for ratio-balancing constraints
            ingredient_ratios = engine.get_ingredient_ratios_for_targets(
                config.crafted_artifact_weights
            )
            if ingredient_ratios:
                logger._log(LogLevel.DETAILED, "BOM", 
                            f"Got ingredient ratios for {len(ingredient_ratios)} targets")
                for target, ratios in ingredient_ratios.items():
                    logger._log(LogLevel.TRACE, "BOM", 
                                f"  {target}: {len(ratios)} base ingredients")
                    target_ingredients.update(ratios.keys())
            
            # For each target, get all contributors and their base equivalence
            for target_name, weight in config.crafted_artifact_weights.items():
                if weight <= 0:
                    continue
                    
                contributors = engine.get_all_contributors_for_target(target_name)
                for drop_name, base_contribs in contributors.items():
                    # Sum of base ingredients this drop contributes
                    total_base_worth = sum(base_contribs.values())
                    base_equivalence[drop_name] = max(
                        base_equivalence.get(drop_name, 0.0),
                        total_base_worth
                    )
            
            if base_equivalence:
                logger._log(LogLevel.DETAILED, "BOM", 
                            f"Calculated base equivalence for {len(base_equivalence)} artifacts")
                # Log top equivalences
                top_equiv = sorted(base_equivalence.items(), key=lambda kv: -kv[1])[:10]
                for name, equiv in top_equiv:
                    logger._log(LogLevel.TRACE, "BOM", 
                                f"  {name}: base_equivalence={equiv:.1f}")
        except Exception as e:
            logger._log(LogLevel.MINIMAL, "BOM", 
                        f"Failed to calculate ingredient equivalence: {e}")
    
    # Determine if we're in crafting mode (have crafting targets with ratio constraints)
    using_crafting_mode = bool(ingredient_ratios)

    # ----- Calculate Efficiency Baselines -----
    # Check if we need to recalculate baselines (mission config changed)
    solver_state = get_solver_state()
    config_hash = _compute_mission_config_hash(config.missions)
    
    efficiency_baselines, cache_miss = solver_state.get_or_compute_baselines(
        config_hash,
        lambda: calculate_efficiency_baselines(inventory, effective_caps, effective_secs, logger)
    )
    
    if cache_miss:
        logger._log(LogLevel.DETAILED, "EFFICIENCY", 
                    f"Recalculated efficiency baselines (config hash: {config_hash[:8]})")
    else:
        logger._log(LogLevel.DETAILED, "EFFICIENCY", 
                    f"Using cached efficiency baselines (config hash: {config_hash[:8]})")

    # ----- LP Setup -----
    logger.log_objective_start()
    prob = pulp.LpProblem("MissionOptimizer", pulp.LpMaximize)

    # Decision variables: how many times to schedule each mission option
    x = [
        pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger)
        for i in range(len(inventory))
    ]

    # Objective: Maximize crafting progress with time efficiency
    # 
    # When crafting targets are set:
    # - Base value = sum of (drops × base_ingredient_equivalence)
    #   A T3 drop worth 49 T1s contributes 49 to base value
    # - Only count drops that contribute to crafting targets
    # 
    # When no crafting targets:
    # - Use mission artifact weights as before
    #
    # Efficiency factors tune the solution (each uses formula: (scale × ratio)^power):
    # - Fuel efficiency: Penalizes high fuel cost per artifact
    # - Time efficiency: Rewards more artifacts per hour
    # - Waste efficiency: Penalizes missions with non-targeted drops
    #
    # The formula:
    #   contrib = base_value × fuel_factor × time_factor × waste_factor
    #
    # Each factor = 1.0 when power = 0 (ignored)
    
    obj_terms = []
    weights = config.cost_weights
    objective_coefficients: List[float] = []
    objective_components: List[Dict[str, float]] = []
    
    for i, m in enumerate(inventory):
        cap = effective_caps[i] if effective_caps[i] > 0 else 1
        ratios = drop_ratios_list[i]
        time_hours = effective_secs[i] / 3600.0
        
        # Calculate value based on mode
        total_base_ingredient_value = 0.0  # Sum of base ingredient equivalences
        total_weighted_value = 0.0  # For non-crafting mode
        targeted_drops = 0.0
        waste_drops = 0.0
        
        for art in all_artifacts:
            ratio = ratios.get(art, 0.0)
            if ratio <= 0:
                continue
            
            expected_drops = ratio * cap
            
            if using_crafting_mode:
                # CRAFTING MODE: Value drops by their base ingredient equivalence
                # Only count drops that are contributors to crafting targets
                equiv = base_equivalence.get(art, 0.0)
                
                if equiv > 0:
                    # This drop contributes to crafting targets
                    # Value = number of drops × base ingredient worth per drop
                    contribution = expected_drops * equiv
                    total_base_ingredient_value += contribution
                    targeted_drops += expected_drops * equiv  # Weight by equivalence
                    
                    logger.log_objective_artifact_detail(
                        i, art, ratio, expected_drops, equiv, contribution
                    )
                else:
                    # Not a crafting contributor - check if waste
                    art_weight = art_weights.get(art, 0.0)
                    if art_weight < 0:
                        waste_drops += expected_drops
            else:
                # NON-CRAFTING MODE: Use mission artifact weights
                # Default weight is 1.0 if no weights are specified (empty dict)
                # This means "all artifacts are equally valuable"
                if art_weights:
                    art_value = art_weights.get(art, 0.0)  # Explicit weights: default 0
                else:
                    art_value = 1.0  # No weights specified: all artifacts equal
                
                if art_value > 0:
                    contribution = expected_drops * art_value
                    total_weighted_value += contribution
                    targeted_drops += expected_drops
                elif art_value < 0:
                    waste_drops += expected_drops
                    contribution = 0.0
                else:
                    contribution = 0.0
                
                logger.log_objective_artifact_detail(
                    i, art, ratio, expected_drops, art_value, contribution
                )
        
        # Calculate efficiency ratio (for logging and non-crafting mode)
        efficiency_denominator = targeted_drops + waste_drops
        efficiency_ratio = targeted_drops / efficiency_denominator if efficiency_denominator > 0 else 1.0
        
        # Calculate efficiency factors for this mission
        eff_factors = calculate_efficiency_factors(
            mission_idx=i,
            mission=m,
            effective_cap=cap,
            effective_secs=effective_secs[i],
            drop_ratios=ratios,
            mission_artifact_weights=config.mission_artifact_weights,
            baselines=efficiency_baselines,
            weights=weights,
            logger=logger,
        )
        
        if using_crafting_mode:
            # CRAFTING MODE OBJECTIVE:
            # 
            # contrib = base_value × efficiency_factors
            #
            # Efficiency factors (multiplicative):
            # - fuel_factor: (fuel_scale × fuel_ratio)^fuel_power
            # - time_factor: (time_scale × time_ratio)^time_power  
            # - waste_factor: (waste_scale × waste_ratio)^waste_power
            # Each factor = 1.0 when power = 0 (ignored)
            
            base_value = total_base_ingredient_value
            combined_efficiency = eff_factors['combined_factor']
            
            # Artifact value scaled by efficiency factors
            artifact_value_term = base_value * combined_efficiency
            
            contrib = artifact_value_term
            
            objective_components.append({
                'artifact_value': artifact_value_term,
                'base_ingredient_value': base_value,
                'fuel_factor': eff_factors['fuel_factor'],
                'time_factor': eff_factors['time_factor'],
                'waste_factor': eff_factors['waste_factor'],
                'combined_efficiency': combined_efficiency,
                'fuel_ratio': eff_factors['fuel_ratio'],
                'time_ratio': eff_factors['time_ratio'],
                'waste_ratio': eff_factors['waste_ratio'],
            })
        else:
            # NON-CRAFTING MODE: Apply efficiency factors to artifact value
            # 
            # contrib = weighted_value × efficiency_factors
            #
            # Same efficiency factors as crafting mode
            combined_efficiency = eff_factors['combined_factor']
            artifact_value = total_weighted_value * combined_efficiency
            
            contrib = artifact_value
            
            objective_components.append({
                'artifact_value': artifact_value,
                'fuel_factor': eff_factors['fuel_factor'],
                'time_factor': eff_factors['time_factor'],
                'waste_factor': eff_factors['waste_factor'],
                'combined_efficiency': combined_efficiency,
                'fuel_ratio': eff_factors['fuel_ratio'],
                'time_ratio': eff_factors['time_ratio'],
                'waste_ratio': eff_factors['waste_ratio'],
            })
        
        obj_terms.append(contrib * x[i])
        objective_coefficients.append(contrib)
        
        logger.log_objective_mission_contribution(
            i, f"{m.ship}_{m.duration_type}", 
            objective_components[-1].get('artifact_value', 0),
            objective_components[-1].get('combined_efficiency', 1.0),
            contrib
        )

    # Note: We'll add obj_terms to the objective later, after ratio terms are computed
    
    # Log objective summary
    logger.log_objective_summary(objective_coefficients)
    logger.log_objective_coefficients_table(inventory, objective_coefficients, objective_components)

    # ----- Constraints -----
    logger.log_constraints_start()

    # Constraint: max missions per type (forces mission diversity)
    if config.constraints.max_missions_per_type > 0:
        for i, m in enumerate(inventory):
            prob += x[i] <= config.constraints.max_missions_per_type, f"MaxPerType_{i}"
        logger.log_constraint_added("MaxMissionsPerType", 
                                    f"Each mission type <= {config.constraints.max_missions_per_type}",
                                    config.constraints.max_missions_per_type)

    # Constraint: total time (accounting for concurrent execution)
    # With num_ships concurrent slots, actual elapsed time = sum(mission_times) / num_ships
    time_expr = pulp.lpSum(
        (effective_secs[i] / 3600.0) * x[i] for i in range(len(inventory))
    )
    prob += time_expr <= config.constraints.max_time_hours * num_ships, "MaxTotalTime"
    logger.log_constraint_added("MaxTotalTime", 
                                f"Total mission time <= {config.constraints.max_time_hours}h * {num_ships} ships = {config.constraints.max_time_hours * num_ships}h",
                                config.constraints.max_time_hours * num_ships)

    # Constraint: fuel tank capacity (excludes Humility egg)
    fuel_coeffs = get_fuel_coefficients(inventory)
    fuel_tank_capacity = config.constraints.fuel_tank_capacity * 1e12  # Convert from T to raw
    logger.log_fuel_coefficients(fuel_coeffs)
    
    # Sum of all tank fuels must not exceed capacity
    tank_fuel_expr = None
    if fuel_coeffs:
        tank_fuel_expr = pulp.lpSum(
            fuel_coeffs[egg][i] * x[i]
            for egg in fuel_coeffs
            for i in range(len(inventory))
        )
        prob += tank_fuel_expr <= fuel_tank_capacity, "FuelTankCapacity"
        logger.log_constraint_added("FuelTankCapacity",
                                    f"Tank fuel <= {config.constraints.fuel_tank_capacity}T",
                                    fuel_tank_capacity)

    # ----- Ingredient Ratio Balancing Constraints -----
    # For each crafted target, we add constraints that encourage balanced ingredient collection.
    # 
    # For a target needing r_A of ingredient A and r_B of ingredient B:
    # - craftable_target <= total_drops_A / r_A  (limited by A)
    # - craftable_target <= total_drops_B / r_B  (limited by B)
    # 
    # By maximizing craftable_target (weighted by target value), the solver is
    # incentivized to collect ingredients in the right proportions.
    # 
    # IMPORTANT: Mission drops are often NOT base ingredients. For example:
    # - Missions drop "Adequate quantum metronome" (not "Misaligned")
    # - "Adequate" = 7x "Misaligned" in crafting terms
    # 
    # We convert all drops to "base ingredient equivalence" using the BOM.
    
    craftable_vars: Dict[str, pulp.LpVariable] = {}
    ratio_objective_terms: List[pulp.LpAffineExpression] = []
    
    if ingredient_ratios and config.crafted_artifact_weights:
        logger._log(LogLevel.DETAILED, "RATIO", 
                    f"Adding ratio-balancing constraints for {len(ingredient_ratios)} targets")
        
        # For each target, get all contributing artifacts and their base equivalence
        engine = get_bom_engine()
        
        for target_name, base_ratios in ingredient_ratios.items():
            target_weight = config.crafted_artifact_weights.get(target_name, 0.0)
            if target_weight <= 0:
                continue
            
            # Get all artifacts that contribute to this target
            contributors = engine.get_all_contributors_for_target(target_name)
            
            if not contributors:
                logger._log(LogLevel.TRACE, "RATIO", 
                            f"  {target_name}: no contributors found, skipping")
                continue
            
            # Build expressions for effective base ingredient supply
            # Each dropped artifact contributes its base equivalence
            base_ingredient_supply: Dict[str, pulp.LpAffineExpression] = {}
            
            for base_ing in base_ratios.keys():
                # Sum up contributions from all drops that provide this base ingredient
                supply_terms = []
                
                for drop_name in all_artifacts:
                    # Check if this drop contributes to this base ingredient
                    drop_contribution = contributors.get(drop_name, {})
                    equivalence = drop_contribution.get(base_ing, 0.0)
                    
                    if equivalence > 0:
                        # This drop provides `equivalence` units of base_ing per drop
                        for i in range(len(inventory)):
                            drop_rate = drop_ratios_list[i].get(drop_name, 0.0)
                            if drop_rate > 0:
                                supply_terms.append(
                                    drop_rate * effective_caps[i] * equivalence * x[i]
                                )
                
                if supply_terms:
                    base_ingredient_supply[base_ing] = pulp.lpSum(supply_terms)
            
            if not base_ingredient_supply:
                logger._log(LogLevel.TRACE, "RATIO", 
                            f"  {target_name}: no supply expressions, skipping")
                continue
            
            # Create variable for craftable units of this target
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', target_name)
            craftable_var = pulp.LpVariable(f"craftable_{safe_name}", lowBound=0, cat=pulp.LpContinuous)
            craftable_vars[target_name] = craftable_var
            
            # Add constraints: craftable_target <= supply_base_ing / ratio
            constraints_added = 0
            for base_ing, ratio in base_ratios.items():
                if ratio <= 0:
                    continue
                
                supply_expr = base_ingredient_supply.get(base_ing)
                if supply_expr is None:
                    continue
                
                # craftable_target * ratio <= supply
                prob += craftable_var * ratio <= supply_expr, f"Ratio_{safe_name}_{constraints_added}"
                constraints_added += 1
            
            if constraints_added > 0:
                # Add to objective: maximize craftable units weighted by target value
                ratio_objective_terms.append(
                    target_weight * craftable_var
                )
                logger._log(LogLevel.DETAILED, "RATIO", 
                            f"  {target_name}: {constraints_added} ratio constraints, weight={target_weight}")
        
        if ratio_objective_terms:
            logger._log(LogLevel.SUMMARY, "RATIO", 
                        f"Added ratio-balancing objective for {len(craftable_vars)} targets")

    # ----- Set Combined Objective -----
    # Combine mission objective terms with ratio-balancing terms
    all_objective_terms = obj_terms + ratio_objective_terms
    prob += pulp.lpSum(all_objective_terms), "TotalObjective"

    # ----- Solve -----
    solver_verbose = verbose or logger.level >= LogLevel.DEBUG
    logger.log_solver_start("PULP_CBC_CMD", solver_verbose)
    
    # Enable warm-start for faster resolves when weights change but structure is similar
    # On Windows, warmStart requires keepFiles=True to work properly
    import platform
    use_keep_files = platform.system() == "Windows"
    solver = pulp.PULP_CBC_CMD(msg=solver_verbose, warmStart=True, keepFiles=use_keep_files)
    prob.solve(solver)
    
    solve_time_ms = (time.perf_counter() - start_time) * 1000

    status = pulp.LpStatus[prob.status]
    raw_objective = pulp.value(prob.objective)
    objective_value = raw_objective if raw_objective is not None else 0.0
    logger.log_solver_complete(status, objective_value, solve_time_ms)

    # ----- Extract solution -----
    selected: List[Tuple[MissionOption, int]] = []
    total_drops: Dict[str, float] = {art: 0.0 for art in all_artifacts}
    slack_drops: Dict[str, float] = {}  # Only artifacts with combined value <= 0
    total_time_hours = 0.0
    total_slack_drops = 0.0
    total_all_drops = 0.0
    
    # Build capacity list aligned with selected missions for logging
    selected_capacities: List[int] = []

    for i, m in enumerate(inventory):
        count = int(pulp.value(x[i]) or 0)
        if count > 0:
            selected.append((m, count))
            selected_capacities.append(effective_caps[i])
            cap = effective_caps[i] if effective_caps[i] > 0 else 1
            ratios = drop_ratios_list[i]
            for art, ratio in ratios.items():
                # Expected drops = ratio * capacity * count
                expected = ratio * cap * count
                total_drops[art] += expected
                total_all_drops += expected
                
                # Track slack (unwanted) artifacts
                # In crafting mode: slack = not a crafting contributor
                # In non-crafting mode: slack = weight <= 0
                if using_crafting_mode:
                    is_slack = base_equivalence.get(art, 0.0) <= 0
                else:
                    if art_weights:
                        is_slack = art_weights.get(art, 0.0) <= 0
                    else:
                        is_slack = False  # All artifacts equally valuable
                
                if is_slack:
                    slack_drops[art] = slack_drops.get(art, 0.0) + expected
                    total_slack_drops += expected
                    
            total_time_hours += (effective_secs[i] / 3600.0) * count
            
            logger._log(LogLevel.DEBUG, "SOLUTION",
                        f"Selected: {m.ship} {m.duration_type} x{count} "
                        f"(cap={cap}, time={effective_secs[i]/3600:.1f}h)")
    
    # Convert total mission time to elapsed time accounting for concurrent ships
    elapsed_time_hours = total_time_hours / num_ships if num_ships > 0 else total_time_hours

    # Calculate fuel usage and slack percentage
    fuel_usage = calculate_fuel_usage(selected)
    slack_pct = (total_slack_drops / total_all_drops * 100) if total_all_drops > 0 else 0.0

    # Build artifact values for logging
    # In crafting mode: use base_equivalence (0 = not a contributor)
    # In non-crafting mode: use art_weights (or 1.0 for all if empty)
    log_art_values: Dict[str, float] = {}
    if using_crafting_mode:
        log_art_values = base_equivalence.copy()
    elif art_weights:
        log_art_values = art_weights.copy()
    else:
        log_art_values = {art: 1.0 for art in all_artifacts}

    # Log solution details
    logger.log_solution_summary(status, len(selected), elapsed_time_hours, 
                                fuel_usage.tank_total, objective_value)
    logger.log_selected_missions(selected, selected_capacities)
    logger.log_expected_drops(total_drops, log_art_values)
    logger.log_slack_analysis(slack_drops, slack_pct)

    # Perform BOM rollup if crafting weights are configured
    bom_rollup: Optional[RollupResult] = None
    if config.crafted_artifact_weights:
        try:
            engine = get_bom_engine()
            bom_rollup = engine.rollup_with_display_names(
                inventory=total_drops,
                crafting_weights=config.crafted_artifact_weights,
            )
            logger.log_bom_rollup(bom_rollup)
        except Exception as e:
            # Gracefully handle BOM errors - rollup is optional
            logger._log(LogLevel.MINIMAL, "BOM", f"BOM rollup failed: {e}")
            bom_rollup = None

    logger._log(LogLevel.MINIMAL, "SOLVER", 
                f"Solve completed in {solve_time_ms:.1f}ms")

    return SolverResult(
        status=status,
        objective_value=objective_value,
        selected_missions=selected,
        total_drops=total_drops,
        total_time_hours=elapsed_time_hours,
        fuel_usage=fuel_usage,
        bom_rollup=bom_rollup,
        slack_drops=slack_drops,
        slack_percentage=slack_pct,
    )
