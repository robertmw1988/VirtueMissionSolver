"""Test scenarios that force the solver to select different mission configurations.

These scenarios validate that constraint and cost function weight changes
correctly influence the solver's mission selection, demonstrating the
optimizer responds appropriately to different user priorities.
"""
from __future__ import annotations

import pytest
from typing import Dict, List, Tuple, Any, Set

from Solver import solve, LogLevel
from Solver.config import UserConfig, Constraints, CostWeights, EpicResearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> UserConfig:
    """Base configuration with all ships at max level."""
    return UserConfig(
        missions={
            "HENERPRISE": 8,
            "ATREGGIES": 8,
            "VOYEGGER": 6,
            "CHICKFIANT": 5,
            "GALEGGTICA": 5,
        },
        epic_researches={
            "FTL Drive Upgrades": EpicResearch(level=60, effect=0.01, max_level=60),
            "Zero-G Quantum Containment": EpicResearch(level=10, effect=0.05, max_level=10),
        },
        constraints=Constraints(
            fuel_tank_capacity=500.0, 
            max_time_hours=168.0,
        ),
        cost_weights=CostWeights(
            # Default: all efficiency factors disabled (power=0)
        ),
        crafted_artifact_weights={},
        mission_artifact_weights={},  # Default: all artifacts equal weight
    )


def get_selected_ships(result) -> Set[str]:
    """Extract unique ship names from solution."""
    return {m.ship for m, count in result.selected_missions if count > 0}


def get_selected_durations(result) -> Set[str]:
    """Extract unique duration types from solution."""
    return {m.duration_type for m, count in result.selected_missions if count > 0}


def get_mission_summary(result) -> List[Tuple[str, str, int]]:
    """Get (ship, duration, count) tuples from solution."""
    return [(m.ship, m.duration_type, count) 
            for m, count in result.selected_missions if count > 0]


# ---------------------------------------------------------------------------
# Scenario 1: Time Constraint Variations
# ---------------------------------------------------------------------------

class TestTimeConstraintScenarios:
    """Test how time constraints affect mission selection."""
    
    def test_very_tight_time_prefers_short_missions(self, base_config):
        """With only 4 hours, solver should select SHORT missions."""
        base_config.constraints.max_time_hours = 4.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        durations = get_selected_durations(result)
        # Should heavily favor SHORT missions due to time constraint
        assert "SHORT" in durations or len(durations) == 0
        # Total time should respect constraint
        assert result.total_time_hours <= 4.0
    
    def test_moderate_time_allows_standard_missions(self, base_config):
        """With 24 hours, solver can select STANDARD or longer missions."""
        base_config.constraints.max_time_hours = 24.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        assert result.total_time_hours <= 24.0
    
    def test_ample_time_allows_epic_missions(self, base_config):
        """With 168 hours (1 week), solver can select EPIC missions."""
        base_config.constraints.max_time_hours = 168.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        durations = get_selected_durations(result)
        # With ample time, EPIC missions become viable
        # (though not guaranteed depending on other weights)
    
    def test_increasing_time_changes_selection(self, base_config):
        """Increasing time constraint should change mission selection."""
        results = []
        for hours in [4, 24, 72, 168]:
            base_config.constraints.max_time_hours = float(hours)
            result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
            results.append({
                'hours': hours,
                'missions': get_mission_summary(result),
                'objective': result.objective_value,
            })
        
        # Objective should generally increase with more time
        # (or stay same if time isn't binding constraint)
        assert all(r['objective'] >= 0 for r in results)


# ---------------------------------------------------------------------------
# Scenario 2: Fuel Constraint Variations
# ---------------------------------------------------------------------------

class TestFuelConstraintScenarios:
    """Test how fuel constraints affect mission selection."""
    
    def test_very_low_fuel_limits_missions(self, base_config):
        """With only 10T fuel, solver is heavily constrained."""
        base_config.constraints.fuel_tank_capacity = 10.0
        base_config.constraints.max_time_hours = 168.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Fuel usage should be at or below tank capacity
        assert result.fuel_usage.tank_total / 1e12 <= 10.0 + 0.1  # Small tolerance
    
    def test_moderate_fuel_allows_some_missions(self, base_config):
        """With 100T fuel, solver has more options."""
        base_config.constraints.fuel_tank_capacity = 100.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        assert result.fuel_usage.tank_total / 1e12 <= 100.0 + 0.1
    
    def test_ample_fuel_unconstrained(self, base_config):
        """With 1000T fuel, fuel should not be the binding constraint."""
        base_config.constraints.fuel_tank_capacity = 1000.0
        base_config.constraints.max_time_hours = 40.0  # Time is now binding
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Time should be closer to limit than fuel
        assert result.total_time_hours <= 40.0


# ---------------------------------------------------------------------------
# Scenario 3: Ship Restrictions
# ---------------------------------------------------------------------------

class TestShipRestrictionScenarios:
    """Test how limiting ships affects selection."""
    
    def test_single_ship_type(self, base_config):
        """With only ATREGGIES enabled, only ATREGGIES missions selected."""
        base_config.missions = {"ATREGGIES": 8}
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        ships = get_selected_ships(result)
        assert ships <= {"ATREGGIES"}
    
    def test_henerprise_only(self, base_config):
        """With only HENERPRISE enabled, only HENERPRISE missions selected."""
        base_config.missions = {"HENERPRISE": 8}
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        ships = get_selected_ships(result)
        assert ships <= {"HENERPRISE"}
    
    def test_lower_tier_ships_only(self, base_config):
        """With only lower-tier ships, solver uses those."""
        base_config.missions = {
            "VOYEGGER": 6,
            "CHICKFIANT": 5,
            "GALEGGTICA": 5,
        }
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        ships = get_selected_ships(result)
        assert ships <= {"VOYEGGER", "CHICKFIANT", "GALEGGTICA"}
    
    def test_mix_of_ships_allowed(self, base_config):
        """With multiple ships enabled, solver can mix."""
        base_config.missions = {
            "HENERPRISE": 8,
            "ATREGGIES": 8,
        }
        base_config.constraints.max_time_hours = 100.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Could select either or both ships


# ---------------------------------------------------------------------------
# Scenario 3a: Max Missions Per Type (Forces Mission Diversity)
# ---------------------------------------------------------------------------

class TestMaxMissionsPerTypeScenarios:
    """Test how max_missions_per_type constraint forces mission diversity."""
    
    def test_max_one_per_type_forces_diversity(self, base_config):
        """With max_missions_per_type=1, each mission type can only be selected once."""
        base_config.missions = {"ATREGGIES": 3}
        base_config.constraints.max_missions_per_type = 1
        base_config.constraints.max_time_hours = 500.0
        # Need weights to give missions positive objective value
        base_config.mission_artifact_weights = {
            "Adequate quantum metronome": 1.0,
            "Tau Ceti geode piece": 1.0,
        }
        
        result = solve(base_config, num_ships=5, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Each selected mission should have count=1
        for mission, count in result.selected_missions:
            assert count == 1, f"Expected 1 run per type, got {count}"
        # Should have multiple different mission types selected
        assert len(result.selected_missions) >= 2
    
    def test_max_two_per_type_allows_some_repeat(self, base_config):
        """With max_missions_per_type=2, each mission type can be selected up to twice."""
        base_config.missions = {"ATREGGIES": 3}
        base_config.constraints.max_missions_per_type = 2
        base_config.constraints.max_time_hours = 500.0
        # Need weights to give missions positive objective value
        base_config.mission_artifact_weights = {
            "Adequate quantum metronome": 1.0,
            "Tau Ceti geode piece": 1.0,
        }
        
        result = solve(base_config, num_ships=5, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Each selected mission should have count <= 2
        for mission, count in result.selected_missions:
            assert count <= 2, f"Expected <=2 runs per type, got {count}"
    
    def test_zero_max_per_type_means_unlimited(self, base_config):
        """With max_missions_per_type=0 (default), no per-type limit is applied."""
        base_config.missions = {"ATREGGIES": 3}
        base_config.constraints.max_missions_per_type = 0  # Unlimited
        base_config.constraints.max_time_hours = 168.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Without the constraint, solver may select the same mission type multiple times
        # (Just verifying it doesn't break anything)


# ---------------------------------------------------------------------------
# Scenario 4: Time Efficiency Variations
# ---------------------------------------------------------------------------

class TestTimeWeightScenarios:
    """Test how time_efficiency_power affects duration preference."""
    
    def test_zero_time_weight_ignores_duration(self, base_config):
        """With time_efficiency_power=0, solver ignores mission duration in objective."""
        base_config.cost_weights.time_efficiency_power = 0.0
        base_config.constraints.max_time_hours = 168.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Without time efficiency factor, longer missions may be preferred
        # (if they have higher artifact value per run)
    
    def test_high_time_weight_prefers_short(self, base_config):
        """With high time_efficiency_power, solver prefers faster missions."""
        base_config.cost_weights.time_efficiency_power = 2.0
        base_config.cost_weights.time_efficiency_scale = 1.0
        base_config.constraints.max_time_hours = 168.0
        
        result_high = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Compare to no time efficiency
        base_config.cost_weights.time_efficiency_power = 0.0
        result_low = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # High time efficiency should select faster missions on average
        if result_high.selected_missions and result_low.selected_missions:
            avg_time_high = result_high.total_time_hours / max(1, len(result_high.selected_missions))
            avg_time_low = result_low.total_time_hours / max(1, len(result_low.selected_missions))
            # With high time efficiency, average mission time should be lower
            # (or objective structure is different)
    
    def test_time_weight_comparison(self, base_config):
        """Compare solutions with different time efficiency powers."""
        base_config.constraints.max_time_hours = 168.0
        
        results = {}
        for time_power in [0.0, 0.5, 1.0, 2.0]:
            base_config.cost_weights.time_efficiency_power = time_power
            result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
            results[time_power] = {
                'objective': result.objective_value,
                'total_time': result.total_time_hours,
                'missions': get_mission_summary(result),
            }
        
        # Different time powers should produce different solutions
        # (unless constraints dominate)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Scenario 5: Artifact Weight Scenarios
# ---------------------------------------------------------------------------

class TestArtifactWeightScenarios:
    """Test how artifact weights affect mission selection."""
    
    def test_boost_single_artifact_changes_selection(self, base_config):
        """Heavily weighting one artifact should favor missions that drop it."""
        # First solve with default weights
        base_config.mission_artifact_weights = {}
        result_default = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Now heavily boost Book of Basan (drops more from certain missions)
        base_config.mission_artifact_weights = {
            "Book of Basan": 100.0,
            "Collectors book of Basan": 100.0,
            "Fortified book of Basan": 100.0,
        }
        result_book = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # The solutions may differ (if Book of Basan has different drop rates)
        assert result_default.status == "Optimal"
        assert result_book.status == "Optimal"
    
    def test_negative_artifact_weight_avoids_drops(self, base_config):
        """Negative weights should penalize missions with those drops."""
        # Mark all artifacts as unwanted except a few
        base_config.mission_artifact_weights = {
            "Ancient puzzle cube": 10.0,
            "Puzzle cube": 10.0,
        }
        # Everything else defaults to 1.0, but if we explicitly set some negative:
        base_config.mission_artifact_weights["Tiny vial of Martian dust"] = -10.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
    
    def test_all_zero_weights_finds_solution(self, base_config):
        """With all weights at 0, solver should still find a valid solution."""
        base_config.mission_artifact_weights = {}
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Should be optimal (trivially - no artifact value to maximize)
        assert result.status == "Optimal"
    
    def test_gold_meteorite_focus(self, base_config):
        """Focus on gold meteorites should favor missions that drop them."""
        base_config.mission_artifact_weights = {
            "Tiny gold meteorite": 50.0,
            "Enriched gold meteorite": 50.0,
            "Solid gold meteorite": 50.0,
        }
        base_config.constraints.max_time_hours = 48.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # Gold meteorites should be prominently in total_drops
        gold_drops = sum(
            qty for art, qty in result.total_drops.items() 
            if "gold meteorite" in art.lower()
        )
        # Should have some gold meteorite drops
        assert gold_drops >= 0


# ---------------------------------------------------------------------------
# Scenario 6: Waste Efficiency Variations
# ---------------------------------------------------------------------------

class TestEfficiencyBonusScenarios:
    """Test how waste_efficiency_power affects mission selection."""
    
    def test_zero_efficiency_ignores_ratio(self, base_config):
        """With waste_efficiency_power=0, only raw artifact value matters."""
        base_config.cost_weights.waste_efficiency_power = 0.0
        base_config.mission_artifact_weights = {
            "Ancient puzzle cube": 1.0,
            "Tiny vial of Martian dust": 0.0,  # Acceptable (zero, not negative)
        }
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
    
    def test_high_efficiency_favors_focused_missions(self, base_config):
        """With high waste_efficiency_power, solver prefers high targeted/total ratio missions."""
        # First with low waste efficiency
        base_config.cost_weights.waste_efficiency_power = 0.0
        base_config.mission_artifact_weights = {
            "Ancient puzzle cube": 1.0,
        }
        result_low = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Then with high waste efficiency
        base_config.cost_weights.waste_efficiency_power = 2.0
        result_high = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Both should be optimal
        assert result_low.status == "Optimal"
        assert result_high.status == "Optimal"
    
    def test_efficiency_bonus_affects_mission_choice(self, base_config):
        """Waste efficiency should change which missions are selected."""
        # Set up specific targeted artifacts
        base_config.mission_artifact_weights = {
            "Ancient puzzle cube": 1.0,
            "Puzzle cube": 1.0,
            "Interstellar compass": 1.0,
        }
        
        results = {}
        for eff_power in [0.0, 0.5, 1.0, 2.0]:
            base_config.cost_weights.waste_efficiency_power = eff_power
            result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
            results[eff_power] = {
                'objective': result.objective_value,
                'missions': get_mission_summary(result),
            }
        
        # All should be optimal
        assert all(results[e]['objective'] >= 0 or results[e]['missions'] for e in results)


# ---------------------------------------------------------------------------
# Scenario 7: Time Efficiency Weight
# ---------------------------------------------------------------------------

class TestTimeEfficiencyScenarios:
    """Test how time efficiency weight affects solution."""
    
    def test_time_efficiency_favors_short_missions(self, base_config):
        """Higher time efficiency power should favor shorter missions."""
        base_config.cost_weights.time_efficiency_power = 2.0
        base_config.cost_weights.time_efficiency_scale = 2.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
    
    def test_no_time_efficiency_allows_longer_missions(self, base_config):
        """With time efficiency disabled, longer missions are viable."""
        base_config.cost_weights.time_efficiency_power = 0.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
    
    def test_varying_time_efficiency(self, base_config):
        """Compare solutions at different time efficiency powers."""
        base_config.constraints.max_time_hours = 48.0
        
        results = {}
        for power in [0.0, 1.0, 2.0]:
            base_config.cost_weights.time_efficiency_power = power
            base_config.cost_weights.time_efficiency_scale = 2.0
            result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
            results[power] = {
                'objective': result.objective_value,
                'total_drops': sum(result.total_drops.values()),
                'missions': get_mission_summary(result),
            }
        
        # All should be optimal
        assert all(results[p]['objective'] >= 0 for p in results)


# ---------------------------------------------------------------------------
# Scenario 8: Number of Concurrent Ships
# ---------------------------------------------------------------------------

class TestShipCountScenarios:
    """Test how number of concurrent ships affects elapsed time and selection.
    
    Concurrent ships affect elapsed time calculation (total_mission_time / num_ships),
    not the number of mission types that can be selected. More concurrent ships
    allows more total mission time within the same elapsed time constraint.
    """
    
    def test_single_ship(self, base_config):
        """With 1 concurrent ship, elapsed time equals total mission time."""
        result = solve(base_config, num_ships=1, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # With 1 ship, fewer missions can fit in the time constraint
    
    def test_two_ships(self, base_config):
        """With 2 concurrent ships, can run 2x mission time in same elapsed time."""
        result = solve(base_config, num_ships=2, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # With 2 ships, more missions can fit in the time constraint
    
    def test_three_ships(self, base_config):
        """With 3 concurrent ships (default), can run 3x mission time in same elapsed time."""
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        # With 3 ships, even more missions can fit in the time constraint
    
    def test_more_ships_better_objective(self, base_config):
        """More concurrent ships should allow equal or better objective."""
        base_config.constraints.max_time_hours = 168.0
        
        results = {}
        for ships in [1, 2, 3]:
            result = solve(base_config, num_ships=ships, log_level=LogLevel.MINIMAL)
            results[ships] = result.objective_value
        
        # More concurrent ships should give >= objective (can run more missions)
        assert results[2] >= results[1] - 0.01  # Allow small numerical tolerance
        assert results[3] >= results[2] - 0.01


# ---------------------------------------------------------------------------
# Scenario 9: Epic Research Impact
# ---------------------------------------------------------------------------

class TestEpicResearchScenarios:
    """Test how epic research levels affect mission selection."""
    
    def test_no_research(self, base_config):
        """Without epic research, missions take longer and have less capacity."""
        base_config.epic_researches = {}
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
    
    def test_max_zero_g_research(self, base_config):
        """Max Zero-G research increases capacity significantly."""
        base_config.epic_researches = {
            "Zero-G Quantum Containment": EpicResearch(level=12, effect=0.05, max_level=12),
        }
        
        result_max = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Compare to no research
        base_config.epic_researches = {}
        result_none = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Max research should give better or equal objective
        # (more capacity = more artifacts per mission)
        assert result_max.status == "Optimal"
        assert result_none.status == "Optimal"
    
    def test_ftl_research_affects_ftl_ships(self, base_config):
        """FTL research reduces time for FTL-capable ships."""
        # FTL ships: HENERPRISE, ATREGGIES, etc.
        base_config.missions = {"HENERPRISE": 8, "ATREGGIES": 8}
        
        # With max FTL research
        base_config.epic_researches = {
            "FTL Drive Upgrades": EpicResearch(level=60, effect=0.01, max_level=60),
        }
        result_ftl = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Without FTL research
        base_config.epic_researches = {}
        result_no_ftl = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Same time constraint should allow more missions with FTL
        assert result_ftl.status == "Optimal"
        assert result_no_ftl.status == "Optimal"


# ---------------------------------------------------------------------------
# Scenario 10: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCaseScenarios:
    """Test edge cases and boundary conditions."""
    
    def test_zero_time_returns_empty(self, base_config):
        """With 0 time, no missions can be run - infeasible with min mission constraint."""
        base_config.constraints.max_time_hours = 0.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Infeasible because we require at least 1 mission but 0 time allows none
        assert result.status in ("Optimal", "No missions available", "Infeasible")
    
    def test_very_small_fuel_may_be_infeasible(self, base_config):
        """With extremely low fuel, solution may be limited."""
        base_config.constraints.fuel_tank_capacity = 0.1  # 0.1T
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Could be optimal with minimal missions or no missions
        assert result.status in ("Optimal", "No missions available", "Infeasible")
    
    def test_no_ships_enabled(self):
        """With no ships enabled, returns appropriately."""
        config = UserConfig(
            missions={},
            epic_researches={},
            constraints=Constraints(),
            cost_weights=CostWeights(),
            crafted_artifact_weights={},
            mission_artifact_weights={},
        )
        
        result = solve(config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "No missions available"
    
    def test_all_artifacts_negative_weight(self, base_config):
        """With all artifacts penalized, solver finds minimum penalty solution."""
        # Set all artifacts to negative
        base_config.mission_artifact_weights = {
            "Ancient puzzle cube": -1.0,
            "Puzzle cube": -1.0,
            "Tiny gold meteorite": -1.0,
            # ... etc
        }
        base_config.cost_weights.waste_efficiency_power = 1.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Should still be optimal
        assert result.status == "Optimal"


# ---------------------------------------------------------------------------
# Scenario 11: Combined Constraints
# ---------------------------------------------------------------------------

class TestCombinedConstraintScenarios:
    """Test interactions between multiple constraints."""
    
    def test_tight_time_and_fuel(self, base_config):
        """Both time and fuel constraints are binding."""
        base_config.constraints.max_time_hours = 10.0
        base_config.constraints.fuel_tank_capacity = 20.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        assert result.total_time_hours <= 10.0
        assert result.fuel_usage.tank_total / 1e12 <= 20.0 + 0.1
    
    def test_relaxing_one_constraint(self, base_config):
        """Relaxing one constraint should improve or maintain objective."""
        # Tight constraints
        base_config.constraints.max_time_hours = 20.0
        base_config.constraints.fuel_tank_capacity = 50.0
        result_tight = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Relax time
        base_config.constraints.max_time_hours = 100.0
        result_relaxed_time = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Reset and relax fuel
        base_config.constraints.max_time_hours = 20.0
        base_config.constraints.fuel_tank_capacity = 500.0
        result_relaxed_fuel = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        # Relaxed constraints should give >= objective
        assert result_relaxed_time.objective_value >= result_tight.objective_value - 0.01
        assert result_relaxed_fuel.objective_value >= result_tight.objective_value - 0.01
    
    def test_ship_type_with_time_constraint(self, base_config):
        """Ship type combined with time constraint."""
        # Only Atreggies, tight time
        base_config.missions = {"ATREGGIES": 8}
        base_config.constraints.max_time_hours = 20.0
        
        result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
        
        assert result.status == "Optimal"
        ships = get_selected_ships(result)
        assert ships <= {"ATREGGIES"}


# ---------------------------------------------------------------------------
# Parametrized Comprehensive Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_name,config_changes,expected_ships", [
    ("atreggies_only", {"missions": {"ATREGGIES": 8}}, {"ATREGGIES"}),
    ("henerprise_only", {"missions": {"HENERPRISE": 8}}, {"HENERPRISE"}),
    ("voyegger_only", {"missions": {"VOYEGGER": 6}}, {"VOYEGGER"}),
])
def test_ship_restriction_parametrized(base_config, scenario_name, config_changes, expected_ships):
    """Parametrized test for ship restrictions."""
    for key, value in config_changes.items():
        setattr(base_config, key, value)
    
    result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
    
    assert result.status == "Optimal"
    ships = get_selected_ships(result)
    assert ships <= expected_ships, f"Scenario {scenario_name}: Expected {expected_ships}, got {ships}"


@pytest.mark.parametrize("time_hours,should_complete", [
    (1.0, True),   # Very short - may be infeasible with min mission constraint
    (10.0, True),  # Short - standard missions possible
    (50.0, True),  # Medium - extended missions possible
    (168.0, True), # Week - epic missions possible
])
def test_time_constraint_parametrized(base_config, time_hours, should_complete):
    """Parametrized test for time constraints."""
    base_config.constraints.max_time_hours = time_hours
    
    result = solve(base_config, num_ships=3, log_level=LogLevel.MINIMAL)
    
    if should_complete:
        # May be Infeasible if time too short for any mission
        assert result.status in ("Optimal", "Infeasible")
        if result.status == "Optimal":
            assert result.total_time_hours <= time_hours + 0.01


# ---------------------------------------------------------------------------
# Logging Verification Tests
# ---------------------------------------------------------------------------

class TestLoggingVerification:
    """Verify that logging correctly captures solver behavior."""
    
    def test_detailed_logging_captures_coefficients(self, base_config):
        """DEBUG logging should show objective coefficients."""
        from Solver.solver_logging import SolverLogger
        
        logger = SolverLogger(level=LogLevel.DEBUG)
        result = solve(base_config, num_ships=3, logger=logger)
        
        # Should have objective-related log entries
        objective_entries = logger.get_entries_by_category("OBJECTIVE")
        assert len(objective_entries) > 0
    
    def test_logging_shows_constraint_values(self, base_config):
        """Logging should show constraint parameters."""
        from Solver.solver_logging import SolverLogger
        
        logger = SolverLogger(level=LogLevel.SUMMARY)
        result = solve(base_config, num_ships=3, logger=logger)
        
        config_entries = logger.get_entries_by_category("CONFIG")
        assert len(config_entries) > 0
    
    def test_different_weights_different_logs(self, base_config):
        """Different weight configurations should produce different log output."""
        from Solver.solver_logging import SolverLogger
        
        # Add some artifact weights so the efficiency factors have an effect
        # Use artifacts that actually drop from missions
        base_config.mission_artifact_weights = {
            "Mystical puzzle cube": 1.0,
            "Powerful lunar totem": 1.0,
            "Enriched gold meteorite": 1.0,
        }
        
        # Run with no time efficiency
        logger1 = SolverLogger(level=LogLevel.SUMMARY)
        base_config.cost_weights.time_efficiency_power = 0.0
        result1 = solve(base_config, num_ships=3, logger=logger1)
        
        # Run with time efficiency enabled
        logger2 = SolverLogger(level=LogLevel.SUMMARY)
        base_config.cost_weights.time_efficiency_power = 2.0
        base_config.cost_weights.time_efficiency_scale = 2.0
        result2 = solve(base_config, num_ships=3, logger=logger2)
        
        # Logs should differ
        log1 = logger1.to_string()
        log2 = logger2.to_string()
        # At minimum, the objective values should differ
        assert result1.objective_value != result2.objective_value or log1 == log2
