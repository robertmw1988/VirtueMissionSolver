"""Bill of Materials (BOM) rollup engine for artifact crafting.

This module provides functionality to:
1. Load recipe data from eiafx-data.json
2. Build a dependency graph for crafting
3. Flatten BOM requirements using topological sort
4. Roll up inventory based on crafting priorities with shared ingredient allocation
5. Handle partial crafts when ingredients are insufficient
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from .resources import get_resource_path

# Paths - using resource helper for PyInstaller compatibility
EIAFX_DATA_PATH = get_resource_path("Wasmegg/eiafx-data.json")

# Minimum ingredient threshold for partial crafts
DEFAULT_INGREDIENT_THRESHOLD = 0.001

# Rarity suffixes used in display names
RARITY_SUFFIXES = frozenset({"(Rare)", "(Epic)", "(Legendary)"})


@dataclass
class Recipe:
    """Represents a crafting recipe for an artifact tier."""
    artifact_id: str
    artifact_name: str
    ingredients: List[Tuple[str, str, int]]  # [(id, name, count), ...]
    
    def __hash__(self) -> int:
        return hash(self.artifact_id)
    

@dataclass
class BOMRequirement:
    """Represents total requirements for a single ingredient after BOM expansion."""
    artifact_id: str
    artifact_name: str
    quantity: float
    is_base_ingredient: bool = False  # True if this is a non-craftable base material


@dataclass
class RollupResult:
    """Result of a BOM rollup operation."""
    # Artifacts successfully crafted (id -> quantity)
    crafted: Dict[str, float] = field(default_factory=dict)
    # Base ingredients consumed (id -> quantity)
    consumed: Dict[str, float] = field(default_factory=dict)
    # Remaining inventory after rollup (id -> quantity)
    remaining: Dict[str, float] = field(default_factory=dict)
    # Shortfall for ingredients that were insufficient (id -> shortfall amount)
    shortfall: Dict[str, float] = field(default_factory=dict)
    # Partial craft progress (target_id -> fraction completed toward next craft)
    partial_progress: Dict[str, float] = field(default_factory=dict)


class BOMEngine:
    """
    Bill of Materials engine for artifact crafting.
    
    Loads recipe data, builds dependency graphs, and performs BOM rollup
    calculations with priority-based allocation of shared ingredients.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the BOM engine.
        
        Parameters
        ----------
        data_path : Path, optional
            Path to eiafx-data.json. Defaults to Wasmegg/eiafx-data.json.
        """
        self._data_path = data_path or EIAFX_DATA_PATH
        self._recipes: Dict[str, Recipe] = {}  # artifact_id -> Recipe
        self._id_to_name: Dict[str, str] = {}  # artifact_id -> display name
        self._name_to_id: Dict[str, str] = {}  # display name -> artifact_id
        self._base_name_to_id: Dict[str, str] = {}  # base name (no rarity) -> artifact_id
        self._craftable: Set[str] = set()  # set of craftable artifact IDs
        self._dependencies: Dict[str, List[str]] = {}  # artifact_id -> [dependency_ids]
        self._dependents: Dict[str, List[str]] = {}  # artifact_id -> [dependent_ids]
        
        # Caches for performance (populated lazily)
        self._flatten_bom_cache: Dict[str, Dict[str, float]] = {}  # artifact_id -> base requirements
        self._ingredient_ratios_cache: Dict[str, Dict[str, float]] = {}  # target_name -> ratios
        
        self._load_data()
        self._build_dependency_graph()
    
    def _load_data(self) -> None:
        """Load recipe data from eiafx-data.json."""
        if not self._data_path.exists():
            raise FileNotFoundError(f"eiafx-data.json not found at {self._data_path}")
        
        with self._data_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        
        artifact_families = data.get("artifact_families", [])
        
        for family in artifact_families:
            tiers = family.get("tiers", [])
            for tier in tiers:
                artifact_id = tier.get("id", "")
                artifact_name = tier.get("name", "")
                craftable = tier.get("craftable", False)
                
                if not artifact_id:
                    continue
                
                # Store name mappings
                self._id_to_name[artifact_id] = artifact_name
                self._name_to_id[artifact_name] = artifact_id
                self._base_name_to_id[artifact_name] = artifact_id
                
                # Also map rarity variants to the same base ID
                for rarity in ["Rare", "Epic", "Legendary"]:
                    rarity_name = f"{artifact_name} ({rarity})"
                    self._name_to_id[rarity_name] = artifact_id
                
                if craftable:
                    self._craftable.add(artifact_id)
                    recipe_data = tier.get("recipe")
                    if recipe_data:
                        ingredients = []
                        for ing in recipe_data.get("ingredients", []):
                            ing_id = ing.get("id", "")
                            ing_name = ing.get("name", "")
                            ing_count = ing.get("count", 0)
                            if ing_id and ing_count > 0:
                                ingredients.append((ing_id, ing_name, ing_count))
                        
                        self._recipes[artifact_id] = Recipe(
                            artifact_id=artifact_id,
                            artifact_name=artifact_name,
                            ingredients=ingredients,
                        )
    
    def _build_dependency_graph(self) -> None:
        """Build the dependency graph from recipes."""
        for artifact_id, recipe in self._recipes.items():
            deps = [ing_id for ing_id, _, _ in recipe.ingredients]
            self._dependencies[artifact_id] = deps
            
            for dep_id in deps:
                if dep_id not in self._dependents:
                    self._dependents[dep_id] = []
                self._dependents[dep_id].append(artifact_id)
    
    def get_ingredient_ratios(
        self,
        target_name: str,
    ) -> Dict[str, float]:
        """
        Get the ratio of each base ingredient needed per target artifact.
        
        This is used for building ratio constraints in the LP solver.
        The ratios tell us how many of each ingredient are needed per craft.
        Results are cached for performance.
        
        Parameters
        ----------
        target_name : str
            Display name of the target artifact.
        
        Returns
        -------
        dict
            Mapping of display_name -> quantity needed per target.
            Empty dict if target not found.
        """
        # Check cache first
        if target_name in self._ingredient_ratios_cache:
            return self._ingredient_ratios_cache[target_name]
        
        target_id = self._name_to_id.get(target_name)
        if not target_id:
            return {}
        
        base_requirements = self.flatten_bom(target_id, 1.0)
        
        # Convert to display names
        result = {
            self._id_to_name.get(ing_id, ing_id): qty
            for ing_id, qty in base_requirements.items()
        }
        
        # Cache the result
        self._ingredient_ratios_cache[target_name] = result
        return result
    
    def get_base_equivalence(
        self,
        artifact_name: str,
    ) -> Dict[str, float]:
        """
        Get how many base ingredients this artifact is equivalent to.
        
        For example, if artifact A requires 7 of ingredient B to craft,
        then getting 1 of A is equivalent to getting 7 of B.
        
        Parameters
        ----------
        artifact_name : str
            Display name of the artifact.
        
        Returns
        -------
        dict
            Mapping of base_ingredient_name -> equivalence count.
            Returns {artifact_name: 1.0} if artifact has no recipe (is base).
        """
        artifact_id = self._name_to_id.get(artifact_name)
        if not artifact_id:
            return {artifact_name: 1.0}
        
        # Flatten BOM to get base requirements
        base_reqs = self.flatten_bom(artifact_id, 1.0)
        
        if not base_reqs:
            # This is a base ingredient itself
            return {artifact_name: 1.0}
        
        # Convert to display names
        return {
            self._id_to_name.get(ing_id, ing_id): qty
            for ing_id, qty in base_reqs.items()
        }
    
    def get_all_contributors_for_target(
        self,
        target_name: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get all artifacts that contribute to crafting a target, with their base equivalence.
        
        This returns a mapping from each contributing artifact to its base ingredient
        equivalence. For example, for target T requiring base ingredients A and B:
        - "Adequate X" (crafted from 7 A) -> {A: 7}
        - "Perfect X" (crafted from 7 Adequate) -> {A: 49}
        - "A" itself -> {A: 1}
        
        Parameters
        ----------
        target_name : str
            Display name of the target artifact.
        
        Returns
        -------
        dict
            Mapping of artifact_name -> {base_ingredient_name -> equivalence}.
        """
        target_id = self._name_to_id.get(target_name)
        if not target_id:
            return {}
        
        # Get all artifacts in the dependency tree
        all_nodes: Set[str] = set()
        queue = deque([target_id])
        
        while queue:
            node = queue.popleft()
            if node in all_nodes:
                continue
            all_nodes.add(node)
            
            for dep_id in self._dependencies.get(node, []):
                queue.append(dep_id)
        
        # For each node, get its base equivalence
        result: Dict[str, Dict[str, float]] = {}
        for node_id in all_nodes:
            node_name = self._id_to_name.get(node_id, node_id)
            base_reqs = self.flatten_bom(node_id, 1.0)
            
            if base_reqs:
                result[node_name] = {
                    self._id_to_name.get(ing_id, ing_id): qty
                    for ing_id, qty in base_reqs.items()
                }
            else:
                # This is a base ingredient
                result[node_name] = {node_name: 1.0}
        
        return result
    
    def get_ingredient_ratios_for_targets(
        self,
        crafting_weights: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Get ingredient ratios for all weighted targets.
        
        Parameters
        ----------
        crafting_weights : dict
            Mapping of display name -> weight for crafted artifacts.
        
        Returns
        -------
        dict
            Mapping of target_name -> {ingredient_name -> qty_per_target}
        """
        result: Dict[str, Dict[str, float]] = {}
        
        for target_name, weight in crafting_weights.items():
            if weight <= 0:
                continue
            ratios = self.get_ingredient_ratios(target_name)
            if ratios:
                result[target_name] = ratios
        
        return result

    def calculate_ingredient_values(
        self,
        crafting_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate the value of each ingredient based on what it can craft.
        
        The value of an ingredient is proportional to:
        target_value / base_ingredients_needed_per_target
        
        This means rare ingredients (needed in smaller quantities) have 
        HIGHER value per unit, which encourages the solver to get them.
        
        For example, if target X worth 100 needs:
        - 1000 of ingredient A -> value_A = 100/1000 = 0.1 per unit
        - 100 of ingredient B -> value_B = 100/100 = 1.0 per unit
        
        This way, getting 1 of B is worth as much as getting 10 of A,
        which properly reflects the recipe ratio.
        
        Parameters
        ----------
        crafting_weights : dict
            Mapping of display name -> weight for crafted artifacts.
            Only artifacts with weight > 0 are considered targets.
        
        Returns
        -------
        dict
            Mapping of artifact_id -> derived value per unit.
        """
        # Convert weights to IDs and filter positive weights
        target_values: Dict[str, float] = {}
        for name, weight in crafting_weights.items():
            if weight <= 0:
                continue
            artifact_id = self._name_to_id.get(name)
            if artifact_id:
                target_values[artifact_id] = max(
                    target_values.get(artifact_id, 0.0), weight
                )
        
        if not target_values:
            return {}
        
        ingredient_values: Dict[str, float] = {}
        
        # Assign target values to the targets themselves
        for artifact_id, value in target_values.items():
            ingredient_values[artifact_id] = value
        
        # For each target, calculate value per ingredient
        # 
        # Key insight: We want ingredients to be as valuable as direct mission targets.
        # If the target is worth W=1.0, and we need 99 of ingredient A, then:
        #   - Old approach: value_per_A = 1.0/99 = 0.01 (too small to compete!)
        #   - New approach: value_per_A = 1.0 (each ingredient equally valuable)
        #
        # The ratio-balancing constraints in the solver ensure we collect the
        # RIGHT PROPORTIONS. The ingredient values just need to make the solver
        # WANT to collect ingredients at all.
        #
        # So we give each needed ingredient a value equal to the target value.
        # This ensures missions that drop ingredients are preferred.
        #
        for target_id, target_value in target_values.items():
            # Get flattened BOM - which base ingredients are needed
            base_requirements = self.flatten_bom(target_id, 1.0)
            
            if not base_requirements:
                continue
            
            # Each ingredient type gets value = target_value
            # The solver's ratio-balancing will ensure correct proportions
            for ing_id, qty_needed in base_requirements.items():
                if qty_needed > 0:
                    # All ingredients equally valuable - ratio constraints handle proportions
                    ingredient_values[ing_id] = ingredient_values.get(ing_id, 0.0) + target_value
            
            # Also value intermediate craftables based on their recipe
            # Walk up from base ingredients
            visited: Set[str] = set(base_requirements.keys())
            to_process = list(base_requirements.keys())
            
            while to_process:
                ing_id = to_process.pop()
                # Check what this ingredient can be used to craft
                for dependent_id in self._dependents.get(ing_id, []):
                    if dependent_id in visited:
                        continue
                    if dependent_id not in target_values and dependent_id != target_id:
                        # This is an intermediate craftable
                        recipe = self._recipes.get(dependent_id)
                        if recipe:
                            # Value = parent_value based on how many we need
                            # Find how many of this intermediate per target
                            # by seeing what parent needs
                            parent_reqs = self.flatten_bom(target_id, 1.0)
                            # Intermediate value is proportional to target value
                            # divided by how many intermediates we can make
                            # from the limiting ingredient
                            min_craftable = float('inf')
                            for sub_id, _, sub_count in recipe.ingredients:
                                if sub_id in parent_reqs and sub_count > 0:
                                    craftable = parent_reqs.get(sub_id, 0) / sub_count
                                    min_craftable = min(min_craftable, craftable)
                            
                            if min_craftable < float('inf') and min_craftable > 0:
                                intermediate_value = target_value / min_craftable
                                ingredient_values[dependent_id] = ingredient_values.get(dependent_id, 0.0) + intermediate_value
                        
                        visited.add(dependent_id)
                        to_process.append(dependent_id)
        
        return ingredient_values
    
    def calculate_ingredient_values_by_name(
        self,
        crafting_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Same as calculate_ingredient_values but returns display names as keys.
        
        Parameters
        ----------
        crafting_weights : dict
            Mapping of display name -> weight for crafted artifacts.
        
        Returns
        -------
        dict
            Mapping of display_name -> derived value as an ingredient.
        """
        id_values = self.calculate_ingredient_values(crafting_weights)
        return {
            self._id_to_name.get(k, k): v
            for k, v in id_values.items()
        }
    
    def get_recipe(self, artifact_id: str) -> Optional[Recipe]:
        """Get the recipe for an artifact by ID."""
        return self._recipes.get(artifact_id)
    
    def is_craftable(self, artifact_id: str) -> bool:
        """Check if an artifact is craftable."""
        return artifact_id in self._craftable
    
    def name_to_id(self, display_name: str) -> Optional[str]:
        """Convert a display name (with optional rarity) to artifact ID."""
        return self._name_to_id.get(display_name)
    
    def id_to_name(self, artifact_id: str) -> Optional[str]:
        """Convert an artifact ID to its display name."""
        return self._id_to_name.get(artifact_id)
    
    def strip_rarity(self, display_name: str) -> str:
        """
        Strip rarity suffix from a display name.
        
        Examples:
            "Puzzle cube (Epic)" -> "Puzzle cube"
            "Ancient puzzle cube" -> "Ancient puzzle cube"
        """
        for suffix in RARITY_SUFFIXES:
            if display_name.endswith(suffix):
                return display_name[: -len(suffix)].rstrip()
        return display_name
    
    def normalize_inventory(
        self,
        inventory: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize inventory by aggregating rarity variants to base artifact IDs.
        
        All rarities are treated as equivalent (demoted to common) for crafting.
        
        Parameters
        ----------
        inventory : dict
            Mapping of display name -> quantity (may include rarity variants)
        
        Returns
        -------
        dict
            Mapping of artifact_id -> aggregated quantity
        """
        normalized: Dict[str, float] = defaultdict(float)
        
        for name, qty in inventory.items():
            if qty <= 0:
                continue
            
            # Try direct lookup first
            artifact_id = self._name_to_id.get(name)
            if artifact_id:
                normalized[artifact_id] += qty
            else:
                # Try stripping rarity and looking up
                base_name = self.strip_rarity(name)
                artifact_id = self._base_name_to_id.get(base_name)
                if artifact_id:
                    normalized[artifact_id] += qty
        
        return dict(normalized)
    
    def topological_sort(self, target_ids: Set[str]) -> List[str]:
        """
        Perform topological sort on target artifacts and their dependencies.
        
        Returns artifacts in order from leaves (base ingredients) to roots (final targets).
        This ensures we process dependencies before dependents during rollup.
        
        Parameters
        ----------
        target_ids : set
            Set of artifact IDs to include in the sort
        
        Returns
        -------
        list
            Artifact IDs in topologically sorted order (dependencies first)
        """
        # Collect all nodes needed (targets and their transitive dependencies)
        all_nodes: Set[str] = set()
        queue = deque(target_ids)
        
        while queue:
            node = queue.popleft()
            if node in all_nodes:
                continue
            all_nodes.add(node)
            
            # Add dependencies
            for dep_id in self._dependencies.get(node, []):
                if dep_id not in all_nodes:
                    queue.append(dep_id)
        
        # Kahn's algorithm for topological sort
        # Build in-degree map (only for nodes in our subgraph)
        in_degree: Dict[str, int] = {node: 0 for node in all_nodes}
        for node in all_nodes:
            for dep_id in self._dependencies.get(node, []):
                if dep_id in all_nodes:
                    in_degree[node] += 1
        
        # Start with nodes that have no dependencies
        ready = deque([node for node, degree in in_degree.items() if degree == 0])
        sorted_order: List[str] = []
        
        while ready:
            node = ready.popleft()
            sorted_order.append(node)
            
            # Reduce in-degree for dependents
            for dependent in self._dependents.get(node, []):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        ready.append(dependent)
        
        return sorted_order
    
    def _flatten_bom_uncached(
        self,
        artifact_id: str,
    ) -> Dict[str, float]:
        """
        Internal: Flatten BOM for quantity=1.0, used for caching.
        
        Parameters
        ----------
        artifact_id : str
            The artifact ID to expand
        
        Returns
        -------
        dict
            Mapping of base ingredient ID -> quantity required for 1 craft
        """
        requirements: Dict[str, float] = defaultdict(float)
        
        # Use iterative approach with stack to avoid recursion limits
        stack: List[Tuple[str, float]] = [(artifact_id, 1.0)]
        
        while stack:
            current_id, current_qty = stack.pop()
            
            recipe = self._recipes.get(current_id)
            if not recipe:
                # Base ingredient (not craftable)
                requirements[current_id] += current_qty
                continue
            
            # Expand recipe
            for ing_id, _, ing_count in recipe.ingredients:
                needed = current_qty * ing_count
                stack.append((ing_id, needed))
        
        return dict(requirements)
    
    def flatten_bom(
        self,
        artifact_id: str,
        quantity: float = 1.0,
    ) -> Dict[str, float]:
        """
        Flatten the BOM for a single artifact, returning total base ingredient requirements.
        
        Recursively expands the recipe tree to find all base (non-craftable) ingredients.
        Results for quantity=1.0 are cached for performance.
        
        Parameters
        ----------
        artifact_id : str
            The artifact ID to expand
        quantity : float
            Number of artifacts to craft
        
        Returns
        -------
        dict
            Mapping of base ingredient ID -> total quantity required
        """
        # For quantity=1.0, use cached version
        if quantity == 1.0:
            if artifact_id not in self._flatten_bom_cache:
                self._flatten_bom_cache[artifact_id] = self._flatten_bom_uncached(artifact_id)
            # Return a copy to prevent mutation of cached data
            return self._flatten_bom_cache[artifact_id].copy()
        
        # For other quantities, scale from cached unit requirements
        if artifact_id not in self._flatten_bom_cache:
            self._flatten_bom_cache[artifact_id] = self._flatten_bom_uncached(artifact_id)
        
        unit_reqs = self._flatten_bom_cache[artifact_id]
        return {k: v * quantity for k, v in unit_reqs.items()}
    
    def calculate_craft_ratios(
        self,
        crafting_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate crafting priority ratios from weights.
        
        Only includes artifacts with weight > 0 (artifacts with <= 0 are ingredient-only).
        Ratios are normalized so they sum to 1.0 within each shared ingredient group.
        
        Parameters
        ----------
        crafting_weights : dict
            Mapping of display name -> weight (from craftedArtifactTargetWeights)
        
        Returns
        -------
        dict
            Mapping of artifact_id -> normalized priority ratio
        """
        # Convert to IDs and filter out non-positive weights
        id_weights: Dict[str, float] = {}
        for name, weight in crafting_weights.items():
            if weight <= 0:
                continue
            artifact_id = self._name_to_id.get(name)
            if artifact_id and artifact_id in self._craftable:
                # Aggregate weights across rarities to the same ID
                if artifact_id not in id_weights:
                    id_weights[artifact_id] = 0.0
                id_weights[artifact_id] = max(id_weights[artifact_id], weight)
        
        # Normalize to sum to 1.0
        total = sum(id_weights.values())
        if total <= 0:
            return {}
        
        return {k: v / total for k, v in id_weights.items()}
    
    def _calculate_max_crafts_for_target(
        self,
        target_id: str,
        inventory: Dict[str, float],
        ingredient_threshold: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate maximum number of a target that can be crafted from inventory.
        
        Works backwards from the target, considering both direct ingredients
        and craftable intermediates in the inventory.
        
        Parameters
        ----------
        target_id : str
            The artifact ID to craft
        inventory : dict
            Current inventory (artifact_id -> quantity)
        ingredient_threshold : float
            Minimum quantity threshold
        
        Returns
        -------
        tuple
            (max_crafts, ingredients_needed) where ingredients_needed maps
            each required ingredient to quantity per craft (only base/limiting ingredients)
        """
        # Get the recipe for the target
        recipe = self._recipes.get(target_id)
        if not recipe:
            # Can't craft this - it's not a craftable item
            return 0.0, {}
        
        # Get topological order for this target (dependencies first)
        sorted_ids = self.topological_sort({target_id})
        
        # Work backwards to compute what ingredients we need per target craft
        # Start with needing 1 of the target (to calculate ratios)
        need: Dict[str, float] = {target_id: 1.0}
        
        # Track which items we actually need from inventory 
        # (either base ingredients or intermediates we need to consume)
        actual_inventory_needs: Dict[str, float] = {}
        
        # Process in reverse topological order (target first, base last)
        for craft_id in reversed(sorted_ids):
            if craft_id not in need:
                continue
            
            qty_needed = need[craft_id]
            
            # Skip the target itself - we're calculating what we need TO CRAFT it
            if craft_id == target_id:
                craft_recipe = self._recipes.get(craft_id)
                if craft_recipe:
                    for ing_id, _, ing_count in craft_recipe.ingredients:
                        if ing_count <= 0:
                            continue
                        need[ing_id] = need.get(ing_id, 0.0) + qty_needed * ing_count
                continue
            
            # Check if we have this item in inventory
            available = inventory.get(craft_id, 0.0)
            
            # Use what we have first
            from_inventory = min(available, qty_needed)
            if from_inventory > 0:
                actual_inventory_needs[craft_id] = from_inventory
            
            # Need to craft the rest
            to_craft = qty_needed - from_inventory
            
            if to_craft <= ingredient_threshold:
                # Don't need to craft any more of this
                continue
            
            recipe_for_craft = self._recipes.get(craft_id)
            if not recipe_for_craft:
                # Can't craft this - it's a base ingredient we need more of
                if craft_id not in actual_inventory_needs:
                    actual_inventory_needs[craft_id] = 0.0
                # Record that we need this from inventory
                actual_inventory_needs[craft_id] = actual_inventory_needs.get(craft_id, 0.0) + to_craft
                continue
            
            # Add recipe requirements to our needs
            for ing_id, _, ing_count in recipe_for_craft.ingredients:
                if ing_count <= 0:
                    continue
                need[ing_id] = need.get(ing_id, 0.0) + to_craft * ing_count
        
        # Now calculate max crafts based on what we need from inventory vs what we have
        # The limiting factor is the ingredient with the smallest ratio
        max_crafts = float("inf")
        ingredients_per_craft: Dict[str, float] = {}
        
        for ing_id, total_needed in actual_inventory_needs.items():
            available = inventory.get(ing_id, 0.0)
            needed_per_craft = total_needed  # Since we calculated for 1 target
            ingredients_per_craft[ing_id] = needed_per_craft
            
            if needed_per_craft > ingredient_threshold:
                possible = available / needed_per_craft
                max_crafts = min(max_crafts, possible)
        
        if max_crafts == float("inf"):
            max_crafts = 0.0
        
        return max_crafts, ingredients_per_craft
    
    def rollup(
        self,
        inventory: Dict[str, float],
        crafting_weights: Dict[str, float],
        ingredient_threshold: float = DEFAULT_INGREDIENT_THRESHOLD,
    ) -> RollupResult:
        """
        Perform BOM rollup on inventory based on crafting priorities.
        
        This is the main rollup function that:
        1. Normalizes inventory (aggregates rarities)
        2. Determines target artifacts based on weights (weight > 0)
        3. Calculates ingredient requirements for each target (accounting for intermediates in inventory)
        4. Allocates shared ingredients proportionally by weight across targets
        5. Processes artifacts in topological order (dependencies first)
        6. Handles partial crafts when ingredients are insufficient
        
        Parameters
        ----------
        inventory : dict
            Current inventory as display_name -> quantity
        crafting_weights : dict
            Crafting priorities as display_name -> weight
            Artifacts with weight <= 0 are NOT crafted, only used as ingredients
        ingredient_threshold : float
            Minimum ingredient quantity to consider for crafting (default 0.001)
            Below this threshold, crafting is skipped and remaining is recorded
        
        Returns
        -------
        RollupResult
            Contains crafted quantities, consumed ingredients, remaining inventory,
            shortfalls, and partial craft progress
        """
        result = RollupResult()
        
        # Normalize inventory to artifact IDs
        working_inv = self.normalize_inventory(inventory)
        
        # Get craft targets (weight > 0) and their ratios
        craft_ratios = self.calculate_craft_ratios(crafting_weights)
        if not craft_ratios:
            # Nothing to craft, return inventory as-is
            result.remaining = working_inv.copy()
            return result
        
        target_ids = set(craft_ratios.keys())
        
        # Step 1: For each target, calculate its ingredient requirements
        # This accounts for intermediate items already in inventory
        target_requirements: Dict[str, Dict[str, float]] = {}  # target_id -> {ing_id -> qty_per_craft}
        target_max_unconstrained: Dict[str, float] = {}  # target_id -> max crafts if alone
        
        for target_id in target_ids:
            max_crafts, ing_per_craft = self._calculate_max_crafts_for_target(
                target_id, working_inv, ingredient_threshold
            )
            target_requirements[target_id] = ing_per_craft
            target_max_unconstrained[target_id] = max_crafts
        
        # Step 2: Find which ingredients are contested (needed by multiple targets)
        # Build: ingredient_id -> list of (target_id, qty_needed_per_craft)
        ingredient_demand: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for target_id, requirements in target_requirements.items():
            for ing_id, qty_per_craft in requirements.items():
                if qty_per_craft > 0:
                    ingredient_demand[ing_id].append((target_id, qty_per_craft))
        
        # Step 3: Allocate contested ingredients proportionally by target weight
        # For each target, calculate max crafts allowed based on its share
        target_max_crafts: Dict[str, float] = {t: target_max_unconstrained.get(t, 0.0) for t in target_ids}
        
        for ing_id, demand_list in ingredient_demand.items():
            if len(demand_list) <= 1:
                # No contention - single target uses all it can
                continue
            
            available = working_inv.get(ing_id, 0.0)
            if available < ingredient_threshold:
                # Not enough to matter
                for target_id, _ in demand_list:
                    target_max_crafts[target_id] = 0.0
                continue
            
            # Multiple targets share this ingredient - allocate by weight ratio
            total_weight = sum(craft_ratios.get(t, 0.0) for t, _ in demand_list)
            
            if total_weight <= 0:
                continue
            
            # Allocate ingredient to each target proportionally
            for target_id, qty_per_craft in demand_list:
                weight = craft_ratios.get(target_id, 0.0)
                if weight <= 0 or qty_per_craft <= 0:
                    continue
                
                # This target's share of the ingredient
                share_ratio = weight / total_weight
                allocated = available * share_ratio
                
                # How many can we craft with this allocation?
                max_from_ing = allocated / qty_per_craft
                target_max_crafts[target_id] = min(target_max_crafts[target_id], max_from_ing)
        
        # Step 4: Calculate allocations upfront based on original inventory
        # This ensures that all targets get their fair share regardless of processing order
        target_allocations: Dict[str, Dict[str, float]] = {}  # target_id -> {ing_id -> allocated_qty}
        
        for target_id in target_ids:
            craft_qty = target_max_crafts.get(target_id, 0.0)
            if craft_qty < ingredient_threshold:
                continue
            
            allocated_from_inventory: Dict[str, float] = {}
            for ing_id, qty_per_craft in target_requirements.get(target_id, {}).items():
                # Check if this is a contested ingredient
                if ing_id in ingredient_demand and len(ingredient_demand[ing_id]) > 1:
                    # Use proportional allocation from ORIGINAL inventory
                    total_weight = sum(craft_ratios.get(t, 0.0) for t, _ in ingredient_demand[ing_id])
                    if total_weight > 0:
                        share_ratio = craft_ratios.get(target_id, 0.0) / total_weight
                        allocated_from_inventory[ing_id] = working_inv.get(ing_id, 0.0) * share_ratio
                    else:
                        allocated_from_inventory[ing_id] = 0.0
                else:
                    # Not contested - allocate what this target needs
                    # Limited by craft_qty to avoid over-allocating
                    needed = craft_qty * qty_per_craft
                    allocated_from_inventory[ing_id] = min(working_inv.get(ing_id, 0.0), needed)
            
            target_allocations[target_id] = allocated_from_inventory
        
        # Step 5: Execute crafting for each target using pre-calculated allocations
        for target_id in target_ids:
            craft_qty = target_max_crafts.get(target_id, 0.0)
            
            if craft_qty < ingredient_threshold:
                # Record partial progress
                if craft_qty > 0:
                    result.partial_progress[target_id] = craft_qty
                continue
            
            # Get topological order for this target
            sorted_ids = self.topological_sort({target_id})
            craftable_ids = set(sorted_ids) & self._craftable
            
            # Use pre-calculated allocations
            allocated_from_inventory = target_allocations.get(target_id, {}).copy()
            
            # Process from base to target
            crafted_this_target: Dict[str, float] = {}  # Track intermediates created
            
            for craft_id in sorted_ids:
                if craft_id not in craftable_ids:
                    continue
                
                recipe = self._recipes.get(craft_id)
                if not recipe:
                    continue
                
                # Calculate how many we can craft
                max_crafts_here = float("inf")
                for ing_id, _, ing_count in recipe.ingredients:
                    if ing_count <= 0:
                        continue
                    # Available = allocated from inventory + any we've crafted in this chain
                    avail_from_inv = allocated_from_inventory.get(ing_id, 0.0)
                    avail_from_crafted = crafted_this_target.get(ing_id, 0.0)
                    available = avail_from_inv + avail_from_crafted
                    possible = available / ing_count
                    if possible < max_crafts_here:
                        max_crafts_here = possible
                
                if max_crafts_here < ingredient_threshold:
                    continue
                
                # Execute crafts
                for ing_id, _, ing_count in recipe.ingredients:
                    consumed = max_crafts_here * ing_count
                    # First consume from crafted intermediates
                    from_crafted = min(crafted_this_target.get(ing_id, 0.0), consumed)
                    crafted_this_target[ing_id] = crafted_this_target.get(ing_id, 0.0) - from_crafted
                    # Then from allocated inventory
                    from_inv = consumed - from_crafted
                    allocated_from_inventory[ing_id] = allocated_from_inventory.get(ing_id, 0.0) - from_inv
                    
                    result.consumed[ing_id] = result.consumed.get(ing_id, 0.0) + consumed
                
                # Add crafted items
                crafted_this_target[craft_id] = crafted_this_target.get(craft_id, 0.0) + max_crafts_here
                
                # Record if this is the target
                if craft_id == target_id:
                    result.crafted[target_id] = result.crafted.get(target_id, 0.0) + max_crafts_here
        
        # Calculate remaining inventory by deducting all allocations
        for target_id, allocations in target_allocations.items():
            for ing_id, allocated_qty in allocations.items():
                working_inv[ing_id] = working_inv.get(ing_id, 0.0) - allocated_qty
        
        # Record remaining inventory
        for k, v in working_inv.items():
            if v > ingredient_threshold:
                result.remaining[k] = result.remaining.get(k, 0.0) + v
        
        # Add crafted targets to remaining
        for target_id, qty in result.crafted.items():
            if qty > ingredient_threshold:
                result.remaining[target_id] = result.remaining.get(target_id, 0.0) + qty
        
        return result
    
    def rollup_with_display_names(
        self,
        inventory: Dict[str, float],
        crafting_weights: Dict[str, float],
        ingredient_threshold: float = DEFAULT_INGREDIENT_THRESHOLD,
    ) -> RollupResult:
        """
        Perform BOM rollup and convert result IDs back to display names.
        
        Same as rollup() but returns results keyed by display names instead of IDs.
        """
        result = self.rollup(inventory, crafting_weights, ingredient_threshold)
        
        def to_names(d: Dict[str, float]) -> Dict[str, float]:
            return {
                self._id_to_name.get(k, k): v
                for k, v in d.items()
            }
        
        return RollupResult(
            crafted=to_names(result.crafted),
            consumed=to_names(result.consumed),
            remaining=to_names(result.remaining),
            shortfall=to_names(result.shortfall),
            partial_progress=to_names(result.partial_progress),
        )


# Module-level singleton for convenience
_engine: Optional[BOMEngine] = None


def get_bom_engine() -> BOMEngine:
    """Get or create the singleton BOM engine instance."""
    global _engine
    if _engine is None:
        _engine = BOMEngine()
    return _engine


def flatten_bom(artifact_id: str, quantity: float = 1.0) -> Dict[str, float]:
    """
    Convenience function to flatten BOM for an artifact.
    
    See BOMEngine.flatten_bom for details.
    """
    return get_bom_engine().flatten_bom(artifact_id, quantity)


def rollup_inventory(
    inventory: Dict[str, float],
    crafting_weights: Dict[str, float],
    ingredient_threshold: float = DEFAULT_INGREDIENT_THRESHOLD,
) -> RollupResult:
    """
    Convenience function to perform BOM rollup on inventory.
    
    See BOMEngine.rollup for details.
    """
    return get_bom_engine().rollup(inventory, crafting_weights, ingredient_threshold)


def rollup_missions(
    missions: List[Tuple[str, str, Optional[str], int]],
    crafting_weights: Optional[Dict[str, float]] = None,
    mission_level: int = 0,
    capacity_bonus: float = 0.0,
    ingredient_threshold: float = DEFAULT_INGREDIENT_THRESHOLD,
) -> RollupResult:
    """
    Calculate BOM rollup from a list of mission specifications.
    
    This is the main convenience function for getting a rollup from missions.
    
    Parameters
    ----------
    missions : list of tuples
        Each tuple is (ship, duration, target_artifact, count):
        - ship: Ship name (e.g., "Henerprise", "HENERPRISE", "henerprise")
        - duration: Duration type (e.g., "Short", "SHORT", "Epic")
        - target_artifact: Target artifact filter or None for any
        - count: Number of missions to run
    crafting_weights : dict, optional
        Crafting priorities. If None, uses default weights (all 1.0)
    mission_level : int
        Mission level for capacity calculation (default 0)
    capacity_bonus : float
        Zero-G research bonus (e.g., 0.5 for 50%)
    ingredient_threshold : float
        Minimum quantity to consider for crafting
    
    Returns
    -------
    RollupResult
        BOM rollup results with display names
    
    Examples
    --------
    >>> # Single mission type
    >>> result = rollup_missions([("Henerprise", "Short", "Gold Meteorite", 20)])
    
    >>> # Multiple mission types
    >>> result = rollup_missions([
    ...     ("Henerprise", "Epic", None, 10),
    ...     ("Atreggies", "Short", "Book of Basan", 5),
    ... ])
    
    >>> # With custom weights (only craft certain items)
    >>> weights = {"Solid gold meteorite": 1.0, "Enriched gold meteorite": 0.0}
    >>> result = rollup_missions([("Henerprise", "Short", None, 20)], weights)
    """
    from .mission_data import build_mission_inventory
    
    # Build full inventory to find matching missions
    all_missions = build_mission_inventory()
    
    # Aggregate drops from all specified missions
    total_drops: Dict[str, float] = {}
    
    for ship, duration, target, count in missions:
        # Normalize inputs
        ship_norm = ship.upper().replace(" ", "_")
        duration_norm = duration.upper()
        
        # Find matching mission(s)
        matches = [
            m for m in all_missions
            if m.ship.upper() == ship_norm
            and m.duration_type.upper() == duration_norm
        ]
        
        # Filter by target if specified
        if target:
            target_lower = target.lower()
            matches = [
                m for m in matches
                if m.target_artifact and target_lower in m.target_artifact.lower()
            ]
        else:
            # No target specified - use UNKNOWN target mission (no specific target)
            matches = [m for m in matches if m.target_artifact == "UNKNOWN"]
        
        if not matches:
            # Try friendly name match
            ship_friendly = ship.lower().replace("_", " ")
            matches = [
                m for m in all_missions
                if m.ship_label.lower() == ship_friendly
                and m.duration_type.upper() == duration_norm
            ]
            if target:
                matches = [
                    m for m in matches
                    if m.target_artifact and target_lower in m.target_artifact.lower()
                ]
        
        if not matches:
            print(f"Warning: No mission found for {ship}/{duration}/{target}")
            continue
        
        # Use first match only (not all variants)
        mission = matches[0]
        drops = mission.expected_drops(mission_level, capacity_bonus)
        for art, qty in drops.items():
            total_drops[art] = total_drops.get(art, 0.0) + qty * count
    
    # Use default weights if not provided
    if crafting_weights is None:
        # Default: craft everything with equal weight
        engine = get_bom_engine()
        crafting_weights = {
            engine.id_to_name(art_id): 1.0
            for art_id in engine._craftable
            if engine.id_to_name(art_id)
        }
    
    # Perform rollup
    engine = get_bom_engine()
    return engine.rollup_with_display_names(
        inventory=total_drops,
        crafting_weights=crafting_weights,
        ingredient_threshold=ingredient_threshold,
    )


def rollup_mission(
    ship: str,
    duration: str,
    count: int,
    target: Optional[str] = None,
    crafting_weights: Optional[Dict[str, float]] = None,
    mission_level: int = 0,
    capacity_bonus: float = 0.0,
) -> RollupResult:
    """
    Calculate BOM rollup for a single mission type.
    
    Convenience wrapper around rollup_missions for single mission type.
    
    Parameters
    ----------
    ship : str
        Ship name (e.g., "Henerprise", "HENERPRISE")
    duration : str  
        Duration type (e.g., "Short", "Epic")
    count : int
        Number of missions
    target : str, optional
        Target artifact filter
    crafting_weights : dict, optional
        Crafting priorities
    mission_level : int
        Mission level (default 0)
    capacity_bonus : float
        Zero-G bonus (default 0.0)
    
    Returns
    -------
    RollupResult
    
    Examples
    --------
    >>> result = rollup_mission("Henerprise", "Short", 20, target="Gold Meteorite")
    >>> print(f"Crafted: {result.crafted}")
    """
    return rollup_missions(
        missions=[(ship, duration, target, count)],
        crafting_weights=crafting_weights,
        mission_level=mission_level,
        capacity_bonus=capacity_bonus,
    )


def calculate_mission_list_score(
    missions: List[Tuple[str, str, Optional[str], int]],
    crafting_weights: Optional[Dict[str, float]] = None,
    mission_weights: Optional[Dict[str, float]] = None,
    mission_level: int = 0,
    capacity_bonus: float = 0.0,
) -> Tuple[float, Dict[str, float], RollupResult]:
    """
    Calculate score, drops, and BOM for a manual mission list.
    
    This function is used by the Mission Planner to evaluate user-defined
    mission lists, as opposed to the solver which finds optimal missions.
    
    Parameters
    ----------
    missions : list of tuples
        Each tuple is (ship, duration, target_artifact, count)
    crafting_weights : dict, optional
        Crafting target weights (artifact name -> weight).
        Positive = desired, negative = waste, 0 = acceptable.
    mission_weights : dict, optional
        Direct mission drop weights (artifact name -> weight).
        Used in score calculation.
    mission_level : int
        Mission level for capacity calculation
    capacity_bonus : float
        Zero-G research bonus (e.g., 0.5 for 50%)
        
    Returns
    -------
    Tuple[float, Dict[str, float], RollupResult]
        (score, total_drops, bom_rollup)
        - score: Weighted sum of artifact values
        - total_drops: Expected drops from all missions
        - bom_rollup: BOM rollup result with crafting analysis
    """
    from .mission_data import build_mission_inventory
    
    # Build full inventory to find matching missions
    all_missions = build_mission_inventory()
    
    # Aggregate drops from all specified missions
    total_drops: Dict[str, float] = {}
    total_time_seconds = 0
    
    for ship, duration, target, count in missions:
        # Normalize inputs
        ship_norm = ship.upper().replace(" ", "_")
        duration_norm = duration.upper()
        
        # Find matching mission(s)
        matches = [
            m for m in all_missions
            if m.ship.upper() == ship_norm
            and m.duration_type.upper() == duration_norm
        ]
        
        # Filter by target if specified
        if target and target.lower() not in ("any", "unknown"):
            target_lower = target.lower()
            matches = [
                m for m in matches
                if m.target_artifact and target_lower in m.target_artifact.lower()
            ]
        else:
            # No target or 'any'/'unknown' - use UNKNOWN target mission
            matches = [m for m in matches if m.target_artifact == "UNKNOWN"]
        
        if not matches:
            # Try friendly name match
            ship_friendly = ship.lower().replace("_", " ")
            matches = [
                m for m in all_missions
                if m.ship_label.lower() == ship_friendly
                and m.duration_type.upper() == duration_norm
            ]
            if target and target.lower() not in ("any", "unknown"):
                matches = [
                    m for m in matches
                    if m.target_artifact and target_lower in m.target_artifact.lower()
                ]
            else:
                matches = [m for m in matches if m.target_artifact == "UNKNOWN"]
        
        if not matches:
            continue
        
        # Use first match only (not all variants)
        mission = matches[0]
        drops = mission.expected_drops(mission_level, capacity_bonus)
        for art, qty in drops.items():
            total_drops[art] = total_drops.get(art, 0.0) + qty * count
        # Add time (parallel missions would need different calculation)
        total_time_seconds += mission.seconds * count
    
    # Calculate score based on weights
    score = 0.0
    
    # Use mission weights for direct drop value
    if mission_weights:
        for art, qty in total_drops.items():
            weight = mission_weights.get(art, 0.0)
            score += qty * weight
    
    # If crafting weights provided, add value from craftable items
    # Use ingredient derivation to value base ingredients
    if crafting_weights:
        engine = get_bom_engine()
        ingredient_values = engine.calculate_ingredient_values_by_name(crafting_weights)
        for art, qty in total_drops.items():
            if art in ingredient_values:
                score += qty * ingredient_values[art]
    
    # If no weights at all, score is just total drops
    if not mission_weights and not crafting_weights:
        score = sum(total_drops.values())
    
    # Perform BOM rollup
    rollup_result = rollup_missions(
        missions=missions,
        crafting_weights=crafting_weights,
        mission_level=mission_level,
        capacity_bonus=capacity_bonus,
    )
    
    return score, total_drops, rollup_result


def print_rollup(result: RollupResult, show_remaining: bool = True) -> None:
    """
    Pretty-print a rollup result.
    
    Parameters
    ----------
    result : RollupResult
        The rollup result to display
    show_remaining : bool
        Whether to show remaining inventory (default True)
    """
    print("\n=== BOM Rollup Results ===\n")
    
    if result.crafted:
        print("Crafted:")
        for name, qty in sorted(result.crafted.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                print(f"  {name}: {qty:.3f}")
    else:
        print("Crafted: (none)")
    
    if result.consumed:
        print("\nConsumed as ingredients:")
        for name, qty in sorted(result.consumed.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                print(f"  {name}: {qty:.3f}")
    
    if result.partial_progress:
        print("\nPartial progress:")
        for name, progress in sorted(result.partial_progress.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {progress*100:.1f}%")
    
    if result.shortfall:
        print("\nShortfall (for next craft):")
        for name, qty in sorted(result.shortfall.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                print(f"  {name}: {qty:.3f} needed")
    
    if show_remaining and result.remaining:
        print("\nRemaining inventory:")
        for name, qty in sorted(result.remaining.items(), key=lambda kv: -kv[1]):
            if qty >= 0.01:
                print(f"  {name}: {qty:.2f}")


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _parse_mission_spec(spec: str) -> Tuple[str, str, Optional[str], int]:
    """
    Parse a mission specification string.
    
    Format: "Ship,Duration,Target,Count" or "Ship,Duration,Count"
    
    Examples:
        "Henerprise,Short,Gold Meteorite,20"
        "Henerprise,Epic,10"
        "ATREGGIES,SHORT,Book of Basan,5"
    """
    parts = [p.strip() for p in spec.split(",")]
    
    if len(parts) == 3:
        ship, duration, count_str = parts
        target = None
        count = int(count_str)
    elif len(parts) == 4:
        ship, duration, target, count_str = parts
        count = int(count_str)
    else:
        raise ValueError(
            f"Invalid mission spec: {spec}\n"
            "Expected format: Ship,Duration,Target,Count or Ship,Duration,Count"
        )
    
    return ship, duration, target, count


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for BOM rollup."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calculate BOM rollup from mission specifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m Solver.bom "Henerprise,Short,Gold Meteorite,20"
  python -m Solver.bom "Henerprise,Epic,10" "Atreggies,Short,5"
  python -m Solver.bom --level 8 --bonus 0.5 "Henerprise,Epic,3"
        """,
    )
    parser.add_argument(
        "missions",
        nargs="+",
        help="Mission specs: 'Ship,Duration,Target,Count' or 'Ship,Duration,Count'",
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=0,
        help="Mission level (default: 0)",
    )
    parser.add_argument(
        "--bonus", "-b",
        type=float,
        default=0.0,
        help="Zero-G capacity bonus as decimal (e.g., 0.5 for 50%%)",
    )
    parser.add_argument(
        "--no-remaining",
        action="store_true",
        help="Don't show remaining inventory",
    )
    
    args = parser.parse_args(argv)
    
    # Parse mission specifications
    parsed_missions = []
    for spec in args.missions:
        try:
            parsed_missions.append(_parse_mission_spec(spec))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    # Calculate rollup
    result = rollup_missions(
        missions=parsed_missions,
        mission_level=args.level,
        capacity_bonus=args.bonus,
    )
    
    # Display results
    print_rollup(result, show_remaining=not args.no_remaining)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
