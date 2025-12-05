"""Tests for BOM (Bill of Materials) rollup functionality.

Validates that:
1. Recipe data is loaded correctly from eiafx-data.json
2. BOM flattening expands recipes to base ingredients
3. Rarity normalization aggregates all rarities to common
4. Topological sort orders dependencies correctly
5. Priority-based allocation distributes shared ingredients by weight
6. Partial crafts are handled when ingredients are insufficient
"""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import Dict

from Solver.bom import (
    BOMEngine,
    RollupResult,
    flatten_bom,
    get_bom_engine,
    rollup_inventory,
    DEFAULT_INGREDIENT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bom_engine() -> BOMEngine:
    """Create a BOM engine instance for testing."""
    return BOMEngine()


# ---------------------------------------------------------------------------
# Tests: Recipe loading and basic queries
# ---------------------------------------------------------------------------

class TestRecipeLoading:
    """Test that recipes are loaded correctly from eiafx-data.json."""

    def test_engine_loads_recipes(self, bom_engine):
        """Engine should load recipes from eiafx-data.json."""
        # Puzzle cube tier 2 should have a recipe
        recipe = bom_engine.get_recipe("puzzle-cube-2")
        assert recipe is not None
        assert recipe.artifact_id == "puzzle-cube-2"
        assert recipe.artifact_name == "Puzzle cube"
        
    def test_puzzle_cube_2_recipe(self, bom_engine):
        """Puzzle cube tier 2 requires 3x tier 1."""
        recipe = bom_engine.get_recipe("puzzle-cube-2")
        assert recipe is not None
        
        # Should have 1 ingredient: puzzle-cube-1
        assert len(recipe.ingredients) == 1
        ing_id, ing_name, ing_count = recipe.ingredients[0]
        assert ing_id == "puzzle-cube-1"
        assert ing_count == 3

    def test_puzzle_cube_3_recipe(self, bom_engine):
        """Puzzle cube tier 3 requires 7x tier 2 + 2x plain gusset."""
        recipe = bom_engine.get_recipe("puzzle-cube-3")
        assert recipe is not None
        
        # Should have 2 ingredients
        assert len(recipe.ingredients) == 2
        
        ingredients = {ing_id: count for ing_id, _, count in recipe.ingredients}
        assert ingredients.get("puzzle-cube-2") == 7
        assert ingredients.get("ornate-gusset-1") == 2

    def test_non_craftable_has_no_recipe(self, bom_engine):
        """Non-craftable artifacts (tier 1) should have no recipe."""
        recipe = bom_engine.get_recipe("puzzle-cube-1")
        assert recipe is None
        
        assert not bom_engine.is_craftable("puzzle-cube-1")

    def test_craftable_flag(self, bom_engine):
        """Craftable artifacts should be marked as such."""
        assert bom_engine.is_craftable("puzzle-cube-2")
        assert bom_engine.is_craftable("puzzle-cube-3")
        assert bom_engine.is_craftable("puzzle-cube-4")


# ---------------------------------------------------------------------------
# Tests: Name/ID conversion and rarity handling
# ---------------------------------------------------------------------------

class TestNameConversion:
    """Test name to ID conversion and rarity handling."""

    def test_name_to_id_base(self, bom_engine):
        """Base names should map to artifact IDs."""
        assert bom_engine.name_to_id("Ancient puzzle cube") == "puzzle-cube-1"
        assert bom_engine.name_to_id("Puzzle cube") == "puzzle-cube-2"
        assert bom_engine.name_to_id("Mystical puzzle cube") == "puzzle-cube-3"

    def test_name_to_id_with_rarity(self, bom_engine):
        """Names with rarity suffixes should map to same base ID."""
        # All these should map to puzzle-cube-2
        assert bom_engine.name_to_id("Puzzle cube") == "puzzle-cube-2"
        assert bom_engine.name_to_id("Puzzle cube (Epic)") == "puzzle-cube-2"
        
        # puzzle-cube-3 has Rare variant
        assert bom_engine.name_to_id("Mystical puzzle cube") == "puzzle-cube-3"
        assert bom_engine.name_to_id("Mystical puzzle cube (Rare)") == "puzzle-cube-3"

    def test_id_to_name(self, bom_engine):
        """Artifact IDs should map to base names."""
        assert bom_engine.id_to_name("puzzle-cube-1") == "Ancient puzzle cube"
        assert bom_engine.id_to_name("puzzle-cube-2") == "Puzzle cube"

    def test_strip_rarity(self, bom_engine):
        """Rarity suffixes should be stripped correctly."""
        assert bom_engine.strip_rarity("Puzzle cube") == "Puzzle cube"
        assert bom_engine.strip_rarity("Puzzle cube (Epic)") == "Puzzle cube"
        assert bom_engine.strip_rarity("Puzzle cube (Rare)") == "Puzzle cube"
        assert bom_engine.strip_rarity("Puzzle cube (Legendary)") == "Puzzle cube"


# ---------------------------------------------------------------------------
# Tests: Inventory normalization
# ---------------------------------------------------------------------------

class TestInventoryNormalization:
    """Test that rarities are aggregated correctly."""

    def test_aggregate_rarities(self, bom_engine):
        """All rarity variants should be summed to same base ID."""
        inventory = {
            "Puzzle cube": 10.0,
            "Puzzle cube (Epic)": 5.0,
        }
        
        normalized = bom_engine.normalize_inventory(inventory)
        
        # Both should be aggregated to puzzle-cube-2
        assert normalized.get("puzzle-cube-2") == 15.0

    def test_multiple_artifacts(self, bom_engine):
        """Multiple artifact types should be normalized independently."""
        inventory = {
            "Ancient puzzle cube": 20.0,
            "Puzzle cube": 8.0,
            "Puzzle cube (Epic)": 2.0,
            "Plain gusset": 5.0,
        }
        
        normalized = bom_engine.normalize_inventory(inventory)
        
        assert normalized.get("puzzle-cube-1") == 20.0
        assert normalized.get("puzzle-cube-2") == 10.0  # 8 + 2
        assert normalized.get("ornate-gusset-1") == 5.0

    def test_zero_and_negative_filtered(self, bom_engine):
        """Zero and negative quantities should be filtered out."""
        inventory = {
            "Puzzle cube": 10.0,
            "Ancient puzzle cube": 0.0,
            "Plain gusset": -5.0,
        }
        
        normalized = bom_engine.normalize_inventory(inventory)
        
        assert "puzzle-cube-2" in normalized
        assert "puzzle-cube-1" not in normalized
        assert "ornate-gusset-1" not in normalized


# ---------------------------------------------------------------------------
# Tests: BOM flattening
# ---------------------------------------------------------------------------

class TestBOMFlattening:
    """Test recursive BOM expansion."""

    def test_flatten_tier_1(self, bom_engine):
        """Tier 1 (non-craftable) should return itself."""
        requirements = bom_engine.flatten_bom("puzzle-cube-1", 1.0)
        
        assert requirements == {"puzzle-cube-1": 1.0}

    def test_flatten_tier_2(self, bom_engine):
        """Tier 2 should expand to 3x tier 1."""
        requirements = bom_engine.flatten_bom("puzzle-cube-2", 1.0)
        
        # 1x puzzle-cube-2 = 3x puzzle-cube-1
        assert requirements.get("puzzle-cube-1") == 3.0

    def test_flatten_tier_2_quantity(self, bom_engine):
        """Quantity should be multiplied through."""
        requirements = bom_engine.flatten_bom("puzzle-cube-2", 5.0)
        
        # 5x puzzle-cube-2 = 15x puzzle-cube-1
        assert requirements.get("puzzle-cube-1") == 15.0

    def test_flatten_tier_3(self, bom_engine):
        """Tier 3 should expand recursively."""
        requirements = bom_engine.flatten_bom("puzzle-cube-3", 1.0)
        
        # 1x puzzle-cube-3 = 7x puzzle-cube-2 + 2x plain gusset
        # 7x puzzle-cube-2 = 21x puzzle-cube-1
        # Total: 21x puzzle-cube-1 + 2x plain gusset
        assert requirements.get("puzzle-cube-1") == 21.0
        assert requirements.get("ornate-gusset-1") == 2.0

    def test_flatten_tier_4(self, bom_engine):
        """Tier 4 should expand fully to base ingredients."""
        requirements = bom_engine.flatten_bom("puzzle-cube-4", 1.0)
        
        # puzzle-cube-4 = 10x puzzle-cube-3 + 2x solid gold meteorite
        # Each puzzle-cube-3 = 21x puzzle-cube-1 + 2x plain gusset
        # So: 210x puzzle-cube-1 + 20x plain gusset + 2x solid gold meteorite
        #
        # But solid gold meteorite is itself craftable:
        # gold-meteorite-3 = 11x gold-meteorite-2
        # gold-meteorite-2 = 11x gold-meteorite-1
        # So 2x gold-meteorite-3 = 2*11*11 = 242x gold-meteorite-1
        
        assert "puzzle-cube-1" in requirements
        assert "ornate-gusset-1" in requirements


# ---------------------------------------------------------------------------
# Tests: Topological sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    """Test topological ordering of dependencies."""

    def test_single_artifact(self, bom_engine):
        """Single non-craftable artifact should return just itself."""
        order = bom_engine.topological_sort({"puzzle-cube-1"})
        
        assert order == ["puzzle-cube-1"]

    def test_tier_2_order(self, bom_engine):
        """Tier 2 should come after tier 1 (dependency first)."""
        order = bom_engine.topological_sort({"puzzle-cube-2"})
        
        # Should include both tier 1 and tier 2
        assert "puzzle-cube-1" in order
        assert "puzzle-cube-2" in order
        
        # Tier 1 must come before tier 2
        idx_1 = order.index("puzzle-cube-1")
        idx_2 = order.index("puzzle-cube-2")
        assert idx_1 < idx_2

    def test_tier_3_order(self, bom_engine):
        """Tier 3 dependencies should be properly ordered."""
        order = bom_engine.topological_sort({"puzzle-cube-3"})
        
        # Should include tier 1, 2, 3 and plain gusset
        assert "puzzle-cube-1" in order
        assert "puzzle-cube-2" in order
        assert "puzzle-cube-3" in order
        assert "ornate-gusset-1" in order
        
        # Dependencies must come before dependents
        assert order.index("puzzle-cube-1") < order.index("puzzle-cube-2")
        assert order.index("puzzle-cube-2") < order.index("puzzle-cube-3")
        assert order.index("ornate-gusset-1") < order.index("puzzle-cube-3")


# ---------------------------------------------------------------------------
# Tests: Craft ratio calculation
# ---------------------------------------------------------------------------

class TestCraftRatios:
    """Test priority ratio calculation from weights."""

    def test_equal_weights(self, bom_engine):
        """Equal weights should give equal ratios."""
        weights = {
            "Puzzle cube": 1.0,
            "Mystical puzzle cube": 1.0,
        }
        
        ratios = bom_engine.calculate_craft_ratios(weights)
        
        # Each should get 50%
        assert ratios.get("puzzle-cube-2") == pytest.approx(0.5)
        assert ratios.get("puzzle-cube-3") == pytest.approx(0.5)

    def test_weighted_ratios(self, bom_engine):
        """Different weights should give proportional ratios."""
        weights = {
            "Puzzle cube": 1.0,
            "Mystical puzzle cube": 3.0,
        }
        
        ratios = bom_engine.calculate_craft_ratios(weights)
        
        # 1:3 ratio = 25%:75%
        assert ratios.get("puzzle-cube-2") == pytest.approx(0.25)
        assert ratios.get("puzzle-cube-3") == pytest.approx(0.75)

    def test_zero_weight_excluded(self, bom_engine):
        """Zero or negative weights should exclude artifacts."""
        weights = {
            "Puzzle cube": 0.0,
            "Mystical puzzle cube": 1.0,
            "Unsolvable puzzle cube": -1.0,
        }
        
        ratios = bom_engine.calculate_craft_ratios(weights)
        
        # Only mystical should be included
        assert "puzzle-cube-2" not in ratios
        assert "puzzle-cube-4" not in ratios
        assert ratios.get("puzzle-cube-3") == pytest.approx(1.0)

    def test_rarity_variants_use_max_weight(self, bom_engine):
        """Multiple rarity entries should use max weight."""
        weights = {
            "Puzzle cube": 1.0,
            "Puzzle cube (Epic)": 2.0,  # Same artifact, higher weight
        }
        
        ratios = bom_engine.calculate_craft_ratios(weights)
        
        # Should use the max of the two (2.0)
        # Since only one artifact, ratio is 100%
        assert ratios.get("puzzle-cube-2") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: Full rollup
# ---------------------------------------------------------------------------

class TestBOMRollup:
    """Test complete BOM rollup functionality."""

    def test_simple_craft(self, bom_engine):
        """Simple craft with sufficient ingredients."""
        inventory = {
            "Ancient puzzle cube": 9.0,  # Enough for 3x tier 2
        }
        weights = {
            "Puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should craft 3x puzzle-cube-2, consuming 9x puzzle-cube-1
        assert result.crafted.get("puzzle-cube-2") == pytest.approx(3.0)
        assert result.consumed.get("puzzle-cube-1") == pytest.approx(9.0)

    def test_insufficient_ingredients(self, bom_engine):
        """Partial craft when ingredients are insufficient but above threshold."""
        inventory = {
            "Ancient puzzle cube": 2.0,  # Not enough for 1x tier 2 (needs 3)
        }
        weights = {
            "Puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should craft fractional amount (2/3 = 0.667)
        # This is the correct behavior - fractional crafts are allowed
        assert result.crafted.get("puzzle-cube-2") == pytest.approx(2.0 / 3.0)
        assert result.consumed.get("puzzle-cube-1") == pytest.approx(2.0)

    def test_threshold_behavior(self, bom_engine):
        """Crafts below threshold should not proceed."""
        inventory = {
            "Ancient puzzle cube": 0.001,  # At threshold
        }
        weights = {
            "Puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup(inventory, weights, ingredient_threshold=0.001)
        
        # Nothing should be crafted
        assert not result.crafted

    def test_zero_weight_not_crafted(self, bom_engine):
        """Artifacts with weight <= 0 should not be crafted as targets."""
        inventory = {
            "Ancient puzzle cube": 30.0,
            "Puzzle cube": 14.0,  # Provide tier 2 for tier 3 crafting
            "Plain gusset": 4.0,  # Need 2 per tier 3
        }
        weights = {
            "Puzzle cube": 0.0,  # Don't craft tier 2 as target
            "Mystical puzzle cube": 1.0,  # Craft tier 3
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should only craft tier 3 as a target
        # Tier 2 should not be crafted since its weight is 0
        assert "puzzle-cube-3" in result.crafted
        assert result.crafted.get("puzzle-cube-3") == pytest.approx(2.0)

    def test_shared_ingredient_allocation(self, bom_engine):
        """Shared ingredients should be allocated by priority weight."""
        # Both tier 3 puzzle cube and tier 2 gusset need tier 1 gusset
        inventory = {
            "Puzzle cube": 14.0,  # Enough for 2x tier 3 (needs 7 each)
            "Plain gusset": 4.0,  # Need 2 for each tier 3
        }
        weights = {
            "Mystical puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should craft 2x tier 3
        assert result.crafted.get("puzzle-cube-3") == pytest.approx(2.0)

    def test_multiple_targets_share_ingredients_equally(self, bom_engine):
        """Multiple targets with equal weights should split shared ingredients proportionally."""
        # puzzle-cube-3 needs: 21 puzzle-cube-1 + 2 ornate-gusset-1 per craft
        # ornate-gusset-2 needs: 5 ornate-gusset-1 per craft
        # Both share ornate-gusset-1 (Plain gusset)
        inventory = {
            "Ancient puzzle cube": 42.0,  # Enough for 2x tier 3 (needs 21 each)
            "Plain gusset": 7.0,  # Shared between both targets
        }
        weights = {
            "Mystical puzzle cube": 1.0,  # puzzle-cube-3
            "Ornate gusset": 1.0,  # ornate-gusset-2
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # With equal weights, 7 gussets split 50/50:
        # - puzzle-cube-3 gets 3.5 gussets -> 1.75 crafts (needs 2 each)
        # - ornate-gusset-2 gets 3.5 gussets -> 0.7 crafts (needs 5 each)
        assert result.crafted.get("puzzle-cube-3") == pytest.approx(1.75)
        assert result.crafted.get("ornate-gusset-2") == pytest.approx(0.7)

    def test_multiple_targets_share_ingredients_weighted(self, bom_engine):
        """Multiple targets with unequal weights should split shared ingredients by ratio."""
        inventory = {
            "Ancient puzzle cube": 42.0,
            "Plain gusset": 7.0,
        }
        weights = {
            "Mystical puzzle cube": 3.0,  # 75% weight
            "Ornate gusset": 1.0,  # 25% weight
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # With 3:1 weights, 7 gussets split:
        # - puzzle-cube-3 gets 5.25 gussets -> but limited to 2 by puzzle-cube-1 (42/21=2)
        # - ornate-gusset-2 gets 1.75 gussets -> 0.35 crafts (needs 5 each)
        assert result.crafted.get("puzzle-cube-3") == pytest.approx(2.0)
        assert result.crafted.get("ornate-gusset-2") == pytest.approx(0.35)

    def test_remaining_inventory(self, bom_engine):
        """Unused ingredients should appear in remaining."""
        inventory = {
            "Ancient puzzle cube": 10.0,  # All will be crafted into tier 2
            "Plain gusset": 5.0,  # unused
        }
        weights = {
            "Puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should craft 10/3 = 3.333 tier 2 (fractional)
        crafted = result.crafted.get("puzzle-cube-2", 0.0)
        
        assert crafted == pytest.approx(10.0 / 3.0)  # 3.333...
        
        # Crafted tier 2 should be in remaining
        assert result.remaining.get("puzzle-cube-2") == pytest.approx(10.0 / 3.0)
        
        # Plain gusset unused
        assert result.remaining.get("ornate-gusset-1") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests: Display name conversion
# ---------------------------------------------------------------------------

class TestDisplayNameRollup:
    """Test rollup with display name output."""

    def test_rollup_with_display_names(self, bom_engine):
        """Result should use display names when requested."""
        inventory = {
            "Ancient puzzle cube": 9.0,
        }
        weights = {
            "Puzzle cube": 1.0,
        }
        
        result = bom_engine.rollup_with_display_names(inventory, weights)
        
        # Should have display names as keys
        assert "Puzzle cube" in result.crafted
        assert "Ancient puzzle cube" in result.consumed


# ---------------------------------------------------------------------------
# Tests: Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_bom_engine_singleton(self):
        """get_bom_engine should return same instance."""
        engine1 = get_bom_engine()
        engine2 = get_bom_engine()
        
        assert engine1 is engine2

    def test_flatten_bom_function(self):
        """flatten_bom convenience function should work."""
        requirements = flatten_bom("puzzle-cube-2", 1.0)
        
        assert requirements.get("puzzle-cube-1") == 3.0

    def test_rollup_inventory_function(self):
        """rollup_inventory convenience function should work."""
        inventory = {"Ancient puzzle cube": 6.0}
        weights = {"Puzzle cube": 1.0}
        
        result = rollup_inventory(inventory, weights)
        
        assert result.crafted.get("puzzle-cube-2") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_inventory(self, bom_engine):
        """Empty inventory should return empty result."""
        result = bom_engine.rollup({}, {"Puzzle cube": 1.0})
        
        assert not result.crafted
        assert not result.consumed
        assert not result.remaining

    def test_empty_weights(self, bom_engine):
        """Empty weights should return inventory as-is."""
        inventory = {"Ancient puzzle cube": 5.0}
        result = bom_engine.rollup(inventory, {})
        
        assert not result.crafted
        assert result.remaining.get("puzzle-cube-1") == 5.0

    def test_unknown_artifact(self, bom_engine):
        """Unknown artifacts should be ignored."""
        inventory = {
            "Unknown Artifact": 10.0,
            "Ancient puzzle cube": 3.0,
        }
        weights = {"Puzzle cube": 1.0}
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should still craft from known inventory
        assert result.crafted.get("puzzle-cube-2") == pytest.approx(1.0)

    def test_fractional_crafts(self, bom_engine):
        """Fractional craft quantities should be handled."""
        inventory = {
            "Ancient puzzle cube": 4.5,  # 1.5x tier 2
        }
        weights = {"Puzzle cube": 1.0}
        
        result = bom_engine.rollup(inventory, weights)
        
        # Should craft 1.5x tier 2
        assert result.crafted.get("puzzle-cube-2") == pytest.approx(1.5)
        assert result.consumed.get("puzzle-cube-1") == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Tests: Ingredient Value Calculation (for solver integration)
# ---------------------------------------------------------------------------

class TestIngredientValueCalculation:
    """Test ingredient value propagation for solver optimization."""

    def test_basic_ingredient_value(self, bom_engine):
        """Ingredients should inherit value from what they craft."""
        # If Puzzle cube (tier 2) has weight 10, Ancient puzzle cube should
        # derive value from being an ingredient
        weights = {"Puzzle cube": 10.0}  # Tier 2
        
        values = bom_engine.calculate_ingredient_values_by_name(weights)
        
        # Ancient puzzle cube (tier 1) should have positive value as ingredient
        assert values.get("Ancient puzzle cube", 0) > 0
        # Puzzle cube itself should have its weight
        assert values.get("Puzzle cube", 0) == 10.0

    def test_higher_tier_propagates_more_value(self, bom_engine):
        """Higher tier targets should propagate more value to ingredients."""
        # High weight on tier 3
        weights1 = {"Mystical puzzle cube": 100.0}
        values1 = bom_engine.calculate_ingredient_values_by_name(weights1)
        
        # Low weight on tier 2
        weights2 = {"Puzzle cube": 10.0}
        values2 = bom_engine.calculate_ingredient_values_by_name(weights2)
        
        # Tier 1 should have more value when higher tier is weighted
        ancient_val1 = values1.get("Ancient puzzle cube", 0)
        ancient_val2 = values2.get("Ancient puzzle cube", 0)
        assert ancient_val1 > ancient_val2

    def test_zero_weight_no_value(self, bom_engine):
        """Zero or negative weights should not propagate value."""
        weights = {"Puzzle cube": 0.0}
        
        values = bom_engine.calculate_ingredient_values_by_name(weights)
        
        # No targets with positive weight = no ingredient values
        assert len(values) == 0 or all(v == 0 for v in values.values())

    def test_empty_weights_empty_values(self, bom_engine):
        """Empty weights should produce empty values."""
        values = bom_engine.calculate_ingredient_values_by_name({})
        
        assert len(values) == 0

    def test_multiple_targets_accumulate(self, bom_engine):
        """Ingredient used by multiple targets should accumulate value."""
        # If both tier 2 and tier 3 puzzle cubes are wanted, tier 1 should
        # have value from being an ingredient for both (through tier 2)
        weights1 = {"Puzzle cube": 10.0}  # Just tier 2
        weights2 = {"Puzzle cube": 10.0, "Mystical puzzle cube": 10.0}  # Both
        
        values1 = bom_engine.calculate_ingredient_values_by_name(weights1)
        values2 = bom_engine.calculate_ingredient_values_by_name(weights2)
        
        # With both targets, tier 1 should have more value
        ancient_val1 = values1.get("Ancient puzzle cube", 0)
        ancient_val2 = values2.get("Ancient puzzle cube", 0)
        assert ancient_val2 > ancient_val1
