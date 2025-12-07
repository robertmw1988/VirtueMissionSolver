"""
Display name aliases for missions, ships, and artifacts.

This module provides centralized translation functions to convert internal
API/data file names to user-friendly display names shown in the game.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .resources import get_resource_path

# Data file paths
EIAFX_DATA_PATH = get_resource_path("Wasmegg/eiafx-data.json")
EIAFX_CONFIG_PATH = get_resource_path("Wasmegg/eiafx-config.json")


# =============================================================================
# Ship Name Aliases
# =============================================================================

# Ship API name -> (Display Name, Max Stars)
# These match the in-game ship names exactly
SHIP_ALIASES: Dict[str, Tuple[str, int]] = {
    "CHICKEN_ONE": ("Chicken One", 0),
    "CHICKEN_NINE": ("Chicken Nine", 2),
    "CHICKEN_HEAVY": ("Chicken Heavy", 3),
    "BCR": ("BCR", 4),
    "MILLENIUM_CHICKEN": ("Quintillion Chicken", 4),
    "CORELLIHEN_CORVETTE": ("Cornish-Hen Corvette", 4),
    "GALEGGTICA": ("Galeggtica", 5),
    "CHICKFIANT": ("Defihent", 5),
    "VOYEGGER": ("Voyegger", 6),
    "HENERPRISE": ("Henerprise", 8),
    "ATREGGIES": ("Atreggies Henliner", 8),
}

# Reverse lookup: display name -> API name
SHIP_DISPLAY_TO_API: Dict[str, str] = {
    display: api for api, (display, _) in SHIP_ALIASES.items()
}


def get_ship_display_name(api_name: str) -> str:
    """
    Convert ship API name to in-game display name.
    
    Parameters
    ----------
    api_name : str
        The internal API name (e.g., "MILLENIUM_CHICKEN")
        
    Returns
    -------
    str
        The display name shown in-game (e.g., "Quintillion Chicken")
    """
    if not api_name:
        return ""
    if api_name in SHIP_ALIASES:
        return SHIP_ALIASES[api_name][0]
    # Fallback: convert SCREAMING_SNAKE_CASE to Title Case
    return _screaming_snake_to_title(api_name)


def get_ship_api_name(display_name: str) -> Optional[str]:
    """
    Convert ship display name to API name.
    
    Parameters
    ----------
    display_name : str
        The in-game display name (e.g., "Quintillion Chicken")
        
    Returns
    -------
    str or None
        The API name, or None if not found
    """
    return SHIP_DISPLAY_TO_API.get(display_name)


def get_ship_max_stars(api_name: str) -> int:
    """
    Get the maximum star level for a ship.
    
    Parameters
    ----------
    api_name : str
        The internal API name
        
    Returns
    -------
    int
        Maximum star level (0-8)
    """
    if api_name in SHIP_ALIASES:
        return SHIP_ALIASES[api_name][1]
    return 0


# =============================================================================
# Duration Type Aliases
# =============================================================================

# In-game display: Short, Standard, Extended (no Tutorial/Long/Epic shown)
DURATION_ALIASES: Dict[str, str] = {
    "TUTORIAL": "Tutorial",  # Not typically shown in-game
    "SHORT": "Short",
    "STANDARD": "Standard",
    "LONG": "Standard",      # "LONG" maps to "Standard" in-game
    "EPIC": "Extended",      # "EPIC" maps to "Extended" in-game
    "EXTENDED": "Extended",
}

DURATION_DISPLAY_TO_API: Dict[str, str] = {v: k for k, v in DURATION_ALIASES.items()}


def get_duration_display_name(api_name: str) -> str:
    """
    Convert duration API name to display name.
    
    Parameters
    ----------
    api_name : str
        The internal API name (e.g., "SHORT", "EPIC")
        
    Returns
    -------
    str
        The display name (e.g., "Short", "Epic")
    """
    if not api_name:
        return ""
    upper = api_name.upper()
    return DURATION_ALIASES.get(upper, api_name.capitalize())


def get_duration_api_name(display_name: str) -> Optional[str]:
    """Convert duration display name to API name."""
    return DURATION_DISPLAY_TO_API.get(display_name, display_name.upper())


# =============================================================================
# Rarity Aliases
# =============================================================================

RARITY_ALIASES: Dict[str, str] = {
    "COMMON": "Common",
    "RARE": "Rare",
    "EPIC": "Epic",
    "LEGENDARY": "Legendary",
}

RARITY_INDEX_TO_NAME: Dict[int, str] = {
    0: "COMMON",
    1: "RARE", 
    2: "EPIC",
    3: "LEGENDARY",
}

RARITY_NAME_TO_INDEX: Dict[str, int] = {v: k for k, v in RARITY_INDEX_TO_NAME.items()}


def get_rarity_display_name(api_name: str) -> str:
    """Convert rarity API name to display name."""
    if not api_name:
        return "Common"
    return RARITY_ALIASES.get(api_name.upper(), api_name.capitalize())


def get_rarity_from_index(index: int) -> str:
    """Convert rarity index (0-3) to API name."""
    return RARITY_INDEX_TO_NAME.get(index, "COMMON")


# =============================================================================
# Artifact Name Aliases (loaded from data files)
# =============================================================================

@lru_cache(maxsize=1)
def _load_artifact_data() -> Dict[str, Any]:
    """Load artifact data from eiafx-data.json (cached)."""
    if not EIAFX_DATA_PATH.exists():
        return {}
    try:
        with EIAFX_DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@lru_cache(maxsize=1)
def _build_artifact_aliases() -> Tuple[
    Dict[str, str],           # family_id -> display name
    Dict[str, str],           # tier_id -> display name  
    Dict[Tuple[int, int], str],  # (afx_id, level) -> display name
    Dict[str, str],           # display name -> family_id (reverse lookup)
]:
    """
    Build artifact name lookup tables from data files.
    
    Returns
    -------
    tuple
        (family_aliases, tier_aliases, numeric_tier_aliases, display_to_family)
    """
    data = _load_artifact_data()
    
    family_aliases: Dict[str, str] = {}
    tier_aliases: Dict[str, str] = {}
    numeric_tier_aliases: Dict[Tuple[int, int], str] = {}
    display_to_family: Dict[str, str] = {}
    
    families = data.get("artifact_families", [])
    for family in families:
        if not isinstance(family, dict):
            continue
            
        family_id = family.get("id", "")
        family_name = family.get("name", "")
        afx_id = family.get("afx_id")
        
        if family_id and family_name:
            family_aliases[family_id] = family_name
            display_to_family[family_name.lower()] = family_id
            
        # Process tiers
        for tier in family.get("tiers", []):
            if not isinstance(tier, dict):
                continue
                
            tier_id = tier.get("id", "")
            tier_name = tier.get("name", "")
            level = tier.get("afx_level")
            
            if tier_id and tier_name:
                tier_aliases[tier_id] = tier_name
                
            if afx_id is not None and level is not None:
                numeric_tier_aliases[(afx_id, level)] = tier_name
                
    return family_aliases, tier_aliases, numeric_tier_aliases, display_to_family


def get_artifact_family_display_name(family_id: str) -> str:
    """
    Convert artifact family ID to display name.
    
    Parameters
    ----------
    family_id : str
        The internal family ID (e.g., "puzzle-cube", "demeters-necklace")
        
    Returns
    -------
    str
        The display name (e.g., "Puzzle cube", "Demeters necklace")
    """
    if not family_id:
        return ""
    family_aliases, _, _, _ = _build_artifact_aliases()
    if family_id in family_aliases:
        return family_aliases[family_id]
    # Fallback: convert kebab-case to Title Case
    return _kebab_to_title(family_id)


def get_artifact_tier_display_name(tier_id: str) -> str:
    """
    Convert artifact tier ID to display name.
    
    Parameters
    ----------
    tier_id : str
        The internal tier ID (e.g., "puzzle-cube-1", "ornate-gusset-3")
        
    Returns
    -------
    str
        The display name (e.g., "Ancient puzzle cube", "Distegguished gusset")
    """
    if not tier_id:
        return ""
    _, tier_aliases, _, _ = _build_artifact_aliases()
    if tier_id in tier_aliases:
        return tier_aliases[tier_id]
    # Fallback: convert kebab-case to Title Case
    return _kebab_to_title(tier_id)


def get_artifact_display_name_by_ids(afx_id: int, level: int) -> str:
    """
    Get artifact display name from numeric IDs.
    
    Parameters
    ----------
    afx_id : int
        The artifact family ID number
    level : int
        The artifact tier level (0-based)
        
    Returns
    -------
    str
        The display name for this artifact tier
    """
    _, _, numeric_aliases, _ = _build_artifact_aliases()
    return numeric_aliases.get((afx_id, level), f"Artifact {afx_id}-{level}")


def get_artifact_family_id(display_name: str) -> Optional[str]:
    """
    Convert artifact display name back to family ID.
    
    Parameters
    ----------
    display_name : str
        The display name shown in-game
        
    Returns
    -------
    str or None
        The family ID, or None if not found
    """
    _, _, _, display_to_family = _build_artifact_aliases()
    return display_to_family.get(display_name.lower())


def format_artifact_with_rarity(base_name: str, rarity: str = "COMMON") -> str:
    """
    Format artifact name with rarity suffix.
    
    Parameters
    ----------
    base_name : str
        The base artifact name
    rarity : str
        The rarity (COMMON, RARE, EPIC, LEGENDARY)
        
    Returns
    -------
    str
        Formatted name, e.g., "Puzzle cube" or "Puzzle cube (Rare)"
    """
    rarity_upper = (rarity or "COMMON").upper()
    if rarity_upper == "COMMON":
        return base_name
    rarity_display = get_rarity_display_name(rarity_upper)
    return f"{base_name} ({rarity_display})"


# =============================================================================
# Egg Type Aliases
# =============================================================================

EGG_ALIASES: Dict[str, str] = {
    # Virtue eggs (mission fuels)
    "HUMILITY": "Humility Egg",
    "INTEGRITY": "Integrity Egg",
    "CURIOSITY": "Curiosity Egg",
    "KINDNESS": "Kindness Egg",
    "RESILIENCE": "Resilience Egg",
    # Standard eggs
    "EDIBLE": "Edible Egg",
    "SUPERFOOD": "Superfood Egg",
    "MEDICAL": "Medical Egg",
    "ROCKET_FUEL": "Rocket Fuel Egg",
    "SUPER_MATERIAL": "Super Material Egg",
    "FUSION": "Fusion Egg",
    "QUANTUM": "Quantum Egg",
    "IMMORTALITY": "Immortality Egg",
    "TACHYON": "Tachyon Egg",
    "GRAVITON": "Graviton Egg",
    "DILITHIUM": "Dilithium Egg",
    "PRODIGY": "Prodigy Egg",
    "TERRAFORM": "Terraform Egg",
    "ANTIMATTER": "Antimatter Egg",
    "DARK_MATTER": "Dark Matter Egg",
    "AI": "AI Egg",
    "NEBULA": "Nebula Egg",
    "UNIVERSE": "Universe Egg",
    "ENLIGHTENMENT": "Enlightenment Egg",
    # Special eggs
    "CHOCOLATE": "Chocolate Egg",
    "EASTER": "Easter Egg",
    "WATERBALLOON": "Waterballoon Egg",
    "FIREWORK": "Firework Egg",
    "PUMPKIN": "Pumpkin Egg",
}


def get_egg_display_name(api_name: str) -> str:
    """Convert egg API name to display name."""
    if not api_name:
        return ""
    return EGG_ALIASES.get(api_name.upper(), _screaming_snake_to_title(api_name))


# =============================================================================
# Utility Functions
# =============================================================================

def _screaming_snake_to_title(text: str) -> str:
    """Convert SCREAMING_SNAKE_CASE to Title Case."""
    if not text:
        return ""
    # Handle short acronyms (keep as-is)
    if len(text) <= 4 and text.isupper() and "_" not in text:
        return text
    words = text.lower().split("_")
    return " ".join(word.capitalize() for word in words)


def _kebab_to_title(text: str) -> str:
    """Convert kebab-case to Title Case."""
    if not text:
        return ""
    words = text.split("-")
    return " ".join(word.capitalize() for word in words)


# =============================================================================
# Aggregate Alias Lookup
# =============================================================================

class DisplayNameResolver:
    """
    Centralized resolver for all display name translations.
    
    Provides a unified interface for translating internal names to
    user-friendly display names.
    
    Example
    -------
    >>> resolver = DisplayNameResolver()
    >>> resolver.ship("MILLENIUM_CHICKEN")
    'Quintillion Chicken'
    >>> resolver.artifact_family("puzzle-cube")
    'Puzzle cube'
    >>> resolver.artifact_tier("ornate-gusset-3")
    'Distegguished gusset'
    """
    
    def ship(self, api_name: str) -> str:
        """Get ship display name."""
        return get_ship_display_name(api_name)
    
    def duration(self, api_name: str) -> str:
        """Get duration display name."""
        return get_duration_display_name(api_name)
    
    def rarity(self, api_name: str) -> str:
        """Get rarity display name."""
        return get_rarity_display_name(api_name)
    
    def artifact_family(self, family_id: str) -> str:
        """Get artifact family display name."""
        return get_artifact_family_display_name(family_id)
    
    def artifact_tier(self, tier_id: str) -> str:
        """Get artifact tier display name."""
        return get_artifact_tier_display_name(tier_id)
    
    def artifact(self, afx_id: int, level: int, rarity: str = "COMMON") -> str:
        """Get full artifact display name with optional rarity."""
        base = get_artifact_display_name_by_ids(afx_id, level)
        return format_artifact_with_rarity(base, rarity)
    
    def egg(self, api_name: str) -> str:
        """Get egg display name."""
        return get_egg_display_name(api_name)


# Global resolver instance for convenience
resolver = DisplayNameResolver()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Ship functions
    "get_ship_display_name",
    "get_ship_api_name", 
    "get_ship_max_stars",
    "SHIP_ALIASES",
    # Duration functions
    "get_duration_display_name",
    "get_duration_api_name",
    "DURATION_ALIASES",
    # Rarity functions
    "get_rarity_display_name",
    "get_rarity_from_index",
    "RARITY_ALIASES",
    "RARITY_INDEX_TO_NAME",
    "RARITY_NAME_TO_INDEX",
    # Artifact functions
    "get_artifact_family_display_name",
    "get_artifact_tier_display_name",
    "get_artifact_display_name_by_ids",
    "get_artifact_family_id",
    "format_artifact_with_rarity",
    # Egg functions
    "get_egg_display_name",
    "EGG_ALIASES",
    # Resolver class
    "DisplayNameResolver",
    "resolver",
]
