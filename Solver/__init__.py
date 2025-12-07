"""Solver package for Egg Inc. mission optimization."""
from .config import load_config, UserConfig
from .mission_data import MissionOption, build_mission_inventory
from .mission_solver import solve, SolverResult, FuelUsage, calculate_fuel_usage
from .solver_logging import LogLevel, SolverLogger, create_logger, create_string_logger
from .aliases import (
    resolver,
    DisplayNameResolver,
    get_ship_display_name,
    get_artifact_family_display_name,
    get_artifact_tier_display_name,
    get_duration_display_name,
    get_rarity_display_name,
)

__all__ = [
    "load_config",
    "UserConfig",
    "MissionOption",
    "build_mission_inventory",
    "solve",
    "SolverResult",
    "FuelUsage",
    "calculate_fuel_usage",
    "LogLevel",
    "SolverLogger",
    "create_logger",
    "create_string_logger",
    # Alias functions
    "resolver",
    "DisplayNameResolver",
    "get_ship_display_name",
    "get_artifact_family_display_name",
    "get_artifact_tier_display_name",
    "get_duration_display_name",
    "get_rarity_display_name",
]
