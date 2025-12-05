#!/usr/bin/env python
"""CLI entry point for the mission LP solver."""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

from .config import load_config
from .mission_solver import solve


def format_bom_rollup(result) -> str:
    """Format BOM rollup results for display."""
    if not result.bom_rollup:
        return ""
    
    lines = ["\n--- BOM Rollup Summary ---"]
    rollup = result.bom_rollup
    
    if rollup.crafted:
        lines.append("\nCrafted artifacts:")
        for name, qty in sorted(rollup.crafted.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                lines.append(f"  {name}: {qty:.3f}")
    
    if rollup.consumed:
        lines.append("\nConsumed as ingredients:")
        for name, qty in sorted(rollup.consumed.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                lines.append(f"  {name}: {qty:.3f}")
    
    if rollup.partial_progress:
        lines.append("\nPartial craft progress (toward next):")
        for name, progress in sorted(rollup.partial_progress.items(), key=lambda kv: -kv[1]):
            pct = progress * 100
            lines.append(f"  {name}: {pct:.1f}%")
    
    if rollup.shortfall:
        lines.append("\nIngredient shortfall (for 1 more craft):")
        for name, qty in sorted(rollup.shortfall.items(), key=lambda kv: -kv[1]):
            if qty >= 0.001:
                lines.append(f"  {name}: {qty:.3f} needed")
    
    if rollup.remaining:
        lines.append("\nRemaining inventory (after rollup):")
        for name, qty in sorted(rollup.remaining.items(), key=lambda kv: -kv[1]):
            if qty >= 0.01:
                lines.append(f"  {name}: {qty:.2f}")
    
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimise Egg Inc. mission selection via linear programming."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to user config YAML (default: Solver/DefaultUserConfig.yaml)",
    )
    parser.add_argument(
        "-n",
        "--num-ships",
        type=int,
        default=3,
        help="Number of concurrent mission slots (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show solver output",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with cProfile and display performance statistics",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="cumulative",
        choices=["cumulative", "time", "calls", "name"],
        help="Sort order for profile output (default: cumulative)",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=30,
        help="Number of profile lines to display (default: 30)",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    
    # Run solver (with optional profiling)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        result = solve(config, num_ships=args.num_ships, verbose=args.verbose)
        profiler.disable()
        
        # Print profile statistics
        print("\n" + "=" * 60)
        print("PROFILING RESULTS")
        print("=" * 60)
        
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(args.profile_sort)
        stats.print_stats(args.profile_lines)
        print(stream.getvalue())
        
        # Also show totals
        print(f"\nTotal function calls: {stats.total_calls}")
        print(f"Total time: {stats.total_tt:.3f} seconds")
        print("=" * 60 + "\n")
    else:
        result = solve(config, num_ships=args.num_ships, verbose=args.verbose)

    print(f"Solver status: {result.status}")
    print(f"Objective value: {result.objective_value:.4f}")
    print(f"Total time: {result.total_time_hours:.2f} hours")
    print()
    print("Recommended missions:")
    for mission, count in result.selected_missions:
        target = mission.target_artifact or "Any"
        # Clean up target display
        if target and target.upper() == "UNKNOWN":
            target = "Any"
        print(
            f"  {count}x {mission.ship_label} / {mission.duration_type} / "
            f"Level {mission.level} / Target: {target}"
        )
    print()
    if result.total_drops:
        print("Expected average drops:")
        for art, amt in sorted(result.total_drops.items(), key=lambda kv: -kv[1]):
            if amt > 0:
                print(f"  {art}: {amt:.2f}")

    print()
    print(result.fuel_usage)
    print()
    print(f"Fuel tank capacity: {config.constraints.fuel_tank_capacity:.2f}T")
    tank_used = result.fuel_usage.tank_total / 1e12
    remaining = config.constraints.fuel_tank_capacity - tank_used
    print(f"Tank fuel remaining: {remaining:.2f}T")

    # Display slack artifact summary if any
    if result.slack_drops:
        print()
        print("--- Slack Artifacts (weight <= 0) ---")
        print(f"Efficiency bonus weight: {config.cost_weights.efficiency_bonus:.1f}")
        for art, amt in sorted(result.slack_drops.items(), key=lambda kv: -kv[1]):
            if amt > 0.01:
                print(f"  {art}: {amt:.2f}")
        print(f"Slack percentage: {result.slack_percentage:.1f}% of total drops")

    # Display BOM rollup if available
    bom_output = format_bom_rollup(result)
    if bom_output:
        print(bom_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
