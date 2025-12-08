"""
Data quality analysis for mission drop data.

This module provides tools to assess the reliability of observed drop data.
It calculates confidence intervals and sample size metrics to help identify
missions with insufficient observations.

IMPORTANT LIMITATIONS:
- These calculations treat aggregated drops as independent samples, which
  is a simplification. In reality, drops within a mission are correlated
  (same quality roll affects all slots).
- The "confidence" scores are better understood as data quality indicators
  rather than true statistical significance measures.
- The actual drop probabilities are determined by the game's RNG mechanics
  (base quality values, quality ranges), not just observed frequencies.

Use these metrics as rough data quality filters, not as precise statistical
guarantees about drop rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Common z-scores for confidence levels
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


@dataclass
class DropStatistics:
    """
    Statistical analysis of drop data for a single artifact type.
    
    Attributes
    ----------
    artifact : str
        Name of the artifact
    drop_count : int
        Number of times this artifact was dropped
    total_drops : int
        Total number of drops from the mission type
    drop_rate : float
        Observed drop rate (drop_count / total_drops)
    confidence_interval : Tuple[float, float]
        (lower, upper) bounds of the 95% confidence interval
    margin_of_error : float
        Half-width of the confidence interval
    is_significant : bool
        True if margin of error is within acceptable threshold
    """
    artifact: str
    drop_count: int
    total_drops: int
    drop_rate: float
    confidence_interval: Tuple[float, float]
    margin_of_error: float
    is_significant: bool


@dataclass
class MissionStatistics:
    """
    Statistical summary for a mission type.
    
    Attributes
    ----------
    ship : str
        Ship name
    duration : str
        Duration type
    level : int
        Mission level
    target_artifact : Optional[str]
        Target artifact (if any)
    total_drops : int
        Total number of observed drops
    total_missions : int
        Estimated number of missions in the sample (total_drops / avg_capacity)
    artifact_stats : Dict[str, DropStatistics]
        Per-artifact statistical breakdown
    overall_confidence : float
        Overall confidence score (0-1) based on sample size
    meets_threshold : bool
        True if sample size meets minimum threshold
    """
    ship: str
    duration: str
    level: int
    target_artifact: Optional[str]
    total_drops: int
    total_missions: int
    artifact_stats: Dict[str, DropStatistics]
    overall_confidence: float
    meets_threshold: bool


def calculate_wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    Wilson score is preferred over normal approximation for small samples
    and proportions near 0 or 1.
    
    Parameters
    ----------
    successes : int
        Number of successes (e.g., drops of a specific artifact)
    trials : int
        Total number of trials (e.g., total drops)
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    
    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval
    """
    if trials == 0:
        return (0.0, 1.0)
    
    z = Z_SCORES.get(confidence, 1.960)
    p_hat = successes / trials
    
    denominator = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials) / denominator
    
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    
    return (lower, upper)


def calculate_normal_margin_of_error(
    p: float,
    n: int,
    confidence: float = 0.95,
) -> float:
    """
    Calculate margin of error using normal approximation.
    
    Parameters
    ----------
    p : float
        Observed proportion
    n : int
        Sample size
    confidence : float
        Confidence level
    
    Returns
    -------
    float
        Margin of error
    """
    if n == 0:
        return 1.0
    
    z = Z_SCORES.get(confidence, 1.960)
    return z * math.sqrt(p * (1 - p) / n)


def minimum_sample_size(
    margin_of_error: float = 0.05,
    confidence: float = 0.95,
    estimated_proportion: float = 0.5,
) -> int:
    """
    Calculate minimum sample size needed for a given margin of error.
    
    Parameters
    ----------
    margin_of_error : float
        Desired margin of error (e.g., 0.05 for ±5%)
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    estimated_proportion : float
        Estimated proportion (0.5 is most conservative)
    
    Returns
    -------
    int
        Minimum sample size required
    """
    if margin_of_error <= 0:
        return 2**31  # Very large number for invalid margin
    
    z = Z_SCORES.get(confidence, 1.960)
    n = (z * z * estimated_proportion * (1 - estimated_proportion)) / (margin_of_error * margin_of_error)
    return math.ceil(n)


def analyze_artifact_drops(
    artifact: str,
    drop_count: int,
    total_drops: int,
    margin_threshold: float = 0.10,
    confidence: float = 0.95,
) -> DropStatistics:
    """
    Analyze statistical significance of an artifact's drop rate.
    
    Parameters
    ----------
    artifact : str
        Artifact name
    drop_count : int
        Number of this artifact dropped
    total_drops : int
        Total drops from the mission type
    margin_threshold : float
        Maximum acceptable margin of error (default 0.10 = ±10%)
    confidence : float
        Confidence level for interval calculation
    
    Returns
    -------
    DropStatistics
        Complete statistical analysis
    """
    if total_drops == 0:
        return DropStatistics(
            artifact=artifact,
            drop_count=0,
            total_drops=0,
            drop_rate=0.0,
            confidence_interval=(0.0, 1.0),
            margin_of_error=1.0,
            is_significant=False,
        )
    
    drop_rate = drop_count / total_drops
    ci = calculate_wilson_interval(drop_count, total_drops, confidence)
    moe = (ci[1] - ci[0]) / 2
    
    return DropStatistics(
        artifact=artifact,
        drop_count=drop_count,
        total_drops=total_drops,
        drop_rate=drop_rate,
        confidence_interval=ci,
        margin_of_error=moe,
        is_significant=moe <= margin_threshold,
    )


def calculate_confidence_score(total_drops: int, min_drops: int = 1000) -> float:
    """
    Calculate an overall confidence score based on sample size.
    
    Uses a logarithmic scale to provide diminishing returns for very large samples.
    
    Parameters
    ----------
    total_drops : int
        Total number of drops observed
    min_drops : int
        Minimum drops for full confidence (1.0)
    
    Returns
    -------
    float
        Confidence score from 0.0 to 1.0
    """
    if total_drops <= 0:
        return 0.0
    if total_drops >= min_drops:
        return 1.0
    
    # Logarithmic scaling for smooth progression
    return min(1.0, math.log1p(total_drops) / math.log1p(min_drops))


def analyze_mission_drops(
    ship: str,
    duration: str,
    level: int,
    target_artifact: Optional[str],
    drop_vector: Dict[str, float],
    min_total_drops: int = 100,
    margin_threshold: float = 0.10,
    average_capacity: int = 200,
) -> MissionStatistics:
    """
    Analyze statistical significance for a mission type's drop data.
    
    Parameters
    ----------
    ship : str
        Ship name
    duration : str
        Duration type
    level : int
        Mission level
    target_artifact : Optional[str]
        Target artifact
    drop_vector : Dict[str, float]
        Artifact -> drop count mapping
    min_total_drops : int
        Minimum total drops required for significance
    margin_threshold : float
        Maximum acceptable margin of error per artifact
    average_capacity : int
        Average mission capacity (for estimating mission count)
    
    Returns
    -------
    MissionStatistics
        Complete statistical analysis
    """
    total_drops = int(sum(drop_vector.values()))
    estimated_missions = max(1, total_drops // average_capacity) if average_capacity > 0 else 0
    
    artifact_stats = {}
    for artifact, count in drop_vector.items():
        stats = analyze_artifact_drops(
            artifact=artifact,
            drop_count=int(count),
            total_drops=total_drops,
            margin_threshold=margin_threshold,
        )
        artifact_stats[artifact] = stats
    
    confidence = calculate_confidence_score(total_drops, min_total_drops)
    
    return MissionStatistics(
        ship=ship,
        duration=duration,
        level=level,
        target_artifact=target_artifact,
        total_drops=total_drops,
        total_missions=estimated_missions,
        artifact_stats=artifact_stats,
        overall_confidence=confidence,
        meets_threshold=total_drops >= min_total_drops,
    )


def format_confidence_display(stats: MissionStatistics) -> str:
    """
    Format mission statistics for display.
    
    Parameters
    ----------
    stats : MissionStatistics
        The statistics to format
    
    Returns
    -------
    str
        Human-readable summary
    """
    status = "✓ Significant" if stats.meets_threshold else "⚠ Insufficient data"
    target = stats.target_artifact or "Any"
    
    lines = [
        f"{stats.ship} {stats.duration} L{stats.level} ({target})",
        f"  Total drops: {stats.total_drops:,} (~{stats.total_missions:,} missions)",
        f"  Confidence: {stats.overall_confidence:.0%}",
        f"  Status: {status}",
    ]
    
    return "\n".join(lines)


def get_recommended_threshold(
    total_unique_artifacts: int,
    desired_margin: float = 0.05,
) -> int:
    """
    Get recommended minimum sample size based on number of artifact types.
    
    More artifact types require larger samples for reliable rates.
    
    Parameters
    ----------
    total_unique_artifacts : int
        Number of unique artifact types that can drop
    desired_margin : float
        Desired margin of error
    
    Returns
    -------
    int
        Recommended minimum total drops
    """
    # Base requirement for a single proportion
    base = minimum_sample_size(margin_of_error=desired_margin)
    
    # Scale up for multiple comparisons (Bonferroni-like adjustment)
    # Use sqrt to avoid overly conservative estimates
    multiplier = math.sqrt(total_unique_artifacts) if total_unique_artifacts > 1 else 1
    
    return int(base * multiplier)
