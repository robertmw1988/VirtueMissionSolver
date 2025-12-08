"""Tests for statistical significance analysis."""
from __future__ import annotations

import pytest

from Solver.statistical_significance import (
    calculate_wilson_interval,
    calculate_normal_margin_of_error,
    minimum_sample_size,
    analyze_artifact_drops,
    analyze_mission_drops,
    calculate_confidence_score,
    get_recommended_threshold,
    DropStatistics,
    MissionStatistics,
)


class TestWilsonInterval:
    """Tests for Wilson score confidence interval calculation."""
    
    def test_wilson_interval_50_percent(self):
        """50% proportion with 100 trials."""
        lower, upper = calculate_wilson_interval(50, 100, 0.95)
        # For 50/100, expect interval around [0.40, 0.60]
        assert 0.39 < lower < 0.42
        assert 0.58 < upper < 0.62
    
    def test_wilson_interval_zero_successes(self):
        """Zero successes should give valid interval."""
        lower, upper = calculate_wilson_interval(0, 100, 0.95)
        assert lower == 0.0
        assert 0.0 < upper < 0.1
    
    def test_wilson_interval_all_successes(self):
        """All successes should give valid interval."""
        lower, upper = calculate_wilson_interval(100, 100, 0.95)
        assert upper == pytest.approx(1.0, abs=1e-9)
        assert 0.9 < lower < 1.0
    
    def test_wilson_interval_zero_trials(self):
        """Zero trials should return full range."""
        lower, upper = calculate_wilson_interval(0, 0, 0.95)
        assert lower == 0.0
        assert upper == 1.0
    
    def test_wilson_interval_large_sample(self):
        """Large sample should give narrow interval."""
        lower, upper = calculate_wilson_interval(5000, 10000, 0.95)
        # Should be very close to 0.5
        assert 0.49 < lower < 0.50
        assert 0.50 < upper < 0.51


class TestMinimumSampleSize:
    """Tests for minimum sample size calculation."""
    
    def test_sample_size_5_percent_margin(self):
        """5% margin should require ~385 samples."""
        n = minimum_sample_size(0.05, 0.95, 0.5)
        assert 380 < n < 390
    
    def test_sample_size_1_percent_margin(self):
        """1% margin should require ~9604 samples."""
        n = minimum_sample_size(0.01, 0.95, 0.5)
        assert 9500 < n < 9700
    
    def test_sample_size_10_percent_margin(self):
        """10% margin should require ~97 samples."""
        n = minimum_sample_size(0.10, 0.95, 0.5)
        assert 90 < n < 100
    
    def test_sample_size_lower_proportion(self):
        """Lower estimated proportion needs fewer samples."""
        n_50 = minimum_sample_size(0.05, 0.95, 0.5)
        n_10 = minimum_sample_size(0.05, 0.95, 0.1)
        assert n_10 < n_50


class TestConfidenceScore:
    """Tests for confidence score calculation."""
    
    def test_zero_drops_zero_confidence(self):
        """Zero drops should give zero confidence."""
        assert calculate_confidence_score(0, 1000) == 0.0
    
    def test_min_drops_full_confidence(self):
        """Meeting minimum should give full confidence."""
        assert calculate_confidence_score(1000, 1000) == 1.0
    
    def test_exceeding_min_still_full(self):
        """Exceeding minimum should still be 1.0."""
        assert calculate_confidence_score(5000, 1000) == 1.0
    
    def test_partial_confidence(self):
        """Partial samples should give partial confidence."""
        score = calculate_confidence_score(500, 1000)
        assert 0.5 < score < 1.0


class TestAnalyzeArtifactDrops:
    """Tests for artifact drop analysis."""
    
    def test_basic_analysis(self):
        """Basic artifact analysis should work."""
        stats = analyze_artifact_drops("Test Artifact", 100, 1000)
        
        assert stats.artifact == "Test Artifact"
        assert stats.drop_count == 100
        assert stats.total_drops == 1000
        assert stats.drop_rate == 0.1
        assert 0 < stats.margin_of_error < 1
    
    def test_zero_drops_analysis(self):
        """Zero drops should not crash."""
        stats = analyze_artifact_drops("Test", 0, 0)
        
        assert stats.drop_rate == 0.0
        assert not stats.is_significant
    
    def test_significance_threshold(self):
        """Large sample should be significant."""
        stats = analyze_artifact_drops("Test", 1000, 10000, margin_threshold=0.10)
        assert stats.is_significant


class TestAnalyzeMissionDrops:
    """Tests for mission drop analysis."""
    
    def test_basic_mission_analysis(self):
        """Basic mission analysis should work."""
        drop_vector = {
            "Artifact A": 500.0,
            "Artifact B": 300.0,
            "Artifact C": 200.0,
        }
        
        stats = analyze_mission_drops(
            ship="Henerprise",
            duration="Epic",
            level=8,
            target_artifact="Book of Basan",
            drop_vector=drop_vector,
            min_total_drops=100,
        )
        
        assert stats.ship == "Henerprise"
        assert stats.total_drops == 1000
        assert stats.meets_threshold
        assert len(stats.artifact_stats) == 3
    
    def test_insufficient_data(self):
        """Insufficient data should be detected."""
        drop_vector = {"Artifact A": 10.0}
        
        stats = analyze_mission_drops(
            ship="Test",
            duration="Short",
            level=0,
            target_artifact=None,
            drop_vector=drop_vector,
            min_total_drops=1000,
        )
        
        assert not stats.meets_threshold
        assert stats.overall_confidence < 1.0


class TestRecommendedThreshold:
    """Tests for recommended threshold calculation."""
    
    def test_single_artifact(self):
        """Single artifact type."""
        threshold = get_recommended_threshold(1, 0.05)
        assert threshold > 0
    
    def test_multiple_artifacts(self):
        """More artifacts need larger sample."""
        t1 = get_recommended_threshold(10, 0.05)
        t2 = get_recommended_threshold(100, 0.05)
        assert t2 > t1
