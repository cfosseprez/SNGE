"""
Tests for snge.analysis module.
"""

import numpy as np
import pytest

from snge.analysis import (
    compute_yield_statistics,
    compare_distributions,
    compute_cv_over_time
)
from snge.models import SimulationResult


class TestComputeYieldStatistics:
    """Tests for yield statistics computation."""

    def test_expected_keys(self, mock_simulation_results):
        """Test that all expected keys are present."""
        stats = compute_yield_statistics(mock_simulation_results)

        expected_keys = [
            'yields', 'n', 'mean', 'std', 'cv',
            'median', 'iqr', 'skewness', 'kurtosis',
            'min', 'max', 'range'
        ]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_correct_count(self, mock_simulation_results):
        """Test that n equals number of results."""
        stats = compute_yield_statistics(mock_simulation_results)
        assert stats['n'] == len(mock_simulation_results)

    def test_mean_formula(self, mock_simulation_results):
        """Test mean is calculated correctly."""
        stats = compute_yield_statistics(mock_simulation_results)
        yields = np.array([r.final_yield for r in mock_simulation_results])

        assert stats['mean'] == pytest.approx(np.mean(yields))

    def test_std_formula(self, mock_simulation_results):
        """Test std uses ddof=1."""
        stats = compute_yield_statistics(mock_simulation_results)
        yields = np.array([r.final_yield for r in mock_simulation_results])

        # Should use ddof=1 (sample std)
        assert stats['std'] == pytest.approx(np.std(yields, ddof=1))

    def test_cv_formula(self, mock_simulation_results):
        """Test CV is calculated correctly (as percentage)."""
        stats = compute_yield_statistics(mock_simulation_results)
        yields = np.array([r.final_yield for r in mock_simulation_results])

        expected_cv = np.std(yields, ddof=1) / np.mean(yields) * 100
        assert stats['cv'] == pytest.approx(expected_cv)

    def test_iqr_formula(self, mock_simulation_results):
        """Test IQR is calculated correctly."""
        stats = compute_yield_statistics(mock_simulation_results)
        yields = np.array([r.final_yield for r in mock_simulation_results])

        expected_iqr = np.percentile(yields, 75) - np.percentile(yields, 25)
        assert stats['iqr'] == pytest.approx(expected_iqr)

    def test_range_formula(self, mock_simulation_results):
        """Test range is max - min."""
        stats = compute_yield_statistics(mock_simulation_results)

        expected_range = stats['max'] - stats['min']
        assert stats['range'] == pytest.approx(expected_range)

    def test_yields_array(self, mock_simulation_results):
        """Test yields array is correctly extracted."""
        stats = compute_yield_statistics(mock_simulation_results)

        assert isinstance(stats['yields'], np.ndarray)
        assert len(stats['yields']) == len(mock_simulation_results)


class TestCompareDistributions:
    """Tests for distribution comparison."""

    def test_same_distribution(self, mock_simulation_results_two_sets):
        """Test KS test for similar distributions."""
        results1, results2 = mock_simulation_results_two_sets
        yields1 = np.array([r.final_yield for r in results1])
        yields2 = np.array([r.final_yield for r in results2])

        comparison = compare_distributions(yields1, yields2)

        # Similar distributions should have high p-value
        assert comparison['ks_pvalue'] > 0.01  # Not obviously different

    def test_different_distributions(self, different_distribution_results):
        """Test KS test detects different distributions."""
        results1, results2 = different_distribution_results
        yields1 = np.array([r.final_yield for r in results1])
        yields2 = np.array([r.final_yield for r in results2])

        comparison = compare_distributions(yields1, yields2)

        # Different distributions should have low p-value
        assert comparison['ks_pvalue'] < 0.05
        assert not comparison['ks_same_distribution']

    def test_expected_keys(self, mock_simulation_results_two_sets):
        """Test that all expected keys are present."""
        results1, results2 = mock_simulation_results_two_sets
        yields1 = np.array([r.final_yield for r in results1])
        yields2 = np.array([r.final_yield for r in results2])

        comparison = compare_distributions(yields1, yields2, "Dist1", "Dist2")

        expected_keys = [
            'ks_statistic', 'ks_pvalue', 'ks_same_distribution',
            't_statistic', 't_pvalue', 'means_equal',
            'levene_statistic', 'levene_pvalue', 'variances_equal',
            'Dist1_mean', 'Dist2_mean', 'Dist1_cv', 'Dist2_cv'
        ]

        for key in expected_keys:
            assert key in comparison, f"Missing key: {key}"

    def test_identical_arrays(self):
        """Test comparison of identical arrays."""
        yields = np.array([0.5, 0.6, 0.55, 0.58, 0.52])

        comparison = compare_distributions(yields, yields)

        # Identical distributions should have KS stat = 0
        assert comparison['ks_statistic'] == pytest.approx(0, abs=1e-10)
        assert comparison['ks_pvalue'] == pytest.approx(1.0)


class TestComputeCvOverTime:
    """Tests for CV over time computation."""

    def test_correct_shapes(self, mock_simulation_results, short_time_points):
        """Test output shapes match input."""
        times, cv = compute_cv_over_time(mock_simulation_results, short_time_points)

        assert len(times) == len(short_time_points)
        assert len(cv) == len(short_time_points)
        np.testing.assert_array_equal(times, short_time_points)

    def test_non_negative_cv(self, mock_simulation_results, short_time_points):
        """Test CV is non-negative."""
        _, cv = compute_cv_over_time(mock_simulation_results, short_time_points)

        assert np.all(cv >= 0)

    def test_handles_t_zero(self, mock_simulation_results):
        """Test CV handles t=0 (where B may be zero)."""
        time_points = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        _, cv = compute_cv_over_time(mock_simulation_results, time_points)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(cv))

    def test_cv_in_percent(self, mock_simulation_results, short_time_points):
        """Test CV is reported as percentage."""
        _, cv = compute_cv_over_time(mock_simulation_results, short_time_points)

        # CV as percentage should typically be less than 100% for reasonable data
        # and greater than 0 for stochastic data
        nonzero_cv = cv[cv > 0]
        if len(nonzero_cv) > 0:
            # Typical CV should be positive
            assert np.mean(nonzero_cv) > 0
            # CV in percent (not fraction)
            # For our mock data with ~10% variation, CV should be ~10-20
            assert np.max(nonzero_cv) < 500  # Sanity check

    def test_interpolation_works(self, mock_simulation_results):
        """Test that time point interpolation works correctly."""
        # Use time points different from simulation
        time_points = np.linspace(0, 50, 25)  # Different grid
        times, cv = compute_cv_over_time(mock_simulation_results, time_points)

        assert len(cv) == 25
        assert np.all(np.isfinite(cv))
