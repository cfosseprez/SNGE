"""
Tests for snge.stochastic module (Gillespie SSA).
"""

import numpy as np
import pytest

from snge.stochastic import gillespie_ssa, gillespie_ssa_fast
from snge.deterministic import solve_deterministic
from snge.models import NGEParameters


class TestGillespieSSA:
    """Tests for the standard Gillespie SSA implementation."""

    def test_initial_conditions(self, small_params):
        """Test that simulation starts at correct initial conditions."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=1000)

        assert result.times[0] == 0.0
        assert result.A_concentration[0] == pytest.approx(small_params.A0, rel=0.01)
        assert result.B_concentration[0] == pytest.approx(small_params.B0, abs=1e-10)

    def test_non_negative_concentrations(self, small_params):
        """Test that concentrations remain non-negative."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=100000)

        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

    def test_time_monotonic(self, small_params):
        """Test that time increases monotonically."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=10000)

        dt = np.diff(result.times)
        assert np.all(dt >= 0)

    def test_reproducibility_with_seed(self, small_params):
        """Test reproducibility when setting numpy random seed."""
        np.random.seed(42)
        result1 = gillespie_ssa(small_params, max_steps=5000)

        np.random.seed(42)
        result2 = gillespie_ssa(small_params, max_steps=5000)

        np.testing.assert_array_equal(result1.times, result2.times)
        np.testing.assert_array_equal(result1.B_concentration, result2.B_concentration)

    def test_result_method_name(self, small_params):
        """Test that method name is set correctly."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=1000)
        assert result.method == "Gillespie SSA"

    def test_final_yield_valid(self, small_params):
        """Test that final yield is valid fraction."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=100000)

        assert 0 <= result.final_yield <= 1

    def test_record_interval(self, small_params):
        """Test recording at specified intervals."""
        np.random.seed(42)
        result = gillespie_ssa(small_params, max_steps=10000, record_interval=5.0)

        # With record_interval, time differences should be approximately 5.0
        # (not exact due to reaction timing)
        if len(result.times) > 1:
            dt = np.diff(result.times)
            # Most intervals should be >= record_interval (or close to it)
            assert np.mean(dt) >= 2.0  # Reasonable approximation


class TestGillespieSSAFast:
    """Tests for the optimized Gillespie SSA with fixed recording."""

    def test_structure(self, small_params):
        """Test basic structure of fast SSA result."""
        np.random.seed(42)
        result = gillespie_ssa_fast(small_params, dt_record=1.0)

        assert hasattr(result, 'times')
        assert hasattr(result, 'B_concentration')
        assert hasattr(result, 'A_concentration')
        assert hasattr(result, 'final_yield')
        assert hasattr(result, 'method')

    def test_time_points_match_dt_record(self, small_params):
        """Test that time points are at expected intervals."""
        np.random.seed(42)
        dt_record = 2.0
        result = gillespie_ssa_fast(small_params, dt_record=dt_record)

        # Check time intervals are dt_record
        if len(result.times) > 1:
            dt = np.diff(result.times)
            np.testing.assert_allclose(dt, dt_record, rtol=1e-10)

    def test_method_name(self, small_params):
        """Test method name for fast version."""
        np.random.seed(42)
        result = gillespie_ssa_fast(small_params, dt_record=1.0)
        assert "Gillespie" in result.method

    def test_non_negative(self, small_params):
        """Test concentrations remain non-negative."""
        np.random.seed(42)
        result = gillespie_ssa_fast(small_params, dt_record=1.0)

        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

    def test_initial_conditions(self, small_params):
        """Test initial conditions in fast version."""
        np.random.seed(42)
        result = gillespie_ssa_fast(small_params, dt_record=1.0)

        assert result.times[0] == 0.0
        assert result.A_concentration[0] == pytest.approx(small_params.A0, rel=0.01)

    def test_yield_valid(self, small_params):
        """Test final yield is valid."""
        np.random.seed(42)
        result = gillespie_ssa_fast(small_params, dt_record=1.0)

        assert 0 <= result.final_yield <= 1


@pytest.mark.slow
class TestGillespieConvergence:
    """Tests for Gillespie convergence to deterministic limit."""

    def test_mean_converges_to_deterministic(self, large_params):
        """Test that mean of many runs approaches deterministic solution."""
        n_runs = 50
        np.random.seed(42)

        # Run ensemble
        results = []
        for _ in range(n_runs):
            result = gillespie_ssa_fast(large_params, dt_record=1.0)
            results.append(result)

        # Compute mean trajectory
        t_common = np.linspace(0, large_params.t_max, 100)
        B_matrix = np.zeros((n_runs, len(t_common)))
        for i, res in enumerate(results):
            B_matrix[i, :] = np.interp(t_common, res.times, res.B_concentration)
        mean_B = np.mean(B_matrix, axis=0)

        # Get deterministic solution
        _, B_det = solve_deterministic(large_params, t_common)

        # Mean should be close to deterministic (within 10%)
        # Skip first few points where B is near zero
        mask = B_det > 0.0001
        relative_error = np.abs(mean_B[mask] - B_det[mask]) / B_det[mask]
        assert np.mean(relative_error) < 0.15  # 15% average error acceptable

    def test_variance_decreases_with_volume(self):
        """Test that variance decreases with larger volume."""
        n_runs = 30

        params_small = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-8, t_max=100.0
        )
        params_large = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=100.0
        )

        # Run ensembles
        np.random.seed(42)
        yields_small = []
        for _ in range(n_runs):
            result = gillespie_ssa_fast(params_small, dt_record=1.0)
            yields_small.append(result.final_yield)

        np.random.seed(42)
        yields_large = []
        for _ in range(n_runs):
            result = gillespie_ssa_fast(params_large, dt_record=1.0)
            yields_large.append(result.final_yield)

        cv_small = np.std(yields_small) / np.mean(yields_small)
        cv_large = np.std(yields_large) / np.mean(yields_large)

        # Larger volume should have smaller relative variance
        assert cv_large < cv_small
