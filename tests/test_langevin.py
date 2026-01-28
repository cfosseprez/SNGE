"""
Tests for snge.langevin module (Euler-Maruyama CLE).
"""

import numpy as np
import pytest

from snge.langevin import euler_maruyama_cle, euler_maruyama_cle_simple
from snge.deterministic import solve_deterministic
from snge.models import NGEParameters


class TestEulerMaruyamaCLE:
    """Tests for the full Euler-Maruyama CLE implementation."""

    def test_initial_conditions(self, small_params):
        """Test simulation starts at correct initial conditions."""
        result = euler_maruyama_cle(small_params, dt=0.1, seed=42)

        assert result.times[0] == 0.0
        assert result.A_concentration[0] == pytest.approx(small_params.A0)
        assert result.B_concentration[0] == pytest.approx(small_params.B0)

    def test_bounded_concentrations(self, small_params):
        """Test concentrations stay within physical bounds."""
        result = euler_maruyama_cle(small_params, dt=0.1, seed=42)

        # All concentrations should be non-negative
        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

        # A + B should not exceed A0 (mass conservation)
        total = result.A_concentration + result.B_concentration
        assert np.all(total <= small_params.A0 + 1e-10)

    def test_reproducibility_with_seed(self, small_params):
        """Test reproducibility when using seed."""
        result1 = euler_maruyama_cle(small_params, dt=0.1, seed=42)
        result2 = euler_maruyama_cle(small_params, dt=0.1, seed=42)

        np.testing.assert_array_equal(result1.times, result2.times)
        np.testing.assert_array_equal(result1.B_concentration, result2.B_concentration)
        np.testing.assert_array_equal(result1.A_concentration, result2.A_concentration)

    def test_different_seeds_give_different_results(self, medium_params):
        """Test that different seeds produce different trajectories."""
        # Use medium_params for more molecules, less noise clamping
        result1 = euler_maruyama_cle(medium_params, dt=0.1, seed=42)
        result2 = euler_maruyama_cle(medium_params, dt=0.1, seed=123)

        # Results should differ (at least at some points)
        # Allow for both to be clamped at bounds but differ in details
        assert not np.array_equal(result1.B_concentration, result2.B_concentration)

    def test_method_name(self, small_params):
        """Test method name is set correctly."""
        result = euler_maruyama_cle(small_params, dt=0.1, seed=42)
        assert "Euler-Maruyama" in result.method or "CLE" in result.method

    def test_final_yield_valid(self, small_params):
        """Test final yield is in valid range."""
        result = euler_maruyama_cle(small_params, dt=0.1, seed=42)
        assert 0 <= result.final_yield <= 1

    def test_time_array_correct(self, small_params):
        """Test time array is correctly generated."""
        dt = 0.5
        result = euler_maruyama_cle(small_params, dt=dt, seed=42)

        expected_n_steps = int(small_params.t_max / dt) + 1
        assert len(result.times) == expected_n_steps

        # Time should increase uniformly
        dt_actual = np.diff(result.times)
        np.testing.assert_allclose(dt_actual, dt, rtol=1e-10)


class TestEulerMaruyamaCLESimple:
    """Tests for the simplified Euler-Maruyama CLE."""

    def test_mass_conservation(self, small_params):
        """Test that A + B = A0 (simplified version uses mass conservation)."""
        result = euler_maruyama_cle_simple(small_params, dt=0.1, seed=42)

        total = result.A_concentration + result.B_concentration
        # Use relative tolerance appropriate for small concentrations
        np.testing.assert_allclose(total, small_params.A0, rtol=1e-6)

    def test_matches_full_version_statistics(self, medium_params):
        """Test that simple and full versions produce similar distributions."""
        n_runs = 50

        yields_full = []
        yields_simple = []

        for i in range(n_runs):
            result_full = euler_maruyama_cle(medium_params, dt=0.1, seed=i)
            result_simple = euler_maruyama_cle_simple(medium_params, dt=0.1, seed=i + 1000)
            yields_full.append(result_full.final_yield)
            yields_simple.append(result_simple.final_yield)

        # Means should be similar (within 20%)
        mean_full = np.mean(yields_full)
        mean_simple = np.mean(yields_simple)
        assert abs(mean_full - mean_simple) / mean_full < 0.2

    def test_initial_conditions(self, small_params):
        """Test initial conditions in simple version."""
        result = euler_maruyama_cle_simple(small_params, dt=0.1, seed=42)

        assert result.A_concentration[0] == pytest.approx(small_params.A0)
        assert result.B_concentration[0] == pytest.approx(small_params.B0)

    def test_non_negative(self, small_params):
        """Test concentrations are non-negative."""
        result = euler_maruyama_cle_simple(small_params, dt=0.1, seed=42)

        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

    def test_method_name(self, small_params):
        """Test method name contains identifying info."""
        result = euler_maruyama_cle_simple(small_params, dt=0.1, seed=42)
        assert "simple" in result.method.lower() or "Euler-Maruyama" in result.method


@pytest.mark.slow
class TestEulerMaruyamaConvergence:
    """Tests for convergence to deterministic limit."""

    def test_large_volume_approaches_deterministic(self, large_params):
        """Test that large volume simulation approaches deterministic."""
        # Single run with large volume should be close to deterministic
        result = euler_maruyama_cle(large_params, dt=0.1, seed=42)

        # Get deterministic solution at same time points
        _, B_det = solve_deterministic(large_params, result.times)

        # Relative error should be small for large volume
        mask = B_det > 0.0001  # Avoid division by zero
        relative_error = np.abs(result.B_concentration[mask] - B_det[mask]) / B_det[mask]

        # Should be within 10% on average for large system
        assert np.mean(relative_error) < 0.1

    def test_ensemble_mean_approaches_deterministic(self, medium_params):
        """Test ensemble mean approaches deterministic solution."""
        n_runs = 100

        # Run ensemble
        B_trajectories = []
        for i in range(n_runs):
            result = euler_maruyama_cle(medium_params, dt=0.1, seed=i)
            B_trajectories.append(result.B_concentration)

        mean_B = np.mean(B_trajectories, axis=0)

        # Get deterministic
        t_points = np.linspace(0, medium_params.t_max, int(medium_params.t_max / 0.1) + 1)
        _, B_det = solve_deterministic(medium_params, t_points)

        # Mean should be close to deterministic
        mask = B_det > 0.001
        relative_error = np.abs(mean_B[mask] - B_det[mask]) / B_det[mask]
        assert np.mean(relative_error) < 0.05  # 5% error for ensemble mean

    def test_variance_scaling_with_volume(self):
        """Test that variance scales inversely with volume."""
        n_runs = 50

        params_small_v = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-7, t_max=100.0
        )
        params_large_v = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-5, t_max=100.0
        )

        yields_small_v = [euler_maruyama_cle(params_small_v, dt=0.1, seed=i).final_yield
                         for i in range(n_runs)]
        yields_large_v = [euler_maruyama_cle(params_large_v, dt=0.1, seed=i).final_yield
                         for i in range(n_runs)]

        var_small = np.var(yields_small_v)
        var_large = np.var(yields_large_v)

        # Variance should be smaller for larger volume
        # Theoretically scales as 1/V, so ratio should be ~100
        # Use loose bound since stochastic
        assert var_large < var_small
        assert var_small / var_large > 10  # At least 10x smaller variance
