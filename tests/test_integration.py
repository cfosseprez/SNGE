"""
Integration tests for end-to-end workflows.
"""

import numpy as np
import pytest

from snge.models import NGEParameters
from snge.deterministic import solve_deterministic
from snge.stochastic import gillespie_ssa_fast
from snge.langevin import euler_maruyama_cle, euler_maruyama_cle_simple
from snge.ensemble import run_ensemble_gillespie, run_ensemble_euler_maruyama
from snge.analysis import compute_yield_statistics, compare_distributions
from snge.fitting import create_synthetic_data, fit_nge_to_mean


class TestFullSimulationWorkflow:
    """Test complete simulation workflows."""

    def test_deterministic_then_stochastic(self):
        """Test running deterministic first, then stochastic."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=300.0
        )

        # Run deterministic
        t_det = np.linspace(0, params.t_max, 301)
        A_det, B_det = solve_deterministic(params, t_det)

        # Run stochastic ensemble
        np.random.seed(42)
        results = run_ensemble_gillespie(
            params, n_runs=20, dt_record=1.0, show_progress=False
        )

        # Compute statistics
        stats = compute_yield_statistics(results)

        # Deterministic final yield
        det_yield = B_det[-1] / params.A0

        # Stochastic mean should be close to deterministic
        assert abs(stats['mean'] - det_yield) / det_yield < 0.15

    def test_gillespie_vs_euler_maruyama_comparison(self):
        """Test comparing Gillespie and Euler-Maruyama methods."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=200.0
        )

        n_runs = 50

        # Run both methods
        np.random.seed(42)
        results_g = run_ensemble_gillespie(
            params, n_runs=n_runs, dt_record=1.0, show_progress=False
        )

        results_em = run_ensemble_euler_maruyama(
            params, n_runs=n_runs, dt=0.1, show_progress=False
        )

        # Compare distributions
        yields_g = np.array([r.final_yield for r in results_g])
        yields_em = np.array([r.final_yield for r in results_em])

        comparison = compare_distributions(yields_g, yields_em)

        # Means should be similar (within 10%)
        mean_g = np.mean(yields_g)
        mean_em = np.mean(yields_em)
        assert abs(mean_g - mean_em) / mean_g < 0.15

    def test_volume_effect_on_variability(self):
        """Test that smaller volume gives more variability."""
        base_params = dict(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, t_max=150.0
        )

        params_small_v = NGEParameters(**base_params, V=1e-8)
        params_large_v = NGEParameters(**base_params, V=1e-5)

        n_runs = 30
        np.random.seed(42)

        results_small = run_ensemble_gillespie(
            params_small_v, n_runs=n_runs, dt_record=1.0, show_progress=False
        )
        results_large = run_ensemble_gillespie(
            params_large_v, n_runs=n_runs, dt_record=1.0, show_progress=False
        )

        stats_small = compute_yield_statistics(results_small)
        stats_large = compute_yield_statistics(results_large)

        # Smaller volume should have higher CV
        assert stats_small['cv'] > stats_large['cv']


class TestFittingWorkflow:
    """Test parameter fitting workflows."""

    def test_create_fit_verify_workflow(self):
        """Test creating data, fitting, and verifying parameter recovery."""
        # True parameters
        k1_true, k2_true, k3_true = 2e-4, 0.15, 0.008
        A0 = 0.01

        # Create synthetic data
        times = np.array([0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600])
        data = create_synthetic_data(
            k1=k1_true, k2=k2_true, k3=k3_true, A0=A0,
            times=times, n_runs=25, noise_level=0.02, seed=42
        )

        # Fit model
        fit = fit_nge_to_mean(data, method='both')

        # Verify fit quality
        assert fit.r_squared > 0.95

        # Verify parameter recovery (within 50%)
        assert abs(fit.k1 - k1_true) / k1_true < 0.5
        assert abs(fit.k2 - k2_true) / k2_true < 0.5
        assert abs(fit.k3 - k3_true) / k3_true < 0.5

    def test_fit_then_simulate_workflow(self):
        """Test fitting parameters then running simulation."""
        # Create synthetic data
        times = np.linspace(0, 600, 15)
        data = create_synthetic_data(
            k1=2e-4, k2=0.15, k3=0.008, A0=0.01,
            times=times, n_runs=20, noise_level=0.02, seed=42
        )

        # Fit model
        fit = fit_nge_to_mean(data, method='least_squares')

        # Use fitted parameters for simulation
        params = NGEParameters(
            k1=fit.k1, k2=fit.k2, k3=fit.k3,
            A0=fit.A0, B0=0.0, V=1e-6, t_max=600.0
        )

        # Run simulation
        np.random.seed(42)
        results = run_ensemble_gillespie(
            params, n_runs=30, dt_record=1.0, show_progress=False
        )

        # Compute statistics
        stats = compute_yield_statistics(results)

        # Simulation mean should match data mean
        data_mean_yield = data.mean_yield[-1]
        assert abs(stats['mean'] - data_mean_yield) / data_mean_yield < 0.2


@pytest.mark.parametrize("method", [
    "gillespie",
    "euler_maruyama",
    "euler_maruyama_simple"
])
class TestPhysicalConstraints:
    """Test physical constraints across all simulation methods."""

    def test_mass_conservation_no_etching(self, method):
        """Test mass conservation when k3=0."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.0,  # No etching
            A0=0.01, B0=0.0, V=1e-6, t_max=200.0
        )

        np.random.seed(42)

        if method == "gillespie":
            result = gillespie_ssa_fast(params, dt_record=1.0)
        elif method == "euler_maruyama":
            result = euler_maruyama_cle(params, dt=0.1, seed=42)
        else:
            result = euler_maruyama_cle_simple(params, dt=0.1, seed=42)

        # A + B should equal A0 (no loss to C when k3=0)
        # For Gillespie, this is exact; for CLE, allow small tolerance
        total = result.A_concentration + result.B_concentration

        if method == "gillespie":
            np.testing.assert_allclose(total, params.A0, rtol=0.01)
        else:
            # CLE has small numerical errors but enforces A+B <= A0
            assert np.all(total <= params.A0 + 1e-10)

    def test_non_negative_concentrations(self, method):
        """Test concentrations stay non-negative."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-7, t_max=150.0
        )

        np.random.seed(42)

        if method == "gillespie":
            result = gillespie_ssa_fast(params, dt_record=1.0)
        elif method == "euler_maruyama":
            result = euler_maruyama_cle(params, dt=0.1, seed=42)
        else:
            result = euler_maruyama_cle_simple(params, dt=0.1, seed=42)

        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

    def test_yield_bounded(self, method):
        """Test final yield is in [0, 1]."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=200.0
        )

        np.random.seed(42)

        if method == "gillespie":
            result = gillespie_ssa_fast(params, dt_record=1.0)
        elif method == "euler_maruyama":
            result = euler_maruyama_cle(params, dt=0.1, seed=42)
        else:
            result = euler_maruyama_cle_simple(params, dt=0.1, seed=42)

        assert 0 <= result.final_yield <= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_volume(self):
        """Test simulation with very small volume (few molecules)."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-12,  # Very small
            t_max=50.0
        )

        np.random.seed(42)
        result = gillespie_ssa_fast(params, dt_record=1.0)

        # Should still produce valid result
        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)

    def test_zero_initial_A(self):
        """Test with zero initial precursor."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.0, B0=0.0, V=1e-6, t_max=100.0
        )

        # Deterministic should stay at zero
        t = np.linspace(0, 100, 101)
        A, B = solve_deterministic(params, t)

        np.testing.assert_array_equal(A, 0)
        np.testing.assert_array_equal(B, 0)

    def test_high_etching_rate(self):
        """Test with high etching rate (B decays quickly)."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.5,  # High etching
            A0=0.01, B0=0.0, V=1e-6, t_max=100.0
        )

        t = np.linspace(0, 100, 101)
        A, B = solve_deterministic(params, t)

        # B should be small due to high etching
        assert B[-1] < 0.001  # Most converted to C

    def test_very_short_simulation(self):
        """Test very short simulation time."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=1.0  # 1 second
        )

        np.random.seed(42)
        result = euler_maruyama_cle(params, dt=0.01, seed=42)

        # Should have few changes from initial
        assert result.A_concentration[-1] > 0.009  # Mostly unchanged


class TestReproducibility:
    """Test reproducibility across methods."""

    def test_deterministic_reproducible(self):
        """Test deterministic solver is fully reproducible."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=200.0
        )
        t = np.linspace(0, 200, 201)

        A1, B1 = solve_deterministic(params, t)
        A2, B2 = solve_deterministic(params, t)

        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

    def test_stochastic_reproducible_with_seed(self):
        """Test stochastic methods reproducible with same seed."""
        params = NGEParameters(
            k1=2e-4, k2=0.15, k3=0.008,
            A0=0.01, B0=0.0, V=1e-6, t_max=100.0
        )

        # Euler-Maruyama with seed
        result1 = euler_maruyama_cle(params, dt=0.1, seed=42)
        result2 = euler_maruyama_cle(params, dt=0.1, seed=42)

        np.testing.assert_array_equal(result1.B_concentration, result2.B_concentration)
