"""
DEPRECATED Tests for phenomenological SNGE model.

WARNING: The phenomenological SNGE model with σ(b) = σ₀/(b+ε) is DEPRECATED.
This is NOT the paper's methodology. The paper uses Gillespie SSA where CV
emerges naturally from physical parameters.

These tests are retained for backwards compatibility only.
For the paper's methodology, see test_gillespie_predictions.py.
"""

import warnings
import numpy as np
import pytest

from snge.models import NGEParameters, DimensionlessParameters, DimensionlessParametersPhenomenological, SNGEResult
from snge.dimensionless import (
    snge_ode,
    solve_dimensionless,
    compute_steady_state,
    compute_steady_state_from_nge,
    compute_time_to_threshold,
    compute_max_growth_rate,
)


class TestDimensionlessODE:
    """Tests for dimensionless ODE solver (not deprecated)."""

    def test_snge_ode_at_zero(self):
        """Test ODE at b=0 gives nucleation rate."""
        alpha = 0.025
        beta = 0.1875
        rate = snge_ode(0.0, 0.0, alpha, beta)
        assert rate == pytest.approx(alpha, rel=1e-6)

    def test_snge_ode_at_steady_state(self):
        """Test ODE at steady state gives zero rate."""
        alpha = 0.025
        beta = 0.1875
        b_star = compute_steady_state(alpha, beta)
        rate = snge_ode(b_star, 0.0, alpha, beta)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_solve_dimensionless(self, dimensionless_params):
        """Test dimensionless solver runs and produces valid output."""
        result = solve_dimensionless(dimensionless_params, n_points=100)

        assert isinstance(result, SNGEResult)
        assert len(result.tau) == 100
        assert len(result.b) == 100
        assert result.tau[0] == 0.0
        assert result.b[0] == 0.0
        assert 0.0 <= result.final_yield <= 1.0

    def test_yield_approaches_steady_state(self, dimensionless_params):
        """Test that yield approaches theoretical steady state."""
        # Create params with longer time
        params = DimensionlessParameters(
            alpha=dimensionless_params.alpha,
            beta=dimensionless_params.beta,
            tau_max=20.0  # Long time
        )

        result = solve_dimensionless(params, n_points=1000)
        b_star = compute_steady_state(params.alpha, params.beta)

        # Final yield should be close to steady state
        assert result.final_yield == pytest.approx(b_star, rel=0.05)

    def test_compute_steady_state_no_growth(self):
        """Test steady state with beta=0 (no autocatalytic growth)."""
        alpha = 0.5
        beta = 0.0
        b_star = compute_steady_state(alpha, beta)

        # Should be alpha/(alpha+1)
        expected = alpha / (alpha + 1)
        assert b_star == pytest.approx(expected, rel=1e-6)


class TestDimensionlessParametersPhenomenological:
    """DEPRECATED: Tests for phenomenological parameters."""

    def test_creation_warns(self):
        """Test that creating phenomenological params warns."""
        with pytest.warns(DeprecationWarning):
            params = DimensionlessParametersPhenomenological(
                alpha=0.025,
                beta=0.1875,
                sigma0=0.05,
                epsilon=0.001,
                tau_max=4.0,
                noise_model="inverse"
            )

    def test_invalid_epsilon(self):
        """Test that epsilon must be positive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="epsilon must be positive"):
                DimensionlessParametersPhenomenological(
                    alpha=0.025, beta=0.1875, sigma0=0.05,
                    epsilon=0.0,  # Invalid
                    tau_max=4.0, noise_model="inverse"
                )

    def test_invalid_noise_model(self):
        """Test that noise model must be valid."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="noise_model"):
                DimensionlessParametersPhenomenological(
                    alpha=0.025, beta=0.1875, sigma0=0.05,
                    epsilon=0.001, tau_max=4.0,
                    noise_model="invalid"  # Invalid
                )


class TestPhenomenologicalLangevin:
    """DEPRECATED: Tests for phenomenological Langevin implementation."""

    def test_euler_maruyama_phenomenological_warns(self, dimensionless_params_phenomenological):
        """Test that phenomenological Euler-Maruyama warns."""
        from snge.langevin import euler_maruyama_phenomenological

        with pytest.warns(DeprecationWarning):
            result = euler_maruyama_phenomenological(
                dimensionless_params_phenomenological, dtau=0.01, seed=42
            )

        assert isinstance(result, SNGEResult)
        assert 0.0 <= result.final_yield <= 1.0

    def test_yield_bounds_enforced(self, dimensionless_params_high_noise):
        """Test that yield is always in [0, 1]."""
        from snge.langevin import euler_maruyama_phenomenological

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = euler_maruyama_phenomenological(
                dimensionless_params_high_noise, dtau=0.01, seed=42
            )

        assert np.all(result.b >= 0.0), "Yield went below 0"
        assert np.all(result.b <= 1.0), "Yield went above 1"


class TestPhenomenologicalNumba:
    """DEPRECATED: Tests for Numba-accelerated phenomenological implementation."""

    def test_numba_runs_and_warns(self, dimensionless_params_phenomenological):
        """Test that Numba phenomenological version runs and warns."""
        from snge.langevin_fast import euler_maruyama_phenomenological_numba

        with pytest.warns(DeprecationWarning):
            result = euler_maruyama_phenomenological_numba(
                dimensionless_params_phenomenological, dtau=0.01
            )

        assert isinstance(result, SNGEResult)
        assert 0.0 <= result.final_yield <= 1.0

    def test_ensemble_runs_and_warns(self, dimensionless_params_phenomenological):
        """Test that ensemble runner works and warns."""
        from snge.langevin_fast import run_ensemble_phenomenological_numba

        with pytest.warns(DeprecationWarning):
            results = run_ensemble_phenomenological_numba(
                dimensionless_params_phenomenological, n_runs=10, dtau=0.01, show_progress=False
            )

        assert len(results) == 10
        for r in results:
            assert isinstance(r, SNGEResult)
            assert 0.0 <= r.final_yield <= 1.0


class TestLegacyAliases:
    """Test that legacy aliases still work (with warnings)."""

    def test_snge_euler_maruyama_alias(self, dimensionless_params_phenomenological):
        """Test legacy snge_euler_maruyama alias."""
        from snge.langevin import snge_euler_maruyama

        with pytest.warns(DeprecationWarning):
            result = snge_euler_maruyama(
                dimensionless_params_phenomenological, dtau=0.01, seed=42
            )

        assert isinstance(result, SNGEResult)

    def test_snge_euler_maruyama_numba_alias(self, dimensionless_params_phenomenological):
        """Test legacy snge_euler_maruyama_numba alias."""
        from snge.langevin_fast import snge_euler_maruyama_numba

        with pytest.warns(DeprecationWarning):
            result = snge_euler_maruyama_numba(
                dimensionless_params_phenomenological, dtau=0.01
            )

        assert isinstance(result, SNGEResult)

    def test_run_ensemble_snge_numba_alias(self, dimensionless_params_phenomenological):
        """Test legacy run_ensemble_snge_numba alias."""
        from snge.langevin_fast import run_ensemble_snge_numba

        with pytest.warns(DeprecationWarning):
            results = run_ensemble_snge_numba(
                dimensionless_params_phenomenological, n_runs=5, dtau=0.01, show_progress=False
            )

        assert len(results) == 5


class TestSNGEResult:
    """Tests for SNGEResult dataclass (not deprecated)."""

    def test_creation(self):
        """Test basic result creation."""
        tau = np.linspace(0, 4, 100)
        b = np.linspace(0, 0.6, 100)

        result = SNGEResult(
            tau=tau,
            b=b,
            final_yield=0.6,
            method="Test"
        )

        np.testing.assert_array_equal(result.tau, tau)
        np.testing.assert_array_equal(result.b, b)
        assert result.final_yield == 0.6
        assert result.method == "Test"

    def test_to_dimensional(self, small_params):
        """Test conversion back to dimensional result."""
        tau = np.linspace(0, 4, 100)
        b = np.linspace(0, 0.6, 100)

        snge_result = SNGEResult(
            tau=tau, b=b, final_yield=0.6, method="Test"
        )

        sim_result = snge_result.to_dimensional(small_params)

        # Check time conversion
        expected_times = tau / small_params.k3
        np.testing.assert_array_almost_equal(sim_result.times, expected_times)

        # Check concentration conversion
        expected_B = b * small_params.A0
        np.testing.assert_array_almost_equal(sim_result.B_concentration, expected_B)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_alpha(self, dimensionless_params):
        """Test with alpha=0 (no nucleation)."""
        params = DimensionlessParameters(
            alpha=0.0,  # No nucleation
            beta=0.1875,
            tau_max=4.0
        )

        result = solve_dimensionless(params, n_points=100)

        # Without nucleation, yield should stay at 0 (starting from 0)
        assert result.final_yield == pytest.approx(0.0, abs=1e-6)

    def test_zero_beta(self, dimensionless_params):
        """Test with beta=0 (no autocatalytic growth)."""
        params = DimensionlessParameters(
            alpha=0.5,
            beta=0.0,  # No autocatalytic growth
            tau_max=10.0
        )

        result = solve_dimensionless(params, n_points=100)
        b_star = compute_steady_state(params.alpha, params.beta)

        # Should approach alpha/(alpha+1) = 0.5/1.5 ≈ 0.333
        assert result.final_yield == pytest.approx(b_star, rel=0.1)
