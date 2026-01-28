"""
Tests for snge.deterministic module.
"""

import numpy as np
import pytest

from snge.deterministic import nge_deterministic, solve_deterministic
from snge.models import NGEParameters


class TestNgeDeterministic:
    """Tests for the ODE function."""

    def test_ode_at_initial_condition(self, small_params):
        """Test ODE derivatives at initial condition."""
        y = np.array([small_params.A0, small_params.B0])
        dydt = nge_deterministic(y, 0, small_params)

        # At t=0 with B0=0:
        # dA/dt = -k1*A0 - k2*A0*0 = -k1*A0
        # dB/dt = k1*A0 + k2*A0*0 - k3*0 = k1*A0
        expected_dA = -small_params.k1 * small_params.A0
        expected_dB = small_params.k1 * small_params.A0

        assert dydt[0] == pytest.approx(expected_dA, rel=1e-10)
        assert dydt[1] == pytest.approx(expected_dB, rel=1e-10)

    def test_ode_at_equilibrium(self, small_params):
        """Test ODE approaches small values at long times."""
        # Solve for long time to approach quasi-equilibrium
        # Note: With etching (k3>0), true equilibrium is A=B=0
        t_points = np.linspace(0, 5000, 5001)
        A_sol, B_sol = solve_deterministic(small_params, t_points)

        # Get final state
        y_final = np.array([A_sol[-1], B_sol[-1]])
        dydt = nge_deterministic(y_final, t_points[-1], small_params)

        # Derivatives should be small (but not necessarily zero due to slow dynamics)
        # Use relative tolerance based on A0
        assert abs(dydt[0]) < small_params.A0 * 1e-3
        assert abs(dydt[1]) < small_params.A0 * 1e-3

    def test_ode_mass_balance(self, zero_etching_params):
        """Test mass conservation in ODE (no etching case)."""
        # Without etching, d(A+B)/dt should be 0
        A, B = 0.005, 0.003  # Some intermediate state
        y = np.array([A, B])
        dydt = nge_deterministic(y, 0, zero_etching_params)

        # dA/dt + dB/dt should be 0 when k3=0
        assert dydt[0] + dydt[1] == pytest.approx(0, abs=1e-15)

    def test_ode_returns_correct_shape(self, small_params):
        """Test ODE returns array with correct shape."""
        y = np.array([0.01, 0.0])
        dydt = nge_deterministic(y, 0, small_params)

        assert isinstance(dydt, np.ndarray)
        assert dydt.shape == (2,)


class TestSolveDeterministic:
    """Tests for the ODE solver."""

    def test_initial_conditions(self, small_params, short_time_points):
        """Test that solution starts at correct initial conditions."""
        A_sol, B_sol = solve_deterministic(small_params, short_time_points)

        assert A_sol[0] == pytest.approx(small_params.A0, rel=1e-10)
        assert B_sol[0] == pytest.approx(small_params.B0, rel=1e-10)

    def test_non_negative_concentrations(self, small_params, short_time_points):
        """Test that concentrations remain non-negative."""
        A_sol, B_sol = solve_deterministic(small_params, short_time_points)

        assert np.all(A_sol >= -1e-15)  # Allow tiny numerical errors
        assert np.all(B_sol >= -1e-15)

    def test_mass_conservation_no_etching(self, zero_etching_params, short_time_points):
        """Test mass conservation when k3=0."""
        A_sol, B_sol = solve_deterministic(zero_etching_params, short_time_points)

        # A + B should equal A0 at all times
        total = A_sol + B_sol
        np.testing.assert_allclose(total, zero_etching_params.A0, rtol=1e-6)

    def test_mass_decreases_with_etching(self, small_params, short_time_points):
        """Test that total mass (A+B) decreases with etching."""
        A_sol, B_sol = solve_deterministic(small_params, short_time_points)

        total = A_sol + B_sol
        # Final total should be less than initial (some B converted to C)
        assert total[-1] < total[0]

    def test_reproducibility(self, small_params, short_time_points):
        """Test that deterministic solver gives same result each time."""
        A_sol1, B_sol1 = solve_deterministic(small_params, short_time_points)
        A_sol2, B_sol2 = solve_deterministic(small_params, short_time_points)

        np.testing.assert_array_equal(A_sol1, A_sol2)
        np.testing.assert_array_equal(B_sol1, B_sol2)

    def test_A_decreases_monotonically(self, small_params, short_time_points):
        """Test that precursor A decreases monotonically."""
        A_sol, _ = solve_deterministic(small_params, short_time_points)

        # A should never increase
        dA = np.diff(A_sol)
        assert np.all(dA <= 1e-10)  # Allow tiny numerical errors

    def test_B_increases_initially(self, small_params):
        """Test that product B increases initially."""
        t_points = np.linspace(0, 10, 11)  # Short time
        _, B_sol = solve_deterministic(small_params, t_points)

        # B should increase from 0
        assert B_sol[-1] > B_sol[0]

    def test_solution_shapes(self, small_params, short_time_points):
        """Test that solutions have correct shapes."""
        A_sol, B_sol = solve_deterministic(small_params, short_time_points)

        assert A_sol.shape == short_time_points.shape
        assert B_sol.shape == short_time_points.shape

    def test_long_time_behavior(self, small_params):
        """Test long-time behavior: rate of change decreases."""
        t_points = np.linspace(0, 2000, 2001)
        A_sol, B_sol = solve_deterministic(small_params, t_points)

        # With etching, system slowly decays rather than reaching steady state
        # Check that rate of change is much smaller late vs early
        dB_early = np.abs(np.diff(B_sol[:100])).mean()
        dB_late = np.abs(np.diff(B_sol[-100:])).mean()

        # Late changes should be much smaller than early changes
        assert dB_late < dB_early * 0.1  # At least 10x slower

    def test_different_parameters(self):
        """Test solver works with different parameter values."""
        # Fast kinetics
        fast_params = NGEParameters(
            k1=1e-2, k2=1.0, k3=0.1,
            A0=0.01, B0=0.0, V=1e-6, t_max=50.0
        )
        t_points = np.linspace(0, 50, 51)
        A_sol, B_sol = solve_deterministic(fast_params, t_points)

        assert A_sol[0] == pytest.approx(0.01)
        assert B_sol[0] == pytest.approx(0.0)
        assert np.all(A_sol >= -1e-10)
        assert np.all(B_sol >= -1e-10)
