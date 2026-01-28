"""
Tests for snge.models module.
"""

import numpy as np
import pytest

from snge.models import NGEParameters, SimulationResult


class TestNGEParameters:
    """Tests for NGEParameters dataclass."""

    def test_creation(self):
        """Test basic parameter creation."""
        params = NGEParameters(
            k1=1e-4,
            k2=0.1,
            k3=0.01,
            A0=0.01,
            B0=0.0,
            V=1e-6,
            t_max=300.0
        )

        assert params.k1 == 1e-4
        assert params.k2 == 0.1
        assert params.k3 == 0.01
        assert params.A0 == 0.01
        assert params.B0 == 0.0
        assert params.V == 1e-6
        assert params.t_max == 300.0

    def test_N_A0_property(self, small_params):
        """Test N_A0 calculates initial molecule count correctly."""
        # N_A0 = A0 * V * N_Av
        expected = int(small_params.A0 * small_params.V * 6.022e23)
        assert small_params.N_A0 == expected
        assert small_params.N_A0 > 0

    def test_N_B0_property(self, small_params):
        """Test N_B0 calculates initial B molecule count correctly."""
        # With B0=0, N_B0 should be 0
        assert small_params.N_B0 == 0

        # With non-zero B0
        params_with_B0 = NGEParameters(
            k1=1e-4, k2=0.1, k3=0.01,
            A0=0.01, B0=0.001,  # Non-zero B0
            V=1e-6, t_max=100.0
        )
        expected = int(0.001 * 1e-6 * 6.022e23)
        assert params_with_B0.N_B0 == expected

    def test_volume_scaling(self):
        """Test that molecule counts scale with volume."""
        params_small = NGEParameters(
            k1=1e-4, k2=0.1, k3=0.01,
            A0=0.01, B0=0.0,
            V=1e-9, t_max=100.0
        )
        params_large = NGEParameters(
            k1=1e-4, k2=0.1, k3=0.01,
            A0=0.01, B0=0.0,
            V=1e-6, t_max=100.0
        )

        # Large volume should have 1000x more molecules
        ratio = params_large.N_A0 / params_small.N_A0
        assert ratio == pytest.approx(1000, rel=0.01)

    def test_avogadro_number_used(self):
        """Test that Avogadro's number is used correctly."""
        params = NGEParameters(
            k1=1e-4, k2=0.1, k3=0.01,
            A0=1.0,  # 1 M concentration
            B0=0.0,
            V=1.0,   # 1 L volume
            t_max=100.0
        )
        # Should give approximately Avogadro's number
        assert params.N_A0 == pytest.approx(6.022e23, rel=0.001)


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        times = np.array([0.0, 1.0, 2.0])
        B_conc = np.array([0.0, 0.005, 0.008])
        A_conc = np.array([0.01, 0.005, 0.002])

        result = SimulationResult(
            times=times,
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=0.8,
            method="Test"
        )

        np.testing.assert_array_equal(result.times, times)
        np.testing.assert_array_equal(result.B_concentration, B_conc)
        np.testing.assert_array_equal(result.A_concentration, A_conc)
        assert result.final_yield == 0.8
        assert result.method == "Test"

    def test_result_attributes(self):
        """Test all attributes are accessible."""
        times = np.linspace(0, 100, 101)
        B_conc = np.linspace(0, 0.008, 101)
        A_conc = 0.01 - B_conc

        result = SimulationResult(
            times=times,
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=0.8,
            method="Gillespie SSA"
        )

        assert len(result.times) == 101
        assert len(result.B_concentration) == 101
        assert len(result.A_concentration) == 101
        assert result.final_yield == 0.8
        assert result.method == "Gillespie SSA"

    def test_empty_arrays(self):
        """Test handling of empty arrays (edge case)."""
        result = SimulationResult(
            times=np.array([]),
            B_concentration=np.array([]),
            A_concentration=np.array([]),
            final_yield=0.0,
            method="Empty"
        )

        assert len(result.times) == 0
        assert result.final_yield == 0.0

    def test_single_point(self):
        """Test result with single time point."""
        result = SimulationResult(
            times=np.array([0.0]),
            B_concentration=np.array([0.0]),
            A_concentration=np.array([0.01]),
            final_yield=0.0,
            method="Single"
        )

        assert len(result.times) == 1
        assert result.times[0] == 0.0
        assert result.B_concentration[0] == 0.0

    def test_yield_range(self):
        """Test that yield can span valid range [0, 1]."""
        # Zero yield
        result_zero = SimulationResult(
            times=np.array([0.0, 1.0]),
            B_concentration=np.array([0.0, 0.0]),
            A_concentration=np.array([0.01, 0.01]),
            final_yield=0.0,
            method="Zero yield"
        )
        assert result_zero.final_yield == 0.0

        # Full yield
        result_full = SimulationResult(
            times=np.array([0.0, 1.0]),
            B_concentration=np.array([0.0, 0.01]),
            A_concentration=np.array([0.01, 0.0]),
            final_yield=1.0,
            method="Full yield"
        )
        assert result_full.final_yield == 1.0
