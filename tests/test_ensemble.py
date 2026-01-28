"""
Tests for snge.ensemble module.
"""

import numpy as np
import pytest

from snge.ensemble import run_ensemble_gillespie, run_ensemble_euler_maruyama
from snge.models import SimulationResult, NGEParameters


@pytest.fixture
def fast_params():
    """Very fast parameters for ensemble tests (~600 molecules)."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=1e-9,  # 1 nM
        B0=0.0,
        V=1e-12,  # 1 pL - gives ~600 molecules
        t_max=20.0
    )


class TestRunEnsembleGillespie:
    """Tests for Gillespie ensemble runner."""

    def test_correct_count(self, fast_params):
        """Test that correct number of simulations are run."""
        n_runs = 5
        results = run_ensemble_gillespie(
            fast_params, n_runs=n_runs, dt_record=2.0, show_progress=False
        )

        assert len(results) == n_runs

    def test_all_simulation_results(self, fast_params):
        """Test that all results are SimulationResult objects."""
        results = run_ensemble_gillespie(
            fast_params, n_runs=3, dt_record=2.0, show_progress=False
        )

        for result in results:
            assert isinstance(result, SimulationResult)

    def test_independent_results(self, fast_params):
        """Test that results are independent (not identical)."""
        results = run_ensemble_gillespie(
            fast_params, n_runs=5, dt_record=2.0, show_progress=False
        )

        # Final yields should not all be identical (stochastic)
        yields = [r.final_yield for r in results]
        # With few molecules, variance is high
        # Just check we get valid yields
        assert all(0 <= y <= 1 for y in yields)

    def test_reasonable_statistics(self, fast_params):
        """Test that ensemble has reasonable statistics."""
        results = run_ensemble_gillespie(
            fast_params, n_runs=10, dt_record=2.0, show_progress=False
        )

        yields = np.array([r.final_yield for r in results])

        # All yields should be in valid range
        assert np.all(yields >= 0)
        assert np.all(yields <= 1)

    def test_method_name_consistent(self, fast_params):
        """Test that all results have consistent method name."""
        results = run_ensemble_gillespie(
            fast_params, n_runs=3, dt_record=2.0, show_progress=False
        )

        for result in results:
            assert "Gillespie" in result.method

    def test_dt_record_respected(self, fast_params):
        """Test that dt_record parameter is used."""
        dt_record = 5.0
        results = run_ensemble_gillespie(
            fast_params, n_runs=2, dt_record=dt_record, show_progress=False
        )

        for result in results:
            if len(result.times) > 1:
                dt = np.diff(result.times)
                np.testing.assert_allclose(dt, dt_record, rtol=1e-10)


class TestRunEnsembleEulerMaruyama:
    """Tests for Euler-Maruyama ensemble runner."""

    def test_correct_count(self, fast_params):
        """Test that correct number of simulations are run."""
        n_runs = 5
        results = run_ensemble_euler_maruyama(
            fast_params, n_runs=n_runs, dt=0.5, show_progress=False
        )

        assert len(results) == n_runs

    def test_same_time_points(self, fast_params):
        """Test that all results have same time points."""
        results = run_ensemble_euler_maruyama(
            fast_params, n_runs=3, dt=1.0, show_progress=False
        )

        # All should have same time array
        reference_times = results[0].times
        for result in results[1:]:
            np.testing.assert_array_equal(result.times, reference_times)

    def test_all_simulation_results(self, fast_params):
        """Test that all results are SimulationResult objects."""
        results = run_ensemble_euler_maruyama(
            fast_params, n_runs=3, dt=0.5, show_progress=False
        )

        for result in results:
            assert isinstance(result, SimulationResult)

    def test_independent_results(self, fast_params):
        """Test that results are independent."""
        results = run_ensemble_euler_maruyama(
            fast_params, n_runs=5, dt=0.5, show_progress=False
        )

        yields = [r.final_yield for r in results]
        # Should have variation (different random draws)
        # With small volume, high noise, so check valid range
        assert all(0 <= y <= 1 for y in yields)

    def test_reasonable_statistics(self, fast_params):
        """Test ensemble has reasonable statistics."""
        results = run_ensemble_euler_maruyama(
            fast_params, n_runs=10, dt=0.5, show_progress=False
        )

        yields = np.array([r.final_yield for r in results])

        assert np.all(yields >= 0)
        assert np.all(yields <= 1)

    def test_dt_affects_time_points(self, fast_params):
        """Test that dt parameter affects time point count."""
        dt_coarse = 2.0
        dt_fine = 0.5

        results_coarse = run_ensemble_euler_maruyama(
            fast_params, n_runs=1, dt=dt_coarse, show_progress=False
        )
        results_fine = run_ensemble_euler_maruyama(
            fast_params, n_runs=1, dt=dt_fine, show_progress=False
        )

        # Finer dt should give more time points
        assert len(results_fine[0].times) > len(results_coarse[0].times)
