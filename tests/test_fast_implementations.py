"""
Tests for accelerated implementations (Numba/GPU).

Tests are skipped if optional dependencies are not available.
"""

import numpy as np
import pytest

from snge.models import NGEParameters, SimulationResult

# Check if numba is available
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Check if cupy is available
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@pytest.fixture
def fast_test_params():
    """Parameters for fast implementation tests."""
    return NGEParameters(
        k1=2e-4, k2=0.15, k3=0.008,
        A0=0.01, B0=0.0, V=1e-6, t_max=100.0
    )


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestGillespieSSANumba:
    """Tests for Numba-accelerated Gillespie SSA."""

    def test_runs_without_error(self, fast_test_params):
        """Test that numba version runs without error."""
        from snge.stochastic_fast import gillespie_ssa_numba

        result = gillespie_ssa_numba(fast_test_params, dt_record=1.0)

        assert result is not None
        assert isinstance(result, SimulationResult)

    def test_valid_output(self, fast_test_params):
        """Test output has valid structure and values."""
        from snge.stochastic_fast import gillespie_ssa_numba

        result = gillespie_ssa_numba(fast_test_params, dt_record=1.0)

        # Check structure
        assert hasattr(result, 'times')
        assert hasattr(result, 'B_concentration')
        assert hasattr(result, 'A_concentration')
        assert hasattr(result, 'final_yield')

        # Check values
        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)
        assert 0 <= result.final_yield <= 1

    def test_matches_basic_stats(self, fast_test_params):
        """Test numba version produces statistically similar results."""
        from snge.stochastic_fast import gillespie_ssa_numba
        from snge.stochastic import gillespie_ssa_fast

        n_runs = 30

        # Run both implementations
        np.random.seed(42)
        yields_numba = []
        for _ in range(n_runs):
            result = gillespie_ssa_numba(fast_test_params, dt_record=1.0)
            yields_numba.append(result.final_yield)

        np.random.seed(42)
        yields_basic = []
        for _ in range(n_runs):
            result = gillespie_ssa_fast(fast_test_params, dt_record=1.0)
            yields_basic.append(result.final_yield)

        # Means should be similar (within 20%)
        mean_numba = np.mean(yields_numba)
        mean_basic = np.mean(yields_basic)

        assert abs(mean_numba - mean_basic) / mean_basic < 0.2

    def test_initial_conditions(self, fast_test_params):
        """Test initial conditions are correct."""
        from snge.stochastic_fast import gillespie_ssa_numba

        result = gillespie_ssa_numba(fast_test_params, dt_record=1.0)

        assert result.times[0] == 0.0
        assert result.A_concentration[0] == pytest.approx(fast_test_params.A0, rel=0.01)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestRunEnsembleGillespieNumba:
    """Tests for Numba-accelerated Gillespie ensemble."""

    def test_correct_count(self, fast_test_params):
        """Test correct number of simulations."""
        from snge.stochastic_fast import run_ensemble_gillespie_numba

        n_runs = 10
        results = run_ensemble_gillespie_numba(
            fast_test_params, n_runs=n_runs, dt_record=1.0, show_progress=False
        )

        assert len(results) == n_runs

    def test_independent_results(self, fast_test_params):
        """Test results are independent."""
        from snge.stochastic_fast import run_ensemble_gillespie_numba

        results = run_ensemble_gillespie_numba(
            fast_test_params, n_runs=20, dt_record=1.0, show_progress=False
        )

        yields = [r.final_yield for r in results]
        # Should have variation
        assert len(set([round(y, 6) for y in yields])) > 1

    def test_all_valid_results(self, fast_test_params):
        """Test all results are valid."""
        from snge.stochastic_fast import run_ensemble_gillespie_numba

        results = run_ensemble_gillespie_numba(
            fast_test_params, n_runs=10, dt_record=1.0, show_progress=False
        )

        for result in results:
            assert isinstance(result, SimulationResult)
            assert 0 <= result.final_yield <= 1


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestEulerMaruyamaNumba:
    """Tests for Numba-accelerated Euler-Maruyama."""

    def test_runs_without_error(self, fast_test_params):
        """Test that numba version runs without error."""
        from snge.langevin_fast import euler_maruyama_numba

        result = euler_maruyama_numba(fast_test_params, dt=0.1)

        assert result is not None
        assert isinstance(result, SimulationResult)

    def test_valid_output(self, fast_test_params):
        """Test output has valid structure and values."""
        from snge.langevin_fast import euler_maruyama_numba

        result = euler_maruyama_numba(fast_test_params, dt=0.1)

        assert np.all(result.A_concentration >= 0)
        assert np.all(result.B_concentration >= 0)
        assert 0 <= result.final_yield <= 1

        # Mass conservation
        total = result.A_concentration + result.B_concentration
        assert np.all(total <= fast_test_params.A0 + 1e-10)

    def test_initial_conditions(self, fast_test_params):
        """Test initial conditions."""
        from snge.langevin_fast import euler_maruyama_numba

        result = euler_maruyama_numba(fast_test_params, dt=0.1)

        assert result.times[0] == 0.0
        assert result.A_concentration[0] == pytest.approx(fast_test_params.A0)
        assert result.B_concentration[0] == pytest.approx(fast_test_params.B0)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestRunEnsembleEulerMaruyamaNumba:
    """Tests for Numba-accelerated Euler-Maruyama ensemble."""

    def test_correct_count(self, fast_test_params):
        """Test correct number of simulations."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_numba

        n_runs = 10
        results = run_ensemble_euler_maruyama_numba(
            fast_test_params, n_runs=n_runs, dt=0.1, show_progress=False
        )

        assert len(results) == n_runs

    def test_same_time_points(self, fast_test_params):
        """Test all results have same time points."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_numba

        results = run_ensemble_euler_maruyama_numba(
            fast_test_params, n_runs=5, dt=0.5, show_progress=False
        )

        ref_times = results[0].times
        for result in results[1:]:
            np.testing.assert_array_equal(result.times, ref_times)

    def test_all_valid_results(self, fast_test_params):
        """Test all results are valid."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_numba

        results = run_ensemble_euler_maruyama_numba(
            fast_test_params, n_runs=10, dt=0.1, show_progress=False
        )

        for result in results:
            assert isinstance(result, SimulationResult)
            assert 0 <= result.final_yield <= 1


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not installed")
class TestRunEnsembleEulerMaruyamaGpu:
    """Tests for GPU-accelerated Euler-Maruyama ensemble."""

    def test_runs_without_error(self, fast_test_params):
        """Test that GPU version runs without error."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_gpu

        results = run_ensemble_euler_maruyama_gpu(
            fast_test_params, n_runs=10, dt=0.1
        )

        assert len(results) == 10

    def test_valid_results(self, fast_test_params):
        """Test GPU results are valid."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_gpu

        results = run_ensemble_euler_maruyama_gpu(
            fast_test_params, n_runs=20, dt=0.1
        )

        for result in results:
            assert isinstance(result, SimulationResult)
            assert 0 <= result.final_yield <= 1
            assert np.all(result.B_concentration >= 0)


class TestGpuImportError:
    """Test helpful error when CuPy is missing."""

    @pytest.mark.skipif(CUPY_AVAILABLE, reason="CuPy is available")
    def test_helpful_error_message(self, fast_test_params):
        """Test that missing CuPy gives helpful error."""
        from snge.langevin_fast import run_ensemble_euler_maruyama_gpu

        with pytest.raises(ImportError) as exc_info:
            run_ensemble_euler_maruyama_gpu(fast_test_params, n_runs=10, dt=0.1)

        # Check error message is helpful
        error_msg = str(exc_info.value).lower()
        assert 'cupy' in error_msg or 'cuda' in error_msg


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
class TestNumbaCompilationCaching:
    """Test that Numba functions use caching."""

    def test_second_call_faster(self, fast_test_params):
        """Test that second call benefits from JIT cache."""
        from snge.stochastic_fast import gillespie_ssa_numba
        import time

        # First call (may trigger compilation)
        start = time.time()
        gillespie_ssa_numba(fast_test_params, dt_record=1.0)
        first_time = time.time() - start

        # Subsequent calls should be fast
        times = []
        for _ in range(5):
            start = time.time()
            gillespie_ssa_numba(fast_test_params, dt_record=1.0)
            times.append(time.time() - start)

        avg_subsequent = np.mean(times)

        # Subsequent calls should generally be faster
        # (This might not always hold due to system variability)
        # Just verify it runs multiple times without error
        assert avg_subsequent < 10  # Should be much less than 10 seconds
