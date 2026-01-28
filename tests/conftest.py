"""
Shared pytest fixtures for SNGE test suite.

Note: The phenomenological SNGE model with σ(b) = σ₀/(b+ε) is DEPRECATED.
The paper's methodology uses Gillespie SSA where CV emerges naturally
from physical parameters without fitting to variance.
"""

import warnings
import numpy as np
import pytest

from snge.models import (
    NGEParameters,
    SimulationResult,
    DimensionlessParameters,
    SNGEResult,
    DimensionlessParametersPhenomenological,
)


@pytest.fixture
def small_params():
    """Small NGEParameters for fast tests (~600 molecules)."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=1e-9,  # 1 nM (low concentration for fast tests)
        B0=0.0,
        V=1e-12,  # 1 pL volume - gives ~600 molecules
        t_max=30.0  # Short time
    )


@pytest.fixture
def medium_params():
    """Medium system for validation tests (~6000 molecules)."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=1e-8,  # 10 nM
        B0=0.0,
        V=1e-12,  # 1 pL - gives ~6000 molecules
        t_max=100.0
    )


@pytest.fixture
def large_params():
    """Large system for convergence tests (~60k molecules)."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=1e-7,  # 100 nM
        B0=0.0,
        V=1e-12,  # 1 pL - gives ~60k molecules
        t_max=150.0
    )


@pytest.fixture
def zero_etching_params():
    """Parameters with k3=0 for mass conservation tests."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.0,
        A0=1e-9,  # 1 nM
        B0=0.0,
        V=1e-12,  # ~600 molecules
        t_max=50.0
    )


@pytest.fixture
def seeded_params():
    """Parameters with initial seeding (B0 > 0) to test seeding effect on CV."""
    return NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=1e-9,
        B0=1e-10,  # 10% seeding
        V=1e-12,
        t_max=30.0
    )


@pytest.fixture
def time_points():
    """Standard time array for tests."""
    return np.linspace(0, 300, 301)


@pytest.fixture
def short_time_points():
    """Short time array for fast tests."""
    return np.linspace(0, 50, 51)


@pytest.fixture
def mock_simulation_results(small_params):
    """Mock SimulationResult list for analysis tests."""
    np.random.seed(42)
    n_runs = 50
    n_points = 101
    times = np.linspace(0, small_params.t_max, n_points)

    results = []
    for i in range(n_runs):
        # Create synthetic trajectories with some variability
        base_yield = 0.6 + 0.1 * np.random.randn()
        B_conc = small_params.A0 * base_yield * (1 - np.exp(-0.02 * times))
        A_conc = small_params.A0 - B_conc

        results.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / small_params.A0,
            method="Mock"
        ))

    return results


@pytest.fixture
def mock_simulation_results_two_sets(small_params):
    """Two sets of mock results for distribution comparison tests."""
    np.random.seed(42)
    n_runs = 100
    n_points = 101
    times = np.linspace(0, small_params.t_max, n_points)

    results1 = []
    results2 = []

    # First set: mean yield ~0.6
    for i in range(n_runs):
        base_yield = 0.6 + 0.05 * np.random.randn()
        B_conc = small_params.A0 * base_yield * (1 - np.exp(-0.02 * times))
        A_conc = small_params.A0 - B_conc

        results1.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / small_params.A0,
            method="Mock1"
        ))

    # Second set: same distribution (for same-distribution test)
    np.random.seed(123)
    for i in range(n_runs):
        base_yield = 0.6 + 0.05 * np.random.randn()
        B_conc = small_params.A0 * base_yield * (1 - np.exp(-0.02 * times))
        A_conc = small_params.A0 - B_conc

        results2.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / small_params.A0,
            method="Mock2"
        ))

    return results1, results2


@pytest.fixture
def different_distribution_results(small_params):
    """Two sets of results from different distributions."""
    np.random.seed(42)
    n_runs = 100
    n_points = 101
    times = np.linspace(0, small_params.t_max, n_points)

    results1 = []
    results2 = []

    # First set: mean yield ~0.6, low variance
    for i in range(n_runs):
        base_yield = 0.6 + 0.02 * np.random.randn()
        B_conc = small_params.A0 * base_yield * (1 - np.exp(-0.02 * times))
        A_conc = small_params.A0 - B_conc

        results1.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / small_params.A0,
            method="Low variance"
        ))

    # Second set: mean yield ~0.4, high variance (clearly different)
    for i in range(n_runs):
        base_yield = 0.4 + 0.15 * np.random.randn()
        B_conc = small_params.A0 * max(0, base_yield) * (1 - np.exp(-0.02 * times))
        A_conc = small_params.A0 - B_conc

        results2.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_conc,
            A_concentration=A_conc,
            final_yield=B_conc[-1] / small_params.A0,
            method="High variance"
        ))

    return results1, results2


@pytest.fixture
def dimensionless_params(small_params):
    """DimensionlessParameters for kinetics tests (no phenomenological noise)."""
    return DimensionlessParameters.from_nge_parameters(small_params)


@pytest.fixture
def dimensionless_params_phenomenological():
    """
    DEPRECATED: DimensionlessParametersPhenomenological for legacy tests.

    WARNING: This uses the phenomenological noise model which is NOT
    the paper's methodology.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return DimensionlessParametersPhenomenological(
            alpha=0.025,      # k1/k3 = 2e-4 / 8e-3 = 0.025
            beta=0.01875,     # k2*A0/k3 = 0.15 * 1e-9 / 8e-3 ≈ 0.01875
            sigma0=0.05,      # Moderate noise (DEPRECATED parameter)
            epsilon=0.001,    # Regularization (DEPRECATED parameter)
            tau_max=0.24,     # k3 * t_max = 8e-3 * 30 = 0.24
            noise_model="inverse"  # DEPRECATED parameter
        )


@pytest.fixture
def dimensionless_params_high_noise():
    """
    DEPRECATED: High noise phenomenological parameters for stress testing.

    WARNING: This uses the phenomenological noise model which is NOT
    the paper's methodology.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return DimensionlessParametersPhenomenological(
            alpha=0.025,
            beta=0.01875,
            sigma0=0.3,       # High noise (DEPRECATED)
            epsilon=0.001,
            tau_max=0.24,
            noise_model="inverse"
        )


@pytest.fixture
def mock_snge_results():
    """Mock SNGEResult list for analysis tests."""
    np.random.seed(42)
    n_runs = 50
    n_points = 101
    tau_max = 0.24
    tau = np.linspace(0, tau_max, n_points)

    results = []
    for i in range(n_runs):
        # Create synthetic trajectories with some variability
        base_yield = 0.6 + 0.1 * np.random.randn()
        b = base_yield * (1 - np.exp(-10 * tau / tau_max))
        b = np.clip(b, 0, 1)

        results.append(SNGEResult(
            tau=tau.copy(),
            b=b,
            final_yield=b[-1],
            method="Mock"
        ))

    return results
