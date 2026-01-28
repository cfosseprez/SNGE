"""
Accelerated Euler-Maruyama implementations for CLE.

Provides:
- Numba JIT-compiled version (CPU, 10-50x faster)
- CuPy GPU version (for large ensembles, 100x+ faster)

DEPRECATED: Phenomenological SNGE noise models
- The functions snge_euler_maruyama_numba and run_ensemble_snge_numba are
  DEPRECATED. They use phenomenological noise model σ(b) = σ₀/(b+ε) which
  is NOT the paper's methodology.

For the paper's approach, use:
    from snge.stochastic_fast import run_ensemble_gillespie_numba
    from snge.analysis import predict_cv_from_gillespie
"""

import warnings
import numpy as np
from numba import njit, prange

from .models import NGEParameters, SimulationResult, DimensionlessParameters, SNGEResult


@njit(cache=True)
def _euler_maruyama_kernel(A0: float, B0: float, k1: float, k2: float, k3: float,
                           inv_sqrt_V_NAv: float, t_max: float, dt: float,
                           n_steps: int) -> tuple:
    """
    Numba-compiled Euler-Maruyama kernel with proper 2-species dynamics.

    Tracks both A and B with their own SDEs (not mass conservation shortcut).
    """
    times = np.linspace(0, t_max, n_steps)
    A = np.zeros(n_steps)
    B = np.zeros(n_steps)

    A[0] = A0
    B[0] = B0

    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps - 1):
        A_curr = A[i]
        B_curr = B[i]

        # Reaction rates
        rate1 = k1 * A_curr          # nucleation: A -> B
        rate2 = k2 * A_curr * B_curr  # growth: A + B -> 2B
        rate3 = k3 * B_curr          # etching: B -> C

        # Drift terms
        drift_A = -rate1 - rate2
        drift_B = rate1 + rate2 - rate3

        # Generate correlated noise for each reaction channel
        xi1 = np.random.randn()  # nucleation noise
        xi2 = np.random.randn()  # growth noise
        xi3 = np.random.randn()  # etching noise

        # Noise contributions (properly scaled)
        # Each reaction contributes sqrt(rate) noise
        sqrt_rate1 = np.sqrt(max(0.0, rate1))
        sqrt_rate2 = np.sqrt(max(0.0, rate2))
        sqrt_rate3 = np.sqrt(max(0.0, rate3))

        noise_A = inv_sqrt_V_NAv * (-sqrt_rate1 * xi1 - sqrt_rate2 * xi2)
        noise_B = inv_sqrt_V_NAv * (sqrt_rate1 * xi1 + sqrt_rate2 * xi2 - sqrt_rate3 * xi3)

        # Euler-Maruyama update
        A_new = A_curr + drift_A * dt + noise_A * sqrt_dt
        B_new = B_curr + drift_B * dt + noise_B * sqrt_dt

        # Enforce physical bounds: A, B >= 0
        if A_new < 0:
            A_new = 0
        if B_new < 0:
            B_new = 0

        # Enforce A + B <= A0 (mass conservation - some goes to C)
        total = A_new + B_new
        if total > A0:
            scale = A0 / total
            A_new *= scale
            B_new *= scale

        A[i + 1] = A_new
        B[i + 1] = B_new

    return times, A, B


def euler_maruyama_numba(params: NGEParameters, dt: float = 0.1) -> SimulationResult:
    """
    Numba-accelerated Euler-Maruyama CLE.

    First call includes JIT compilation overhead.
    Subsequent calls are 10-50x faster than pure Python.
    """
    n_steps = int(params.t_max / dt) + 1
    N_Av = 6.022e23
    # Correct CLE noise scaling: σ = √(rate / (V * N_Av))
    inv_sqrt_V_NAv = 1.0 / np.sqrt(params.V * N_Av)

    times, A, B = _euler_maruyama_kernel(
        params.A0, params.B0,
        params.k1, params.k2, params.k3,
        inv_sqrt_V_NAv, params.t_max, dt, n_steps
    )

    final_yield = B[-1] / params.A0

    return SimulationResult(
        times=times,
        B_concentration=B,
        A_concentration=A,
        final_yield=final_yield,
        method="Langevin CLE"
    )


@njit(parallel=True, cache=True)
def _euler_maruyama_ensemble_kernel(n_runs: int, A0: float, B0: float,
                                     k1: float, k2: float, k3: float,
                                     inv_sqrt_V_NAv: float, t_max: float, dt: float,
                                     n_steps: int) -> tuple:
    """
    Parallel Euler-Maruyama ensemble kernel with proper 2-species dynamics.
    """
    sqrt_dt = np.sqrt(dt)

    # All runs share the same time points
    times = np.linspace(0, t_max, n_steps)

    all_A = np.zeros((n_runs, n_steps))
    all_B = np.zeros((n_runs, n_steps))
    final_yields = np.zeros(n_runs)

    for run in prange(n_runs):
        all_A[run, 0] = A0
        all_B[run, 0] = B0

        for i in range(n_steps - 1):
            A_curr = all_A[run, i]
            B_curr = all_B[run, i]

            # Reaction rates
            rate1 = k1 * A_curr          # nucleation: A -> B
            rate2 = k2 * A_curr * B_curr  # growth: A + B -> 2B
            rate3 = k3 * B_curr          # etching: B -> C

            # Drift terms
            drift_A = -rate1 - rate2
            drift_B = rate1 + rate2 - rate3

            # Generate noise for each reaction channel
            xi1 = np.random.randn()
            xi2 = np.random.randn()
            xi3 = np.random.randn()

            # Noise contributions
            sqrt_rate1 = np.sqrt(max(0.0, rate1))
            sqrt_rate2 = np.sqrt(max(0.0, rate2))
            sqrt_rate3 = np.sqrt(max(0.0, rate3))

            noise_A = inv_sqrt_V_NAv * (-sqrt_rate1 * xi1 - sqrt_rate2 * xi2)
            noise_B = inv_sqrt_V_NAv * (sqrt_rate1 * xi1 + sqrt_rate2 * xi2 - sqrt_rate3 * xi3)

            # Euler-Maruyama update
            A_new = A_curr + drift_A * dt + noise_A * sqrt_dt
            B_new = B_curr + drift_B * dt + noise_B * sqrt_dt

            # Enforce physical bounds
            if A_new < 0:
                A_new = 0
            if B_new < 0:
                B_new = 0

            # Enforce A + B <= A0 (some mass goes to C via etching)
            total = A_new + B_new
            if total > A0:
                scale = A0 / total
                A_new *= scale
                B_new *= scale

            all_A[run, i + 1] = A_new
            all_B[run, i + 1] = B_new

        final_yields[run] = all_B[run, -1] / A0

    return times, all_A, all_B, final_yields


def run_ensemble_euler_maruyama_numba(params: NGEParameters,
                                       n_runs: int = 1000,
                                       dt: float = 0.1,
                                       show_progress: bool = True,
                                       batch_size: int = None) -> list:
    """
    Run ensemble of Euler-Maruyama simulations using Numba parallelization.

    Args:
        params: NGE parameters
        n_runs: Number of independent simulations
        dt: Integration time step
        show_progress: Show progress bar between batches
        batch_size: Number of simulations per batch (for progress). None = auto

    Returns:
        List of SimulationResult objects
    """
    import time as time_module
    from tqdm import tqdm

    n_steps = int(params.t_max / dt) + 1
    N_Av = 6.022e23
    inv_sqrt_V_NAv = 1.0 / np.sqrt(params.V * N_Av)

    # Auto batch size
    if batch_size is None:
        batch_size = max(1, n_runs // 20)

    results = []

    if show_progress:
        # Estimate time with small test batch
        test_batch = min(10, n_runs)
        start = time_module.time()
        _euler_maruyama_ensemble_kernel(
            test_batch, params.A0, params.B0,
            params.k1, params.k2, params.k3,
            inv_sqrt_V_NAv, params.t_max, dt, n_steps
        )
        elapsed = time_module.time() - start
        time_per_sim = elapsed / test_batch
        total_est = time_per_sim * n_runs
        print(f"Estimated time: {total_est:.1f}s ({time_per_sim*1000:.2f}ms per simulation)")

        # Run in batches with progress bar
        pbar = tqdm(total=n_runs, desc="Langevin CLE", unit="runs")
        n_done = 0

        while n_done < n_runs:
            batch = min(batch_size, n_runs - n_done)

            times, all_A, all_B, final_yields = _euler_maruyama_ensemble_kernel(
                batch, params.A0, params.B0,
                params.k1, params.k2, params.k3,
                inv_sqrt_V_NAv, params.t_max, dt, n_steps
            )

            for i in range(batch):
                results.append(SimulationResult(
                    times=times.copy(),
                    B_concentration=all_B[i].copy(),
                    A_concentration=all_A[i].copy(),
                    final_yield=final_yields[i],
                    method="Langevin CLE"
                ))

            n_done += batch
            pbar.update(batch)

        pbar.close()
    else:
        # Run all at once
        times, all_A, all_B, final_yields = _euler_maruyama_ensemble_kernel(
            n_runs, params.A0, params.B0,
            params.k1, params.k2, params.k3,
            inv_sqrt_V_NAv, params.t_max, dt, n_steps
        )

        for i in range(n_runs):
            results.append(SimulationResult(
                times=times.copy(),
                B_concentration=all_B[i].copy(),
                A_concentration=all_A[i].copy(),
                final_yield=final_yields[i],
                method="Langevin CLE"
            ))

    return results


# =============================================================================
# GPU VERSION (CuPy) - Optional
# =============================================================================

def run_ensemble_euler_maruyama_gpu(params: NGEParameters,
                                     n_runs: int = 1000,
                                     dt: float = 0.1) -> list:
    """
    GPU-accelerated ensemble Euler-Maruyama using CuPy.

    Runs ALL trajectories simultaneously on GPU - massive parallelism.
    Best for large ensembles (1000+ runs).

    Requires: pip install cupy-cuda11x (or appropriate CUDA version)

    Args:
        params: NGE parameters
        n_runs: Number of independent simulations
        dt: Integration time step

    Returns:
        List of SimulationResult objects
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for GPU acceleration. Install with:\n"
            "  pip install cupy-cuda11x  (for CUDA 11.x)\n"
            "  pip install cupy-cuda12x  (for CUDA 12.x)\n"
            "Or see: https://docs.cupy.dev/en/stable/install.html"
        )

    n_steps = int(params.t_max / dt) + 1
    sqrt_dt = np.sqrt(dt)
    N_Av = 6.022e23
    # Correct CLE noise scaling: σ = √(rate / (V * N_Av))
    inv_sqrt_V_NAv = 1.0 / cp.sqrt(params.V * N_Av)

    # Initialize on GPU - all runs at once
    A = cp.zeros((n_runs, n_steps), dtype=cp.float64)
    B = cp.zeros((n_runs, n_steps), dtype=cp.float64)

    A[:, 0] = params.A0
    B[:, 0] = params.B0

    k1, k2, k3 = params.k1, params.k2, params.k3
    A0 = params.A0

    print(f"Running {n_runs} Euler-Maruyama simulations on GPU...")

    # Vectorized time-stepping across all runs
    for i in range(n_steps - 1):
        A_curr = A[:, i]
        B_curr = B[:, i]

        # Reaction rates
        rate1 = k1 * A_curr
        rate2 = k2 * A_curr * B_curr
        rate3 = k3 * B_curr

        # Drift terms
        drift_A = -rate1 - rate2
        drift_B = rate1 + rate2 - rate3

        # Noise for each reaction channel
        xi1 = cp.random.randn(n_runs)
        xi2 = cp.random.randn(n_runs)
        xi3 = cp.random.randn(n_runs)

        sqrt_rate1 = cp.sqrt(cp.maximum(rate1, 0))
        sqrt_rate2 = cp.sqrt(cp.maximum(rate2, 0))
        sqrt_rate3 = cp.sqrt(cp.maximum(rate3, 0))

        noise_A = inv_sqrt_V_NAv * (-sqrt_rate1 * xi1 - sqrt_rate2 * xi2)
        noise_B = inv_sqrt_V_NAv * (sqrt_rate1 * xi1 + sqrt_rate2 * xi2 - sqrt_rate3 * xi3)

        # Euler-Maruyama update
        A_new = A_curr + drift_A * dt + noise_A * sqrt_dt
        B_new = B_curr + drift_B * dt + noise_B * sqrt_dt

        # Enforce physical bounds
        A_new = cp.maximum(A_new, 0)
        B_new = cp.maximum(B_new, 0)

        # Enforce A + B <= A0
        total = A_new + B_new
        scale = cp.where(total > A0, A0 / total, 1.0)
        A[:, i + 1] = A_new * scale
        B[:, i + 1] = B_new * scale

    # Transfer results back to CPU
    times = np.linspace(0, params.t_max, n_steps)
    A_cpu = cp.asnumpy(A)
    B_cpu = cp.asnumpy(B)
    final_yields = B_cpu[:, -1] / params.A0

    results = []
    for i in range(n_runs):
        results.append(SimulationResult(
            times=times.copy(),
            B_concentration=B_cpu[i],
            A_concentration=A_cpu[i],
            final_yield=final_yields[i],
            method="Langevin CLE"
        ))

    return results


# =============================================================================
# DEPRECATED: Phenomenological State-Dependent Noise (Numba-accelerated)
# =============================================================================
#
# WARNING: The functions below use phenomenological noise model σ(b) = σ₀/(b+ε)
# which is NOT the paper's methodology.
#
# THE PAPER'S APPROACH:
# - Run Gillespie SSA with physical N = [A]₀ × V × Nₐ
# - CV emerges naturally from Poisson statistics of discrete molecular events
# - No parameters are fitted to variance data
#
# Use snge.stochastic_fast.run_ensemble_gillespie_numba() instead.
#

# Noise model constants for Numba (enums not supported)
NOISE_MODEL_INVERSE = 0
NOISE_MODEL_SQRT = 1


@njit(cache=True)
def _snge_euler_maruyama_kernel(alpha: float, beta: float, sigma0: float,
                                 epsilon: float, tau_max: float, dtau: float,
                                 b0: float, noise_model: int, n_steps: int) -> tuple:
    """
    Numba-compiled SNGE Euler-Maruyama kernel.

    The SDE: db = [α(1-b) + β(1-b)b - b]dτ + σ(b)dW

    where σ(b) depends on the noise model:
        inverse (0): σ(b) = σ₀ / (b + ε)
        sqrt (1):    σ(b) = σ₀ · √b · (1 + κ/b)

    Args:
        alpha: k₁/k₃ (nucleation-to-etching ratio)
        beta: k₂[A]₀/k₃ (growth-to-etching ratio)
        sigma0: Perturbation intensity
        epsilon: Regularization parameter
        tau_max: Maximum dimensionless time
        dtau: Dimensionless time step
        b0: Initial dimensionless yield
        noise_model: 0=inverse, 1=sqrt
        n_steps: Number of time steps

    Returns:
        (tau, b): Time and yield arrays
    """
    tau = np.linspace(0.0, tau_max, n_steps)
    b = np.zeros(n_steps)
    b[0] = b0

    sqrt_dtau = np.sqrt(dtau)

    for i in range(n_steps - 1):
        b_curr = b[i]

        # Drift term
        drift = alpha * (1 - b_curr) + beta * (1 - b_curr) * b_curr - b_curr

        # State-dependent noise
        if noise_model == NOISE_MODEL_INVERSE:
            sigma = sigma0 / (b_curr + epsilon)
        else:  # NOISE_MODEL_SQRT
            b_safe = max(b_curr, 1e-10)
            sigma = sigma0 * np.sqrt(b_safe) * (1 + epsilon / b_safe)

        # Random increment
        xi = np.random.randn()

        # Euler-Maruyama update
        b_new = b_curr + drift * dtau + sigma * sqrt_dtau * xi

        # Enforce bounds: 0 ≤ b ≤ 1
        if b_new < 0:
            b_new = 0
        if b_new > 1:
            b_new = 1

        b[i + 1] = b_new

    return tau, b


def euler_maruyama_phenomenological_numba(params, dtau: float = 0.001, b0: float = 0.0) -> SNGEResult:
    """
    DEPRECATED: Numba-accelerated phenomenological Euler-Maruyama.

    WARNING: This is NOT the paper's methodology. The phenomenological noise
    model σ(b) = σ₀/(b+ε) fits noise parameters to variance, contradicting
    the paper's approach where CV emerges naturally from Gillespie SSA.

    Use run_ensemble_gillespie_numba() instead.

    Args:
        params: DimensionlessParametersPhenomenological instance
        dtau: Dimensionless time step
        b0: Initial dimensionless yield

    Returns:
        SNGEResult with tau and b arrays
    """
    warnings.warn(
        "euler_maruyama_phenomenological_numba (formerly snge_euler_maruyama_numba) is DEPRECATED. "
        "This phenomenological noise model is NOT the paper's methodology. "
        "Use run_ensemble_gillespie_numba() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Check for required phenomenological parameters
    if not hasattr(params, 'sigma0') or not hasattr(params, 'epsilon'):
        raise ValueError(
            "euler_maruyama_phenomenological_numba requires DimensionlessParametersPhenomenological "
            "with sigma0, epsilon, and noise_model attributes."
        )

    n_steps = int(params.tau_max / dtau) + 1
    noise_model = NOISE_MODEL_INVERSE if params.noise_model == "inverse" else NOISE_MODEL_SQRT

    tau, b = _snge_euler_maruyama_kernel(
        params.alpha, params.beta, params.sigma0, params.epsilon,
        params.tau_max, dtau, b0, noise_model, n_steps
    )

    return SNGEResult(
        tau=tau,
        b=b,
        final_yield=b[-1],
        method="Phenomenological Euler-Maruyama (Numba, DEPRECATED)"
    )


# Backwards compatibility alias
def snge_euler_maruyama_numba(params, dtau: float = 0.001, b0: float = 0.0) -> SNGEResult:
    """DEPRECATED: Use euler_maruyama_phenomenological_numba instead."""
    return euler_maruyama_phenomenological_numba(params, dtau=dtau, b0=b0)


@njit(parallel=True, cache=True)
def _snge_ensemble_kernel(n_runs: int, alpha: float, beta: float, sigma0: float,
                          epsilon: float, tau_max: float, dtau: float,
                          b0: float, noise_model: int, n_steps: int) -> tuple:
    """
    Parallel SNGE ensemble kernel with Numba prange.

    Runs multiple independent SNGE trajectories in parallel.

    Args:
        n_runs: Number of independent simulations
        alpha, beta, sigma0, epsilon: SNGE parameters
        tau_max: Maximum dimensionless time
        dtau: Time step
        b0: Initial yield
        noise_model: 0=inverse, 1=sqrt
        n_steps: Number of time steps

    Returns:
        (tau, all_b, final_yields): Time array, all trajectories, final yields
    """
    sqrt_dtau = np.sqrt(dtau)

    # Shared time points
    tau = np.linspace(0.0, tau_max, n_steps)

    # Storage for all runs
    all_b = np.zeros((n_runs, n_steps))
    final_yields = np.zeros(n_runs)

    for run in prange(n_runs):
        all_b[run, 0] = b0

        for i in range(n_steps - 1):
            b_curr = all_b[run, i]

            # Drift term
            drift = alpha * (1 - b_curr) + beta * (1 - b_curr) * b_curr - b_curr

            # State-dependent noise
            if noise_model == NOISE_MODEL_INVERSE:
                sigma = sigma0 / (b_curr + epsilon)
            else:
                b_safe = max(b_curr, 1e-10)
                sigma = sigma0 * np.sqrt(b_safe) * (1 + epsilon / b_safe)

            # Random increment
            xi = np.random.randn()

            # Euler-Maruyama update with bounds enforcement
            b_new = b_curr + drift * dtau + sigma * sqrt_dtau * xi

            if b_new < 0:
                b_new = 0
            if b_new > 1:
                b_new = 1

            all_b[run, i + 1] = b_new

        final_yields[run] = all_b[run, -1]

    return tau, all_b, final_yields


def run_ensemble_phenomenological_numba(params,
                                         n_runs: int = 1000,
                                         dtau: float = 0.001,
                                         b0: float = 0.0,
                                         show_progress: bool = True,
                                         batch_size: int = None) -> list:
    """
    DEPRECATED: Run ensemble of phenomenological simulations using Numba.

    WARNING: This is NOT the paper's methodology. The phenomenological noise
    model σ(b) = σ₀/(b+ε) fits noise parameters to variance data.

    THE PAPER'S APPROACH:
    - Run Gillespie SSA with physical N = [A]₀ × V × Nₐ
    - CV emerges naturally - no fitting to variance data
    - Use run_ensemble_gillespie_numba() instead

    Args:
        params: DimensionlessParametersPhenomenological instance
        n_runs: Number of independent simulations
        dtau: Dimensionless time step
        b0: Initial dimensionless yield
        show_progress: Show progress bar between batches
        batch_size: Number of simulations per batch (for progress). None = auto

    Returns:
        List of SNGEResult objects
    """
    warnings.warn(
        "run_ensemble_phenomenological_numba (formerly run_ensemble_snge_numba) is DEPRECATED. "
        "This phenomenological noise model is NOT the paper's methodology. "
        "Use run_ensemble_gillespie_numba() instead, where CV emerges naturally.",
        DeprecationWarning,
        stacklevel=2
    )

    # Check for required phenomenological parameters
    if not hasattr(params, 'sigma0') or not hasattr(params, 'epsilon'):
        raise ValueError(
            "run_ensemble_phenomenological_numba requires DimensionlessParametersPhenomenological "
            "with sigma0, epsilon, and noise_model attributes."
        )

    import time as time_module
    from tqdm import tqdm

    n_steps = int(params.tau_max / dtau) + 1
    noise_model = NOISE_MODEL_INVERSE if params.noise_model == "inverse" else NOISE_MODEL_SQRT

    # Auto batch size
    if batch_size is None:
        batch_size = max(1, n_runs // 20)

    results = []

    if show_progress:
        # Estimate time with small test batch
        test_batch = min(10, n_runs)
        start = time_module.time()
        _snge_ensemble_kernel(
            test_batch, params.alpha, params.beta, params.sigma0, params.epsilon,
            params.tau_max, dtau, b0, noise_model, n_steps
        )
        elapsed = time_module.time() - start
        time_per_sim = elapsed / test_batch
        total_est = time_per_sim * n_runs
        print(f"Estimated time: {total_est:.1f}s ({time_per_sim*1000:.2f}ms per simulation)")

        # Run in batches with progress bar
        pbar = tqdm(total=n_runs, desc="SNGE", unit="runs")
        n_done = 0

        while n_done < n_runs:
            batch = min(batch_size, n_runs - n_done)

            tau, all_b, final_yields = _snge_ensemble_kernel(
                batch, params.alpha, params.beta, params.sigma0, params.epsilon,
                params.tau_max, dtau, b0, noise_model, n_steps
            )

            for i in range(batch):
                results.append(SNGEResult(
                    tau=tau.copy(),
                    b=all_b[i].copy(),
                    final_yield=final_yields[i],
                    method="Phenomenological Euler-Maruyama (Numba, DEPRECATED)"
                ))

            n_done += batch
            pbar.update(batch)

        pbar.close()
    else:
        # Run all at once
        tau, all_b, final_yields = _snge_ensemble_kernel(
            n_runs, params.alpha, params.beta, params.sigma0, params.epsilon,
            params.tau_max, dtau, b0, noise_model, n_steps
        )

        for i in range(n_runs):
            results.append(SNGEResult(
                tau=tau.copy(),
                b=all_b[i].copy(),
                final_yield=final_yields[i],
                method="Phenomenological Euler-Maruyama (Numba, DEPRECATED)"
            ))

    return results


# Backwards compatibility alias
def run_ensemble_snge_numba(params, n_runs: int = 1000, dtau: float = 0.001,
                            b0: float = 0.0, show_progress: bool = True,
                            batch_size: int = None) -> list:
    """DEPRECATED: Use run_ensemble_phenomenological_numba instead."""
    return run_ensemble_phenomenological_numba(
        params, n_runs=n_runs, dtau=dtau, b0=b0,
        show_progress=show_progress, batch_size=batch_size
    )
