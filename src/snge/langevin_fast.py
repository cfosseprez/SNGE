"""
Accelerated Euler-Maruyama CLE implementations.

Provides:
- Numba JIT-compiled version (CPU, 10-50x faster)
- CuPy GPU version (for large ensembles, 100x+ faster)
"""

import numpy as np
from numba import njit, prange

from .models import NGEParameters, SimulationResult


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
