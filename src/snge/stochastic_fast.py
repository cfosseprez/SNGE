"""
Numba-accelerated Gillespie SSA for the NGE model.

Provides 10-100x speedup over pure Python implementation.
"""

import numpy as np
from numba import njit, prange

from .models import NGEParameters, SimulationResult


@njit(cache=True)
def _gillespie_kernel(N_A0: int, N_B0: int, k1: float, k2_eff: float, k3: float,
                      t_max: float, dt_record: float, n_records: int) -> tuple:
    """
    Numba-compiled Gillespie SSA kernel.

    Returns:
        times, N_A_record, N_B_record arrays
    """
    N_A = N_A0
    N_B = N_B0
    t = 0.0

    times = np.zeros(n_records)
    N_A_record = np.zeros(n_records)
    N_B_record = np.zeros(n_records)

    times[0] = t
    N_A_record[0] = N_A
    N_B_record[0] = N_B
    record_idx = 1
    next_record_time = dt_record

    while t < t_max:
        a1 = k1 * N_A
        a2 = k2_eff * N_A * N_B
        a3 = k3 * N_B
        a_total = a1 + a2 + a3

        if a_total <= 0:
            while record_idx < n_records:
                times[record_idx] = next_record_time
                N_A_record[record_idx] = N_A
                N_B_record[record_idx] = N_B
                record_idx += 1
                next_record_time += dt_record
            break

        tau = -np.log(np.random.random()) / a_total
        t += tau

        while t >= next_record_time and record_idx < n_records:
            times[record_idx] = next_record_time
            N_A_record[record_idx] = N_A
            N_B_record[record_idx] = N_B
            record_idx += 1
            next_record_time += dt_record

        if t > t_max:
            break

        r = np.random.random() * a_total
        if r < a1:
            N_A -= 1
            N_B += 1
        elif r < a1 + a2:
            N_A -= 1
            N_B += 1
        else:
            N_B -= 1

        if N_A < 0:
            N_A = 0
        if N_B < 0:
            N_B = 0

    return times[:record_idx], N_A_record[:record_idx], N_B_record[:record_idx]


def gillespie_ssa_numba(params: NGEParameters, dt_record: float = 1.0) -> SimulationResult:
    """
    Numba-accelerated Gillespie SSA.

    First call includes JIT compilation overhead (~1-2 seconds).
    Subsequent calls are 10-100x faster than pure Python.
    """
    N_Av = 6.022e23
    n_records = int(params.t_max / dt_record) + 1
    k2_eff = params.k2 / (params.V * N_Av)

    times, N_A_record, N_B_record = _gillespie_kernel(
        params.N_A0, params.N_B0,
        params.k1, k2_eff, params.k3,
        params.t_max, dt_record, n_records
    )

    A_conc = N_A_record / (params.V * N_Av)
    B_conc = N_B_record / (params.V * N_Av)
    final_yield = B_conc[-1] / params.A0 if len(B_conc) > 0 else 0.0

    return SimulationResult(
        times=times,
        B_concentration=B_conc,
        A_concentration=A_conc,
        final_yield=final_yield,
        method="Gillespie SSA"
    )


@njit(parallel=True, cache=True)
def _gillespie_ensemble_kernel(n_runs: int, N_A0: int, N_B0: int,
                                k1: float, k2_eff: float, k3: float,
                                t_max: float, dt_record: float, n_records: int,
                                V_N_Av: float, A0: float) -> tuple:
    """
    Parallel Gillespie ensemble kernel.

    Runs multiple independent simulations in parallel using Numba's prange.
    """
    # Store final yields and final B concentrations for each run
    final_yields = np.zeros(n_runs)

    # Store full trajectories (all runs have same time points)
    all_times = np.zeros((n_runs, n_records))
    all_B_conc = np.zeros((n_runs, n_records))
    all_A_conc = np.zeros((n_runs, n_records))
    actual_lengths = np.zeros(n_runs, dtype=np.int64)

    for run in prange(n_runs):
        N_A = N_A0
        N_B = N_B0
        t = 0.0

        all_times[run, 0] = t
        all_A_conc[run, 0] = N_A / V_N_Av
        all_B_conc[run, 0] = N_B / V_N_Av
        record_idx = 1
        next_record_time = dt_record

        while t < t_max:
            a1 = k1 * N_A
            a2 = k2_eff * N_A * N_B
            a3 = k3 * N_B
            a_total = a1 + a2 + a3

            if a_total <= 0:
                while record_idx < n_records:
                    all_times[run, record_idx] = next_record_time
                    all_A_conc[run, record_idx] = N_A / V_N_Av
                    all_B_conc[run, record_idx] = N_B / V_N_Av
                    record_idx += 1
                    next_record_time += dt_record
                break

            tau = -np.log(np.random.random()) / a_total
            t += tau

            while t >= next_record_time and record_idx < n_records:
                all_times[run, record_idx] = next_record_time
                all_A_conc[run, record_idx] = N_A / V_N_Av
                all_B_conc[run, record_idx] = N_B / V_N_Av
                record_idx += 1
                next_record_time += dt_record

            if t > t_max:
                break

            r = np.random.random() * a_total
            if r < a1:
                N_A -= 1
                N_B += 1
            elif r < a1 + a2:
                N_A -= 1
                N_B += 1
            else:
                N_B -= 1

            if N_A < 0:
                N_A = 0
            if N_B < 0:
                N_B = 0

        actual_lengths[run] = record_idx
        final_yields[run] = all_B_conc[run, record_idx - 1] / A0

    return all_times, all_A_conc, all_B_conc, final_yields, actual_lengths


def run_ensemble_gillespie_numba(params: NGEParameters,
                                  n_runs: int = 1000,
                                  dt_record: float = 1.0,
                                  show_progress: bool = True,
                                  batch_size: int = None) -> list:
    """
    Run ensemble of Gillespie SSA simulations using Numba parallelization.

    This runs all simulations in parallel across CPU cores.
    First call includes JIT compilation overhead.

    Args:
        params: NGE parameters
        n_runs: Number of independent simulations
        dt_record: Recording interval
        show_progress: Show progress bar between batches
        batch_size: Number of simulations per batch (for progress). None = auto

    Returns:
        List of SimulationResult objects
    """
    import time as time_module
    from tqdm import tqdm

    N_Av = 6.022e23
    n_records = int(params.t_max / dt_record) + 1
    k2_eff = params.k2 / (params.V * N_Av)
    V_N_Av = params.V * N_Av

    # Auto batch size: aim for ~10-20 progress updates
    if batch_size is None:
        batch_size = max(1, n_runs // 20)

    results = []

    if show_progress:
        # Run one small batch first to estimate time
        test_batch = min(10, n_runs)
        start = time_module.time()
        _gillespie_ensemble_kernel(
            test_batch, params.N_A0, params.N_B0,
            params.k1, k2_eff, params.k3,
            params.t_max, dt_record, n_records,
            V_N_Av, params.A0
        )
        elapsed = time_module.time() - start
        time_per_sim = elapsed / test_batch
        total_est = time_per_sim * n_runs
        print(f"Estimated time: {total_est:.1f}s ({time_per_sim*1000:.2f}ms per simulation)")

        # Run in batches with progress bar
        pbar = tqdm(total=n_runs, desc="Gillespie SSA", unit="runs")
        n_done = 0

        while n_done < n_runs:
            batch = min(batch_size, n_runs - n_done)

            all_times, all_A_conc, all_B_conc, final_yields, actual_lengths = _gillespie_ensemble_kernel(
                batch, params.N_A0, params.N_B0,
                params.k1, k2_eff, params.k3,
                params.t_max, dt_record, n_records,
                V_N_Av, params.A0
            )

            for i in range(batch):
                length = actual_lengths[i]
                results.append(SimulationResult(
                    times=all_times[i, :length].copy(),
                    B_concentration=all_B_conc[i, :length].copy(),
                    A_concentration=all_A_conc[i, :length].copy(),
                    final_yield=final_yields[i],
                    method="Gillespie SSA"
                ))

            n_done += batch
            pbar.update(batch)

        pbar.close()
    else:
        # Run all at once without progress
        all_times, all_A_conc, all_B_conc, final_yields, actual_lengths = _gillespie_ensemble_kernel(
            n_runs, params.N_A0, params.N_B0,
            params.k1, k2_eff, params.k3,
            params.t_max, dt_record, n_records,
            V_N_Av, params.A0
        )

        for i in range(n_runs):
            length = actual_lengths[i]
            results.append(SimulationResult(
                times=all_times[i, :length].copy(),
                B_concentration=all_B_conc[i, :length].copy(),
                A_concentration=all_A_conc[i, :length].copy(),
                final_yield=final_yields[i],
                method="Gillespie SSA"
            ))

    return results
