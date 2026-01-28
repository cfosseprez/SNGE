"""
Gillespie Stochastic Simulation Algorithm (SSA) for the NGE model.
"""

import numpy as np

from .models import NGEParameters, SimulationResult


def gillespie_ssa(params: NGEParameters,
                  max_steps: int = 10000000,
                  record_interval: float = None) -> SimulationResult:
    """
    Gillespie Stochastic Simulation Algorithm for NGE model.

    This is the exact stochastic simulation - each reaction event is
    individually sampled from the correct probability distribution.

    Reactions:
        R1: A → B       (nucleation)     propensity a1 = k1 * N_A
        R2: A + B → 2B  (growth)         propensity a2 = k2 * N_A * N_B / (V * N_Av)
        R3: B → C       (etching)        propensity a3 = k3 * N_B

    Args:
        params: NGE model parameters
        max_steps: Maximum number of reaction events
        record_interval: Time interval for recording (None = record all events)

    Returns:
        SimulationResult with time series and final yield
    """
    # Initialize
    N_Av = 6.022e23  # Avogadro's number
    N_A = params.N_A0
    N_B = params.N_B0
    t = 0.0

    # Storage for trajectory
    times = [t]
    N_A_history = [N_A]
    N_B_history = [N_B]

    # Precompute constants
    k1 = params.k1
    k2_eff = params.k2 / (params.V * N_Av)  # Convert to molecular units
    k3 = params.k3

    step = 0
    last_record_time = 0.0

    while t < params.t_max and step < max_steps:
        # Calculate propensities
        a1 = k1 * N_A  # Nucleation
        a2 = k2_eff * N_A * N_B  # Growth
        a3 = k3 * N_B  # Etching
        a_total = a1 + a2 + a3

        # If no reactions possible, we're done
        if a_total <= 0:
            break

        # Time to next reaction (exponential distribution)
        tau = np.random.exponential(1.0 / a_total)
        t += tau

        if t > params.t_max:
            break

        # Select which reaction occurs
        r = np.random.random() * a_total

        if r < a1:
            # Reaction 1: Nucleation (A → B)
            N_A -= 1
            N_B += 1
        elif r < a1 + a2:
            # Reaction 2: Growth (A + B → 2B)
            N_A -= 1
            N_B += 1
        else:
            # Reaction 3: Etching (B → C)
            N_B -= 1

        # Ensure non-negative
        N_A = max(0, N_A)
        N_B = max(0, N_B)

        # Record (either all events or at intervals)
        if record_interval is None or (t - last_record_time) >= record_interval:
            times.append(t)
            N_A_history.append(N_A)
            N_B_history.append(N_B)
            last_record_time = t

        step += 1

    # Convert to concentrations
    times = np.array(times)
    A_conc = np.array(N_A_history) / (params.V * N_Av)
    B_conc = np.array(N_B_history) / (params.V * N_Av)

    # Final yield as fraction of initial precursor converted to graphene
    final_yield = B_conc[-1] / params.A0 if len(B_conc) > 0 else 0.0

    return SimulationResult(
        times=times,
        B_concentration=B_conc,
        A_concentration=A_conc,
        final_yield=final_yield,
        method="Gillespie SSA"
    )


def gillespie_ssa_fast(params: NGEParameters,
                       dt_record: float = 1.0) -> SimulationResult:
    """
    Optimized Gillespie SSA with fixed recording intervals.

    This version is faster for large systems by only recording at
    specified time intervals rather than every reaction event.

    Args:
        params: NGE model parameters
        dt_record: Time interval for recording concentrations

    Returns:
        SimulationResult
    """
    N_Av = 6.022e23
    N_A = params.N_A0
    N_B = params.N_B0
    t = 0.0

    # Pre-allocate recording arrays
    n_records = int(params.t_max / dt_record) + 1
    times = np.zeros(n_records)
    N_A_record = np.zeros(n_records)
    N_B_record = np.zeros(n_records)

    times[0] = t
    N_A_record[0] = N_A
    N_B_record[0] = N_B
    record_idx = 1
    next_record_time = dt_record

    k1 = params.k1
    k2_eff = params.k2 / (params.V * N_Av)
    k3 = params.k3

    while t < params.t_max:
        a1 = k1 * N_A
        a2 = k2_eff * N_A * N_B
        a3 = k3 * N_B
        a_total = a1 + a2 + a3

        if a_total <= 0:
            # Fill remaining records with current state
            while record_idx < n_records:
                times[record_idx] = next_record_time
                N_A_record[record_idx] = N_A
                N_B_record[record_idx] = N_B
                record_idx += 1
                next_record_time += dt_record
            break

        tau = np.random.exponential(1.0 / a_total)
        t += tau

        # Record if we've passed the next recording time
        while t >= next_record_time and record_idx < n_records:
            times[record_idx] = next_record_time
            N_A_record[record_idx] = N_A
            N_B_record[record_idx] = N_B
            record_idx += 1
            next_record_time += dt_record

        if t > params.t_max:
            break

        # Select and execute reaction
        r = np.random.random() * a_total
        if r < a1:
            N_A -= 1
            N_B += 1
        elif r < a1 + a2:
            N_A -= 1
            N_B += 1
        else:
            N_B -= 1

        N_A = max(0, N_A)
        N_B = max(0, N_B)

    # Trim to actual records
    times = times[:record_idx]
    A_conc = N_A_record[:record_idx] / (params.V * N_Av)
    B_conc = N_B_record[:record_idx] / (params.V * N_Av)

    final_yield = B_conc[-1] / params.A0 if len(B_conc) > 0 else 0.0

    return SimulationResult(
        times=times,
        B_concentration=B_conc,
        A_concentration=A_conc,
        final_yield=final_yield,
        method="Gillespie SSA (fast)"
    )
