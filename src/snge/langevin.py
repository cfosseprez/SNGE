"""
Euler-Maruyama integration of the Chemical Langevin Equation (CLE) for the NGE model.
"""

import numpy as np

from .models import NGEParameters, SimulationResult


def euler_maruyama_cle(params: NGEParameters,
                       dt: float = 0.1,
                       seed: int = None) -> SimulationResult:
    """
    Euler-Maruyama integration of the Chemical Langevin Equation for NGE.

    The CLE for the NGE system is:
        d[B] = f([A],[B])dt + σ([A],[B])dW

    where:
        f([A],[B]) = k1[A] + k2[A][B] - k3[B]   (drift)
        σ([A],[B]) = (1/√V) * √(k1[A] + k2[A][B] + k3[B])   (diffusion)

    The Euler-Maruyama scheme:
        B(t+dt) = B(t) + f*dt + σ*√dt*ξ

    where ξ ~ N(0,1)

    Args:
        params: NGE model parameters
        dt: Time step for integration
        seed: Random seed (optional)

    Returns:
        SimulationResult
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    n_steps = int(params.t_max / dt) + 1
    times = np.linspace(0, params.t_max, n_steps)
    A = np.zeros(n_steps)
    B = np.zeros(n_steps)

    A[0] = params.A0
    B[0] = params.B0

    k1, k2, k3 = params.k1, params.k2, params.k3
    V = params.V
    N_Av = 6.022e23
    sqrt_dt = np.sqrt(dt)
    # Correct CLE noise scaling: σ = √(rate / (V * N_Av))
    inv_sqrt_V_NAv = 1.0 / np.sqrt(V * N_Av)

    for i in range(n_steps - 1):
        A_curr = A[i]
        B_curr = B[i]

        # Drift term (deterministic NGE)
        drift_A = -k1 * A_curr - k2 * A_curr * B_curr
        drift_B = k1 * A_curr + k2 * A_curr * B_curr - k3 * B_curr

        # Diffusion terms (noise intensities)
        # Each reaction contributes √(rate) to the noise
        rate1 = k1 * A_curr  # nucleation rate
        rate2 = k2 * A_curr * B_curr  # growth rate
        rate3 = k3 * B_curr  # etching rate

        # Total noise for B: contributions from all three reactions
        # dB = ... + √(rate1)*dW1 + √(rate2)*dW2 - √(rate3)*dW3
        # For variance: σ²_B = (rate1 + rate2 + rate3) / V

        # Generate independent noise for each reaction channel
        xi1 = np.random.normal()
        xi2 = np.random.normal()
        xi3 = np.random.normal()

        # Noise contributions (properly scaled)
        noise_A = inv_sqrt_V_NAv * (-np.sqrt(max(0, rate1)) * xi1 - np.sqrt(max(0, rate2)) * xi2)
        noise_B = inv_sqrt_V_NAv * (
                    np.sqrt(max(0, rate1)) * xi1 + np.sqrt(max(0, rate2)) * xi2 - np.sqrt(max(0, rate3)) * xi3)

        # Euler-Maruyama update
        A[i + 1] = A_curr + drift_A * dt + noise_A * sqrt_dt
        B[i + 1] = B_curr + drift_B * dt + noise_B * sqrt_dt

        # Enforce physical bounds: 0 <= A, B and A + B <= A0
        A[i + 1] = max(0, min(params.A0, A[i + 1]))
        B[i + 1] = max(0, min(params.A0, B[i + 1]))

        # Enforce mass conservation: A + B <= A0
        total = A[i + 1] + B[i + 1]
        if total > params.A0:
            scale = params.A0 / total
            A[i + 1] *= scale
            B[i + 1] *= scale

    final_yield = B[-1] / params.A0

    return SimulationResult(
        times=times,
        B_concentration=B,
        A_concentration=A,
        final_yield=final_yield,
        method="Euler-Maruyama CLE"
    )


def euler_maruyama_cle_simple(params: NGEParameters,
                              dt: float = 0.1,
                              seed: int = None) -> SimulationResult:
    """
    Simplified Euler-Maruyama with combined noise term.

    Uses the combined diffusion coefficient:
        σ = (1/√V) * √(k1[A] + k2[A][B] + k3[B])

    This is mathematically equivalent for computing distributions
    but simpler to implement.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(params.t_max / dt) + 1
    times = np.linspace(0, params.t_max, n_steps)
    A = np.zeros(n_steps)
    B = np.zeros(n_steps)

    A[0] = params.A0
    B[0] = params.B0

    k1, k2, k3 = params.k1, params.k2, params.k3
    sqrt_dt = np.sqrt(dt)
    N_Av = 6.022e23
    # Correct CLE noise scaling: σ = √(rate / (V * N_Av))
    inv_sqrt_V_NAv = 1.0 / np.sqrt(params.V * N_Av)

    for i in range(n_steps - 1):
        A_curr = A[i]
        B_curr = B[i]

        # Drift
        drift_B = k1 * A_curr + k2 * A_curr * B_curr - k3 * B_curr

        # Combined diffusion coefficient for B
        total_rate = k1 * A_curr + k2 * A_curr * B_curr + k3 * B_curr
        sigma_B = inv_sqrt_V_NAv * np.sqrt(max(0, total_rate))

        # Update B
        xi = np.random.normal()
        B[i + 1] = B_curr + drift_B * dt + sigma_B * sqrt_dt * xi
        # Enforce physical bounds: 0 <= B <= A0 (mass conservation)
        B[i + 1] = max(0, min(params.A0, B[i + 1]))

        # Update A by mass conservation
        A[i + 1] = params.A0 - B[i + 1]

    final_yield = B[-1] / params.A0

    return SimulationResult(
        times=times,
        B_concentration=B,
        A_concentration=A,
        final_yield=final_yield,
        method="Euler-Maruyama CLE (simple)"
    )
