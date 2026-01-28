"""
Euler-Maruyama integration of the Chemical Langevin Equation (CLE) for the NGE model.

This module provides:
1. CLE (Chemical Langevin Equation): Noise scales with √(rate/V), volume-dependent
   - This is a valid approximation for large molecule counts

2. DEPRECATED: Phenomenological SNGE with state-dependent noise σ(b) = σ₀/(b+ε)
   - WARNING: This is NOT the paper's methodology
   - The paper uses Gillespie SSA where CV emerges naturally from physics

For the paper's methodology, use:
    from snge.stochastic_fast import run_ensemble_gillespie_numba
    from snge.analysis import predict_cv_from_gillespie
"""

import warnings
import numpy as np

from .models import NGEParameters, SimulationResult, DimensionlessParameters, SNGEResult


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


# =============================================================================
# DEPRECATED: Phenomenological State-Dependent Noise
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

def euler_maruyama_phenomenological(params, dtau: float = 0.001,
                                     b0: float = 0.0,
                                     seed: int = None) -> SNGEResult:
    """
    DEPRECATED: Euler-Maruyama with phenomenological state-dependent noise.

    WARNING: This is NOT the paper's methodology. The phenomenological noise
    model σ(b) = σ₀/(b+ε) fits noise parameters to variance, contradicting
    the paper's approach where CV emerges naturally from Gillespie SSA.

    Use run_ensemble_gillespie_numba() instead.

    The SDE in dimensionless form:
        db = [α(1-b) + β(1-b)b - b]dτ + σ(b)dW

    where the noise intensity σ(b) depends on yield:
        inverse: σ(b) = σ₀ / (b + ε)    - amplified when b is small
        sqrt:    σ(b) = σ₀ · √b · (1 + κ/b)  - alternative model

    Args:
        params: DimensionlessParametersPhenomenological instance
        dtau: Dimensionless time step
        b0: Initial dimensionless yield (default 0)
        seed: Random seed for reproducibility

    Returns:
        SNGEResult with tau and b arrays
    """
    warnings.warn(
        "euler_maruyama_phenomenological (formerly snge_euler_maruyama) is DEPRECATED. "
        "This phenomenological noise model is NOT the paper's methodology. "
        "Use run_ensemble_gillespie_numba() instead, where CV emerges naturally.",
        DeprecationWarning,
        stacklevel=2
    )

    # Check for required phenomenological parameters
    if not hasattr(params, 'sigma0') or not hasattr(params, 'epsilon'):
        raise ValueError(
            "euler_maruyama_phenomenological requires DimensionlessParametersPhenomenological "
            "with sigma0, epsilon, and noise_model attributes."
        )
    if seed is not None:
        np.random.seed(seed)

    # Initialize
    n_steps = int(params.tau_max / dtau) + 1
    tau = np.linspace(0, params.tau_max, n_steps)
    b = np.zeros(n_steps)
    b[0] = b0

    alpha = params.alpha
    beta = params.beta
    sigma0 = params.sigma0
    epsilon = params.epsilon
    sqrt_dtau = np.sqrt(dtau)

    for i in range(n_steps - 1):
        b_curr = b[i]

        # Drift term: deterministic SNGE
        drift = alpha * (1 - b_curr) + beta * (1 - b_curr) * b_curr - b_curr

        # State-dependent noise (KEY DIFFERENCE from CLE)
        if params.noise_model == "inverse":
            # Inverse model: σ(b) = σ₀ / (b + ε)
            # Noise is amplified when b is small (critical period)
            sigma = sigma0 / (b_curr + epsilon)
        else:  # sqrt model
            # Sqrt model: σ(b) = σ₀ · √(b) · (1 + κ/b) where κ = epsilon
            # This gives σ ∝ 1/√b at small b
            sigma = sigma0 * np.sqrt(max(b_curr, 1e-10)) * (1 + epsilon / (b_curr + 1e-10))

        # Random increment
        xi = np.random.normal()

        # Euler-Maruyama update
        b_new = b_curr + drift * dtau + sigma * sqrt_dtau * xi

        # Enforce physical bounds: 0 ≤ b ≤ 1
        b[i + 1] = max(0.0, min(1.0, b_new))

    return SNGEResult(
        tau=tau,
        b=b,
        final_yield=b[-1],
        method="Phenomenological Euler-Maruyama (DEPRECATED)"
    )


def euler_maruyama_phenomenological_ensemble(params,
                                              n_runs: int = 100,
                                              dtau: float = 0.001,
                                              b0: float = 0.0,
                                              seed: int = None) -> list:
    """
    DEPRECATED: Run an ensemble of phenomenological SNGE simulations.

    WARNING: This is NOT the paper's methodology. Use
    run_ensemble_gillespie_numba() instead.

    Args:
        params: DimensionlessParametersPhenomenological instance
        n_runs: Number of independent simulations
        dtau: Dimensionless time step
        b0: Initial dimensionless yield
        seed: Base random seed (each run uses seed+i)

    Returns:
        List of SNGEResult objects
    """
    warnings.warn(
        "euler_maruyama_phenomenological_ensemble (formerly snge_euler_maruyama_ensemble) "
        "is DEPRECATED. This phenomenological noise model is NOT the paper's methodology. "
        "Use run_ensemble_gillespie_numba() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    results = []

    for i in range(n_runs):
        run_seed = (seed + i) if seed is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = euler_maruyama_phenomenological(params, dtau=dtau, b0=b0, seed=run_seed)
        results.append(result)

    return results


# Backwards compatibility aliases (deprecated)
def snge_euler_maruyama(params, dtau: float = 0.001, b0: float = 0.0, seed: int = None) -> SNGEResult:
    """DEPRECATED: Use euler_maruyama_phenomenological instead."""
    return euler_maruyama_phenomenological(params, dtau=dtau, b0=b0, seed=seed)


def snge_euler_maruyama_ensemble(params, n_runs: int = 100, dtau: float = 0.001,
                                  b0: float = 0.0, seed: int = None) -> list:
    """DEPRECATED: Use euler_maruyama_phenomenological_ensemble instead."""
    return euler_maruyama_phenomenological_ensemble(params, n_runs=n_runs, dtau=dtau, b0=b0, seed=seed)
