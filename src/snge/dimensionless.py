"""
Dimensionless deterministic solver for the SNGE model.

The dimensionless formulation transforms the NGE equations:
    d[A]/dt = -k₁[A] - k₂[A][B]
    d[B]/dt = k₁[A] + k₂[A][B] - k₃[B]

Into a single dimensionless ODE:
    db/dτ = α(1-b) + β(1-b)b - b

where:
    τ = k₃·t          (dimensionless time)
    b = [B]/[A]₀      (dimensionless yield)
    α = k₁/k₃         (nucleation-to-etching ratio)
    β = k₂[A]₀/k₃     (growth-to-etching ratio)

This formulation has several advantages:
1. Reduces number of parameters from 4 (k₁, k₂, k₃, A₀) to 2 (α, β)
2. Yield b is naturally bounded in [0, 1]
3. Enables comparison of systems with different scales
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Tuple, Optional

from .models import DimensionlessParameters, SNGEResult, NGEParameters


def snge_ode(b: float, tau: float, alpha: float, beta: float) -> float:
    """
    Right-hand side of the dimensionless SNGE ODE.

    db/dτ = α(1-b) + β(1-b)b - b
          = α - αb + βb - βb² - b
          = α + (β - α - 1)b - βb²

    Args:
        b: Dimensionless yield (0-1)
        tau: Dimensionless time (unused, for odeint compatibility)
        alpha: k₁/k₃ (nucleation-to-etching ratio)
        beta: k₂[A]₀/k₃ (growth-to-etching ratio)

    Returns:
        db/dτ: Rate of change of dimensionless yield
    """
    return alpha * (1 - b) + beta * (1 - b) * b - b


def snge_ode_ivp(tau: float, b: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Right-hand side for solve_ivp (different argument order).

    Args:
        tau: Dimensionless time
        b: Dimensionless yield as array
        alpha: k₁/k₃
        beta: k₂[A]₀/k₃

    Returns:
        db/dτ as array
    """
    return np.array([alpha * (1 - b[0]) + beta * (1 - b[0]) * b[0] - b[0]])


def solve_dimensionless(params: DimensionlessParameters,
                        n_points: int = 1000,
                        b0: float = 0.0,
                        method: str = 'odeint') -> SNGEResult:
    """
    Solve the dimensionless SNGE ODE.

    Args:
        params: DimensionlessParameters instance
        n_points: Number of time points in output
        b0: Initial dimensionless yield (default 0)
        method: 'odeint' or 'solve_ivp'

    Returns:
        SNGEResult with tau and b arrays
    """
    tau = np.linspace(0, params.tau_max, n_points)

    if method == 'odeint':
        b = odeint(snge_ode, b0, tau, args=(params.alpha, params.beta))
        b = b.flatten()
    elif method == 'solve_ivp':
        sol = solve_ivp(
            snge_ode_ivp,
            t_span=(0, params.tau_max),
            y0=[b0],
            args=(params.alpha, params.beta),
            t_eval=tau,
            method='RK45'
        )
        b = sol.y[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Enforce bounds (numerical errors can cause small violations)
    b = np.clip(b, 0.0, 1.0)

    return SNGEResult(
        tau=tau,
        b=b,
        final_yield=b[-1],
        method="Deterministic (dimensionless)"
    )


def compute_steady_state(alpha: float, beta: float) -> float:
    """
    Compute the analytical steady-state yield.

    At steady state: db/dτ = 0
        α(1-b) + β(1-b)b - b = 0
        α + (β - α - 1)b - βb² = 0
        βb² + (α + 1 - β)b - α = 0

    Using quadratic formula:
        b* = [-(α + 1 - β) + √((α + 1 - β)² + 4αβ)] / (2β)

    For β = 0 (no autocatalytic growth):
        b* = α / (α + 1)

    Args:
        alpha: k₁/k₃ (nucleation-to-etching ratio)
        beta: k₂[A]₀/k₃ (growth-to-etching ratio)

    Returns:
        b*: Steady-state dimensionless yield
    """
    if beta == 0:
        # Linear case: no autocatalytic growth
        return alpha / (alpha + 1)

    # Quadratic: βb² + (α + 1 - β)b - α = 0
    a_coef = beta
    b_coef = alpha + 1 - beta
    c_coef = -alpha

    discriminant = b_coef**2 - 4 * a_coef * c_coef

    if discriminant < 0:
        raise ValueError("No real steady state (discriminant < 0)")

    # Take the positive root (physical solution)
    b_star = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

    # Ensure result is in valid range
    return np.clip(b_star, 0.0, 1.0)


def compute_steady_state_from_nge(params: NGEParameters) -> float:
    """
    Compute steady-state yield from NGE parameters.

    Args:
        params: NGEParameters instance

    Returns:
        Steady-state yield as fraction [0, 1]
    """
    if params.k3 == 0:
        # No etching: yield goes to 1
        return 1.0

    alpha = params.k1 / params.k3
    beta = params.k2 * params.A0 / params.k3

    return compute_steady_state(alpha, beta)


def compute_time_to_threshold(params: DimensionlessParameters,
                              threshold: float = 0.1,
                              b0: float = 0.0) -> float:
    """
    Compute dimensionless time to reach a yield threshold.

    Useful for identifying the "critical period" where noise
    has the strongest effect.

    Args:
        params: DimensionlessParameters instance
        threshold: Yield threshold (default 0.1 = 10%)
        b0: Initial yield

    Returns:
        τ_threshold: Dimensionless time to reach threshold
                     Returns tau_max if threshold not reached
    """
    result = solve_dimensionless(params, n_points=1000, b0=b0)

    # Find first crossing of threshold
    crossing_idx = np.where(result.b >= threshold)[0]

    if len(crossing_idx) == 0:
        return params.tau_max

    return result.tau[crossing_idx[0]]


def compute_growth_rate_at_b(alpha: float, beta: float, b: float) -> float:
    """
    Compute the instantaneous growth rate db/dτ at a given yield.

    Useful for understanding when growth is fastest.

    Args:
        alpha: k₁/k₃
        beta: k₂[A]₀/k₃
        b: Current dimensionless yield

    Returns:
        db/dτ at the given b
    """
    return snge_ode(b, 0.0, alpha, beta)


def compute_max_growth_rate(alpha: float, beta: float) -> Tuple[float, float]:
    """
    Find the yield value where growth rate is maximized.

    d²b/dτ² = 0 gives:
        (β - α - 1) - 2βb = 0
        b_max = (β - α - 1) / (2β)

    Args:
        alpha: k₁/k₃
        beta: k₂[A]₀/k₃

    Returns:
        (b_max, rate_max): Yield at max growth and the max growth rate
    """
    if beta == 0:
        # Linear case: rate is monotonically decreasing
        return 0.0, alpha

    b_max = (beta - alpha - 1) / (2 * beta)

    # Ensure b_max is in valid range
    b_max = np.clip(b_max, 0.0, 1.0)

    rate_max = snge_ode(b_max, 0.0, alpha, beta)

    return b_max, rate_max
