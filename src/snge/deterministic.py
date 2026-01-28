"""
Deterministic NGE model solver.
"""

from typing import Tuple

import numpy as np
from scipy.integrate import odeint

from .models import NGEParameters


def nge_deterministic(y: np.ndarray, t: float, params: NGEParameters) -> np.ndarray:
    """
    Deterministic NGE ODEs.

    dy/dt for [A, B] system with mass conservation.
    """
    A, B = y
    k1, k2, k3 = params.k1, params.k2, params.k3

    dA_dt = -k1 * A - k2 * A * B  # Consumed by nucleation and growth
    dB_dt = k1 * A + k2 * A * B - k3 * B  # Created by nucleation/growth, destroyed by etching

    return np.array([dA_dt, dB_dt])


def solve_deterministic(params: NGEParameters, t_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve deterministic NGE model.

    Returns:
        A_conc, B_conc: Arrays of concentrations at each time point
    """
    y0 = [params.A0, params.B0]
    solution = odeint(nge_deterministic, y0, t_points, args=(params,))
    return solution[:, 0], solution[:, 1]
