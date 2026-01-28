"""
Data structures for the NGE model.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class NGEParameters:
    """Parameters for the NGE model"""
    k1: float  # Nucleation rate constant (s^-1)
    k2: float  # Growth rate constant (M^-1 s^-1)
    k3: float  # Etching rate constant (s^-1)
    A0: float  # Initial precursor concentration (M)
    B0: float  # Initial graphene concentration (M), usually 0
    V: float  # System volume (L) - affects stochastic fluctuations
    t_max: float  # Simulation end time (s)

    @property
    def N_A0(self) -> int:
        """Initial number of A molecules"""
        return int(self.A0 * self.V * 6.022e23)  # Avogadro's number

    @property
    def N_B0(self) -> int:
        """Initial number of B molecules"""
        return int(self.B0 * self.V * 6.022e23)


@dataclass
class SimulationResult:
    """Result from a single simulation run"""
    times: np.ndarray
    B_concentration: np.ndarray
    A_concentration: np.ndarray
    final_yield: float
    method: str
