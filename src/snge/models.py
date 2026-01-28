"""
Data structures for the NGE model.
"""

from dataclasses import dataclass
import warnings
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


@dataclass
class DimensionlessParameters:
    """
    Dimensionless parameters for NGE kinetics.

    The dimensionless formulation uses:
        τ = k₃·t         (dimensionless time)
        b = [B]/[A]₀     (dimensionless yield, 0-1)

    The dimensionless ODE is:
        db/dτ = α(1-b) + β(1-b)b - b

    where:
        α = k₁/k₃       (nucleation-to-etching ratio)
        β = k₂[A]₀/k₃   (growth-to-etching ratio)

    IMPORTANT: CV is NOT a parameter - it EMERGES from Gillespie simulations.
    The paper's approach: fit k₁, k₂, k₃ to mean kinetics only, then run
    Gillespie SSA with physical N = [A]₀ × V × Nₐ. The CV emerges naturally
    from the stochastic dynamics - no parameters are fitted to variance data.

    For plasma graphene synthesis: α ~ 0.01-0.1, β ~ 0.5-2 (competitive regime)
    """
    alpha: float       # k₁/k₃ - nucleation-to-etching ratio
    beta: float        # k₂[A]₀/k₃ - growth-to-etching ratio
    tau_max: float     # Dimensionless time limit

    @classmethod
    def from_nge_parameters(cls, params: 'NGEParameters') -> 'DimensionlessParameters':
        """
        Convert NGEParameters to dimensionless parameters.

        Args:
            params: NGEParameters instance

        Returns:
            DimensionlessParameters instance

        Note:
            CV emerges from Gillespie simulations - no noise parameters needed.
        """
        if params.k3 == 0:
            raise ValueError("k3 must be non-zero for dimensionless formulation")

        alpha = params.k1 / params.k3
        beta = params.k2 * params.A0 / params.k3
        tau_max = params.k3 * params.t_max

        return cls(
            alpha=alpha,
            beta=beta,
            tau_max=tau_max
        )


@dataclass
class DimensionlessParametersPhenomenological:
    """
    DEPRECATED: Dimensionless parameters with phenomenological noise model.

    WARNING: This is NOT the paper's methodology. The phenomenological noise
    model σ(b) = σ₀/(b+ε) fits noise parameters to variance data, which
    contradicts the paper's approach where CV emerges naturally from
    Gillespie simulations without fitting to variance.

    Use DimensionlessParameters with Gillespie SSA instead.

    This class is retained only for backwards compatibility and comparison.
    """
    alpha: float       # k₁/k₃ - nucleation-to-etching ratio
    beta: float        # k₂[A]₀/k₃ - growth-to-etching ratio
    sigma0: float      # Perturbation intensity (DEPRECATED)
    epsilon: float     # Regularization to prevent σ→∞ (DEPRECATED)
    tau_max: float     # Dimensionless time limit
    noise_model: str   # "inverse" or "sqrt" (DEPRECATED)

    def __post_init__(self):
        warnings.warn(
            "DimensionlessParametersPhenomenological is DEPRECATED. "
            "This phenomenological noise model is NOT the paper's methodology. "
            "Use DimensionlessParameters with Gillespie SSA instead, where CV "
            "emerges naturally from physical parameters.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.noise_model not in ("inverse", "sqrt"):
            raise ValueError("noise_model must be 'inverse' or 'sqrt'")

    @classmethod
    def from_nge_parameters(cls, params: 'NGEParameters', sigma0: float = 0.01,
                            epsilon: float = 0.001, noise_model: str = "inverse"):
        """
        Convert NGEParameters to dimensionless parameters with phenomenological noise.

        DEPRECATED: Use DimensionlessParameters.from_nge_parameters() instead.
        """
        if params.k3 == 0:
            raise ValueError("k3 must be non-zero for dimensionless formulation")

        alpha = params.k1 / params.k3
        beta = params.k2 * params.A0 / params.k3
        tau_max = params.k3 * params.t_max

        return cls(
            alpha=alpha,
            beta=beta,
            sigma0=sigma0,
            epsilon=epsilon,
            tau_max=tau_max,
            noise_model=noise_model
        )


@dataclass
class SNGEResult:
    """
    Result from a dimensionless simulation.

    Uses the dimensionless formulation:
        τ = k₃·t      (dimensionless time)
        b = [B]/[A]₀  (dimensionless yield)

    Note: This result type is used by both Gillespie SSA (paper's method)
    and the deprecated phenomenological SNGE method.
    """
    tau: np.ndarray       # Dimensionless time array
    b: np.ndarray         # Dimensionless yield trajectory (0-1)
    final_yield: float    # Final b value
    method: str           # Simulation method name

    def to_dimensional(self, params: 'NGEParameters') -> SimulationResult:
        """
        Convert dimensionless result back to dimensional quantities.

        Args:
            params: NGEParameters for scaling

        Returns:
            SimulationResult with dimensional times and concentrations
        """
        times = self.tau / params.k3
        B_concentration = self.b * params.A0
        A_concentration = params.A0 * (1 - self.b)  # Mass conservation (no etching loss)

        return SimulationResult(
            times=times,
            B_concentration=B_concentration,
            A_concentration=A_concentration,
            final_yield=self.final_yield,
            method=self.method
        )
