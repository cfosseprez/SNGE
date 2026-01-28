"""
SNGE - Stochastic Nucleation-Growth-Etching Model
==================================================

This package implements stochastic simulations of the Nucleation-Growth-Etching (NGE)
model for graphene synthesis kinetics.

The NGE reaction scheme:
    A → B           (nucleation, rate k1)
    A + B → 2B      (growth, rate k2)
    B → C           (etching, rate k3)

where A = precursor, B = graphene, C = etched products

Two complementary simulation approaches are provided:
1. Gillespie Stochastic Simulation Algorithm (SSA) - exact discrete stochastic
2. Euler-Maruyama integration of Chemical Langevin Equation (CLE) - continuous approximation
"""

__version__ = "0.1.0"

# Data structures
from .models import NGEParameters, SimulationResult

# Deterministic solver
from .deterministic import nge_deterministic, solve_deterministic

# Stochastic methods (Gillespie SSA)
from .stochastic import gillespie_ssa, gillespie_ssa_fast

# Chemical Langevin Equation methods
from .langevin import euler_maruyama_cle, euler_maruyama_cle_simple

# Ensemble runners
from .ensemble import run_ensemble_gillespie, run_ensemble_euler_maruyama

# Fast implementations (Numba-accelerated)
from .stochastic_fast import gillespie_ssa_numba, run_ensemble_gillespie_numba
from .langevin_fast import (
    euler_maruyama_numba,
    run_ensemble_euler_maruyama_numba,
    run_ensemble_euler_maruyama_gpu,
)

# Analysis functions
from .analysis import compute_yield_statistics, compare_distributions, compute_cv_over_time

# Plotting functions
from .plotting import (
    plot_ensemble_trajectories,
    plot_cv_evolution,
    plot_distribution_comparison,
    plot_summary_figure,
)

# Fitting functions
from .fitting import (
    ExperimentalData,
    FitResult,
    fit_nge_to_mean,
    fit_fw_to_mean,
    compare_nge_vs_fw,
    load_data_from_csv,
    create_synthetic_data,
    plot_fit_results,
    export_parameters_for_simulation,
)

__all__ = [
    # Version
    "__version__",
    # Data structures
    "NGEParameters",
    "SimulationResult",
    # Deterministic
    "nge_deterministic",
    "solve_deterministic",
    # Stochastic (Gillespie)
    "gillespie_ssa",
    "gillespie_ssa_fast",
    # Langevin (CLE)
    "euler_maruyama_cle",
    "euler_maruyama_cle_simple",
    # Ensemble
    "run_ensemble_gillespie",
    "run_ensemble_euler_maruyama",
    # Fast/Accelerated versions (Numba)
    "gillespie_ssa_numba",
    "run_ensemble_gillespie_numba",
    "euler_maruyama_numba",
    "run_ensemble_euler_maruyama_numba",
    "run_ensemble_euler_maruyama_gpu",
    # Analysis
    "compute_yield_statistics",
    "compare_distributions",
    "compute_cv_over_time",
    # Plotting
    "plot_ensemble_trajectories",
    "plot_cv_evolution",
    "plot_distribution_comparison",
    "plot_summary_figure",
    # Fitting
    "ExperimentalData",
    "FitResult",
    "fit_nge_to_mean",
    "fit_fw_to_mean",
    "compare_nge_vs_fw",
    "load_data_from_csv",
    "create_synthetic_data",
    "plot_fit_results",
    "export_parameters_for_simulation",
]
