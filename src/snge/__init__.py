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

THE PAPER'S METHODOLOGY:
========================
1. Fit k₁, k₂, k₃ to MEAN kinetics only (no variance fitting)
2. Run Gillespie SSA with physical N = [A]₀ × V × Nₐ
3. CV emerges naturally from Poisson statistics - no fitting to variance data

"The CV is not imposed or fitted—it emerges from running stochastic
simulations with physical parameters. The model either predicts the
experimental CV or it doesn't."

Simulation approaches:
1. Gillespie SSA (exact discrete stochastic) - RECOMMENDED for CV prediction
2. Chemical Langevin Equation (CLE) - continuous approximation

DEPRECATED: Phenomenological SNGE with σ(b) = σ₀/(b+ε) noise model
This approach fits noise parameters to variance, contradicting the paper's
methodology. It is retained only for backwards compatibility.
"""

__version__ = "0.3.0"

# Data structures
from .models import (
    NGEParameters,
    SimulationResult,
    DimensionlessParameters,
    SNGEResult,
    # Deprecated - for backwards compatibility only
    DimensionlessParametersPhenomenological,
)

# Deterministic solver
from .deterministic import nge_deterministic, solve_deterministic

# Dimensionless deterministic solver
from .dimensionless import (
    snge_ode,
    solve_dimensionless,
    compute_steady_state,
    compute_steady_state_from_nge,
    compute_time_to_threshold,
    compute_max_growth_rate,
)

# Stochastic methods (Gillespie SSA) - RECOMMENDED for CV prediction
from .stochastic import gillespie_ssa, gillespie_ssa_fast

# Chemical Langevin Equation methods
from .langevin import (
    euler_maruyama_cle,
    euler_maruyama_cle_simple,
    # Deprecated - phenomenological noise (NOT paper's methodology)
    euler_maruyama_phenomenological,
    euler_maruyama_phenomenological_ensemble,
    # Legacy aliases (deprecated)
    snge_euler_maruyama,
    snge_euler_maruyama_ensemble,
)

# Ensemble runners
from .ensemble import run_ensemble_gillespie, run_ensemble_euler_maruyama

# Fast implementations (Numba-accelerated)
from .stochastic_fast import gillespie_ssa_numba, run_ensemble_gillespie_numba
from .langevin_fast import (
    euler_maruyama_numba,
    run_ensemble_euler_maruyama_numba,
    run_ensemble_euler_maruyama_gpu,
    # Deprecated - phenomenological noise (NOT paper's methodology)
    euler_maruyama_phenomenological_numba,
    run_ensemble_phenomenological_numba,
    # Legacy aliases (deprecated)
    snge_euler_maruyama_numba,
    run_ensemble_snge_numba,
)

# Analysis functions
from .analysis import (
    # Core analysis (works with any results)
    compute_yield_statistics,
    compare_distributions,
    compute_cv_over_time,
    # CV Prediction from Gillespie (PAPER'S METHODOLOGY)
    predict_cv_from_gillespie,
    validate_cv_prediction,
    compare_seeded_vs_unseeded,
    compute_critical_period_gillespie,
    analyze_trajectory_divergence,
    # Dimensionless parameter utilities
    compute_dimensionless_parameters,
    # Deprecated - phenomenological SNGE analysis
    compute_critical_period,
    analyze_noise_sensitivity,
    compute_snge_yield_statistics,
    compute_snge_cv_over_time,
    compare_cle_vs_snge,
)

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
    # Deprecated - fits σ₀ to variance (NOT paper's methodology)
    fit_sigma0_to_cv,
    fit_sigma0_from_nge_parameters,
)

__all__ = [
    # Version
    "__version__",
    # Data structures
    "NGEParameters",
    "SimulationResult",
    "DimensionlessParameters",
    "SNGEResult",
    "DimensionlessParametersPhenomenological",  # Deprecated
    # Deterministic
    "nge_deterministic",
    "solve_deterministic",
    # Dimensionless deterministic
    "snge_ode",
    "solve_dimensionless",
    "compute_steady_state",
    "compute_steady_state_from_nge",
    "compute_time_to_threshold",
    "compute_max_growth_rate",
    # Stochastic (Gillespie) - RECOMMENDED
    "gillespie_ssa",
    "gillespie_ssa_fast",
    # Langevin (CLE)
    "euler_maruyama_cle",
    "euler_maruyama_cle_simple",
    # Phenomenological (DEPRECATED)
    "euler_maruyama_phenomenological",
    "euler_maruyama_phenomenological_ensemble",
    "snge_euler_maruyama",  # Legacy alias
    "snge_euler_maruyama_ensemble",  # Legacy alias
    # Ensemble
    "run_ensemble_gillespie",
    "run_ensemble_euler_maruyama",
    # Fast/Accelerated versions (Numba)
    "gillespie_ssa_numba",
    "run_ensemble_gillespie_numba",  # RECOMMENDED for CV prediction
    "euler_maruyama_numba",
    "run_ensemble_euler_maruyama_numba",
    "run_ensemble_euler_maruyama_gpu",
    # Phenomenological (DEPRECATED)
    "euler_maruyama_phenomenological_numba",
    "run_ensemble_phenomenological_numba",
    "snge_euler_maruyama_numba",  # Legacy alias
    "run_ensemble_snge_numba",  # Legacy alias
    # Analysis - Core
    "compute_yield_statistics",
    "compare_distributions",
    "compute_cv_over_time",
    # Analysis - CV Prediction (PAPER'S METHODOLOGY)
    "predict_cv_from_gillespie",
    "validate_cv_prediction",
    "compare_seeded_vs_unseeded",
    "compute_critical_period_gillespie",
    "analyze_trajectory_divergence",
    # Analysis - Dimensionless
    "compute_dimensionless_parameters",
    # Analysis - Phenomenological (DEPRECATED)
    "compute_critical_period",
    "analyze_noise_sensitivity",
    "compute_snge_yield_statistics",
    "compute_snge_cv_over_time",
    "compare_cle_vs_snge",
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
    # Fitting - σ₀ (DEPRECATED)
    "fit_sigma0_to_cv",
    "fit_sigma0_from_nge_parameters",
]
