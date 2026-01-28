"""
Analysis functions for NGE simulation results.

The paper's approach: CV emerges naturally from Gillespie SSA simulations
with physical parameters. No parameters are fitted to variance data.
"""

from typing import List, Tuple, Union, Optional
import warnings

import numpy as np
from scipy import stats

from .models import SimulationResult, DimensionlessParameters, SNGEResult, NGEParameters


def compute_yield_statistics(results: List[SimulationResult]) -> dict:
    """
    Compute statistics of final yields from ensemble simulations.

    Returns dictionary with:
        - mean, std, cv (coefficient of variation)
        - median, iqr
        - skewness, kurtosis
        - min, max, range
    """
    yields = np.array([r.final_yield for r in results])

    return {
        'yields': yields,
        'n': len(yields),
        'mean': np.mean(yields),
        'std': np.std(yields, ddof=1),
        'cv': np.std(yields, ddof=1) / np.mean(yields) * 100,  # CV in percent
        'median': np.median(yields),
        'iqr': np.percentile(yields, 75) - np.percentile(yields, 25),
        'skewness': stats.skew(yields),
        'kurtosis': stats.kurtosis(yields),
        'min': np.min(yields),
        'max': np.max(yields),
        'range': np.max(yields) - np.min(yields)
    }


def compare_distributions(yields1: np.ndarray, yields2: np.ndarray,
                          name1: str = "Distribution 1",
                          name2: str = "Distribution 2") -> dict:
    """
    Compare two yield distributions statistically.

    Returns:
        Dictionary with KS test, t-test, and other comparison metrics
    """
    # Kolmogorov-Smirnov test (are they from the same distribution?)
    ks_stat, ks_pvalue = stats.ks_2samp(yields1, yields2)

    # Two-sample t-test (are means different?)
    t_stat, t_pvalue = stats.ttest_ind(yields1, yields2)

    # Levene's test (are variances different?)
    levene_stat, levene_pvalue = stats.levene(yields1, yields2)

    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ks_same_distribution': ks_pvalue > 0.05,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'means_equal': t_pvalue > 0.05,
        'levene_statistic': levene_stat,
        'levene_pvalue': levene_pvalue,
        'variances_equal': levene_pvalue > 0.05,
        f'{name1}_mean': np.mean(yields1),
        f'{name2}_mean': np.mean(yields2),
        f'{name1}_cv': np.std(yields1) / np.mean(yields1) * 100,
        f'{name2}_cv': np.std(yields2) / np.mean(yields2) * 100,
    }


def compute_cv_over_time(results: List[SimulationResult],
                         time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coefficient of variation as a function of time.

    This shows when variability is generated during synthesis.

    Args:
        results: List of simulation results
        time_points: Time points at which to evaluate CV

    Returns:
        time_points, cv_values (in percent)
    """
    n_times = len(time_points)
    n_runs = len(results)

    # Interpolate all trajectories to common time points
    B_matrix = np.zeros((n_runs, n_times))

    for i, res in enumerate(results):
        B_matrix[i, :] = np.interp(time_points, res.times, res.B_concentration)

    # Compute CV at each time point
    means = np.mean(B_matrix, axis=0)
    stds = np.std(B_matrix, axis=0, ddof=1)

    # Avoid division by zero at early times
    cv = np.zeros(n_times)
    nonzero = means > 1e-10
    cv[nonzero] = stds[nonzero] / means[nonzero] * 100

    return time_points, cv


# =============================================================================
# SNGE-Specific Analysis Functions
# =============================================================================

def compute_dimensionless_parameters(params: NGEParameters) -> dict:
    """
    Convert NGE parameters to dimensionless form.

    The dimensionless parameters are:
        α = k₁/k₃       (nucleation-to-etching ratio)
        β = k₂[A]₀/k₃   (growth-to-etching ratio)
        τ_max = k₃·t_max (dimensionless time limit)

    Args:
        params: NGEParameters instance

    Returns:
        Dictionary with alpha, beta, tau_max, and interpretation
    """
    if params.k3 == 0:
        raise ValueError("k3 must be non-zero for dimensionless analysis")

    alpha = params.k1 / params.k3
    beta = params.k2 * params.A0 / params.k3
    tau_max = params.k3 * params.t_max

    # Compute steady state (from dimensionless module)
    if beta == 0:
        b_star = alpha / (alpha + 1)
    else:
        a_coef = beta
        b_coef = alpha + 1 - beta
        c_coef = -alpha
        discriminant = b_coef**2 - 4 * a_coef * c_coef
        b_star = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
        b_star = np.clip(b_star, 0.0, 1.0)

    return {
        'alpha': alpha,
        'beta': beta,
        'tau_max': tau_max,
        'b_steady_state': b_star,
        'interpretation': {
            'nucleation_vs_etching': 'nucleation dominates' if alpha > 1 else 'etching dominates',
            'growth_vs_etching': 'growth dominates' if beta > 1 else 'etching dominates',
            'expected_yield': f'{b_star * 100:.1f}%'
        }
    }


def compute_critical_period(results: List[SNGEResult],
                            threshold: float = 0.1) -> dict:
    """
    DEPRECATED: Identify the "critical period" from phenomenological SNGE results.

    WARNING: This function uses the phenomenological noise model σ(b) = σ₀/(b+ε)
    which is NOT the paper's methodology. Use compute_critical_period_gillespie()
    with Gillespie SSA results instead.

    Args:
        results: List of SNGEResult objects
        threshold: Yield threshold defining end of critical period

    Returns:
        Dictionary with critical period statistics
    """
    warnings.warn(
        "compute_critical_period for SNGEResult is DEPRECATED. "
        "Use compute_critical_period_gillespie() with Gillespie SSA results instead.",
        DeprecationWarning,
        stacklevel=2
    )
    n_runs = len(results)
    tau_critical = np.zeros(n_runs)

    for i, result in enumerate(results):
        # Find first crossing of threshold
        crossing_idx = np.where(result.b >= threshold)[0]
        if len(crossing_idx) > 0:
            tau_critical[i] = result.tau[crossing_idx[0]]
        else:
            tau_critical[i] = result.tau[-1]  # Never reached threshold

    return {
        'threshold': threshold,
        'mean_tau_critical': np.mean(tau_critical),
        'std_tau_critical': np.std(tau_critical, ddof=1),
        'cv_tau_critical': np.std(tau_critical, ddof=1) / np.mean(tau_critical) * 100,
        'min_tau_critical': np.min(tau_critical),
        'max_tau_critical': np.max(tau_critical),
        'tau_critical_values': tau_critical
    }


def analyze_noise_sensitivity(params, b_range: np.ndarray = None) -> dict:
    """
    DEPRECATED: Analyze phenomenological noise intensity σ(b).

    WARNING: This function analyzes the phenomenological noise model
    σ(b) = σ₀/(b+ε) which is NOT the paper's methodology. In the paper's
    approach, CV emerges naturally from Gillespie simulations without
    any σ(b) function.

    Args:
        params: DimensionlessParametersPhenomenological instance
        b_range: Array of b values to analyze (default: 0.001 to 0.99)

    Returns:
        Dictionary with b values, sigma values, and analysis
    """
    warnings.warn(
        "analyze_noise_sensitivity is DEPRECATED. The phenomenological noise "
        "model σ(b) = σ₀/(b+ε) is NOT the paper's methodology. "
        "Use predict_cv_from_gillespie() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Check if params has phenomenological attributes
    if not hasattr(params, 'sigma0') or not hasattr(params, 'epsilon'):
        raise ValueError(
            "analyze_noise_sensitivity requires DimensionlessParametersPhenomenological "
            "with sigma0 and epsilon attributes. This function is DEPRECATED."
        )
    if b_range is None:
        b_range = np.linspace(0.001, 0.99, 100)

    sigma0 = params.sigma0
    epsilon = params.epsilon

    if params.noise_model == "inverse":
        sigma = sigma0 / (b_range + epsilon)
        model_name = f"σ(b) = {sigma0:.3f} / (b + {epsilon:.4f})"
    else:  # sqrt
        sigma = sigma0 * np.sqrt(b_range) * (1 + epsilon / b_range)
        model_name = f"σ(b) = {sigma0:.3f} · √b · (1 + {epsilon:.4f}/b)"

    # Find b where sigma drops to various fractions of max
    sigma_max = sigma[0]
    half_max_idx = np.argmin(np.abs(sigma - sigma_max / 2))
    tenth_max_idx = np.argmin(np.abs(sigma - sigma_max / 10))

    return {
        'b': b_range,
        'sigma': sigma,
        'model': params.noise_model,
        'model_equation': model_name,
        'sigma_max': sigma_max,
        'sigma_at_b_half': sigma[len(sigma) // 2],
        'b_at_half_sigma': b_range[half_max_idx],
        'b_at_tenth_sigma': b_range[tenth_max_idx],
        'amplification_ratio': sigma_max / sigma[-1]
    }


def compute_snge_yield_statistics(results: List[SNGEResult]) -> dict:
    """
    Compute statistics of final yields from SNGEResult objects.

    Note: This works for both Gillespie SSA results (converted to SNGEResult)
    and the deprecated phenomenological SNGE results.

    Returns dictionary with:
        - mean, std, cv (coefficient of variation)
        - median, iqr
        - skewness, kurtosis
        - min, max, range
    """
    yields = np.array([r.final_yield for r in results])

    return {
        'yields': yields,
        'n': len(yields),
        'mean': np.mean(yields),
        'std': np.std(yields, ddof=1),
        'cv': np.std(yields, ddof=1) / np.mean(yields) * 100,  # CV in percent
        'median': np.median(yields),
        'iqr': np.percentile(yields, 75) - np.percentile(yields, 25),
        'skewness': stats.skew(yields),
        'kurtosis': stats.kurtosis(yields),
        'min': np.min(yields),
        'max': np.max(yields),
        'range': np.max(yields) - np.min(yields)
    }


def compute_snge_cv_over_time(results: List[SNGEResult],
                              tau_points: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coefficient of variation as a function of dimensionless time.

    This shows when variability is generated during synthesis,
    particularly highlighting the critical period.

    Args:
        results: List of SNGEResult objects
        tau_points: Dimensionless time points at which to evaluate CV

    Returns:
        tau_points, cv_values (in percent)
    """
    if tau_points is None:
        tau_points = results[0].tau.copy()

    n_times = len(tau_points)
    n_runs = len(results)

    # Interpolate all trajectories to common time points
    b_matrix = np.zeros((n_runs, n_times))

    for i, res in enumerate(results):
        b_matrix[i, :] = np.interp(tau_points, res.tau, res.b)

    # Compute CV at each time point
    means = np.mean(b_matrix, axis=0)
    stds = np.std(b_matrix, axis=0, ddof=1)

    # Avoid division by zero at early times
    cv = np.zeros(n_times)
    nonzero = means > 1e-10
    cv[nonzero] = stds[nonzero] / means[nonzero] * 100

    return tau_points, cv


def compare_cle_vs_snge(cle_results: List[SimulationResult],
                        snge_results: List[SNGEResult],
                        params: NGEParameters = None) -> dict:
    """
    DEPRECATED: Compare CLE and phenomenological SNGE yield distributions.

    Note: The phenomenological SNGE model is NOT the paper's methodology.
    Use Gillespie SSA for CV predictions instead.

    Key comparison: CLE variance scales with 1/V, SNGE variance does not.

    Args:
        cle_results: List of SimulationResult from CLE simulations
        snge_results: List of SNGEResult from SNGE simulations
        params: Optional NGEParameters for additional context

    Returns:
        Dictionary with comparison statistics
    """
    warnings.warn(
        "compare_cle_vs_snge is DEPRECATED. The phenomenological SNGE model "
        "is NOT the paper's methodology. Use predict_cv_from_gillespie() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    cle_yields = np.array([r.final_yield for r in cle_results])
    snge_yields = np.array([r.final_yield for r in snge_results])

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(cle_yields, snge_yields)

    # t-test
    t_stat, t_pvalue = stats.ttest_ind(cle_yields, snge_yields)

    # Levene's test for variance
    levene_stat, levene_pvalue = stats.levene(cle_yields, snge_yields)

    result = {
        'cle_mean': np.mean(cle_yields),
        'cle_std': np.std(cle_yields, ddof=1),
        'cle_cv': np.std(cle_yields, ddof=1) / np.mean(cle_yields) * 100,
        'snge_mean': np.mean(snge_yields),
        'snge_std': np.std(snge_yields, ddof=1),
        'snge_cv': np.std(snge_yields, ddof=1) / np.mean(snge_yields) * 100,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'same_distribution': ks_pvalue > 0.05,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'means_equal': t_pvalue > 0.05,
        'levene_statistic': levene_stat,
        'levene_pvalue': levene_pvalue,
        'variances_equal': levene_pvalue > 0.05,
        'cv_ratio': (np.std(snge_yields, ddof=1) / np.mean(snge_yields)) / \
                    (np.std(cle_yields, ddof=1) / np.mean(cle_yields))
    }

    if params is not None:
        result['volume'] = params.V
        result['N_molecules'] = params.N_A0

    return result


# =============================================================================
# CV Prediction from Gillespie SSA (Paper's Methodology)
# =============================================================================

def predict_cv_from_gillespie(params: NGEParameters,
                               n_runs: int = 10000,
                               show_progress: bool = True) -> dict:
    """
    Predict CV from physical parameters alone using Gillespie SSA.

    This is the paper's key methodology: CV emerges from physical parameters
    without fitting to variance data. The molecule count N = [A]₀ × V × Nₐ
    is calculated from physical conditions, not adjusted to match CV.

    The paper's approach:
    1. Fit rate constants (k₁, k₂, k₃) to mean kinetics only
    2. Run Gillespie simulations with physical N
    3. CV emerges naturally - no fitting to variance data

    "The CV is not imposed or fitted—it emerges from running stochastic
    simulations with physical parameters. The model either predicts the
    experimental CV or it doesn't."

    Args:
        params: NGEParameters with fitted k₁, k₂, k₃ (from mean kinetics)
        n_runs: Number of Gillespie simulations (default 10000 for accuracy)
        show_progress: Show progress bar during simulation

    Returns:
        Dictionary with:
            - predicted_cv: CV that emerges from Gillespie (%)
            - predicted_std: Standard deviation of yields
            - predicted_mean: Mean yield
            - predicted_skewness: Skewness of distribution
            - n_molecules: N = [A]₀ × V × Nₐ
            - alpha: k₁/k₃
            - beta: k₂[A]₀/k₃
            - note: Explanation of methodology
    """
    from .stochastic_fast import run_ensemble_gillespie_numba

    # Run Gillespie ensemble
    results = run_ensemble_gillespie_numba(params, n_runs=n_runs, show_progress=show_progress)

    # Compute statistics
    yields = np.array([r.final_yield for r in results])

    mean_yield = np.mean(yields)
    std_yield = np.std(yields, ddof=1)
    cv = std_yield / mean_yield * 100 if mean_yield > 0 else np.nan

    # Dimensionless parameters
    alpha = params.k1 / params.k3 if params.k3 > 0 else np.inf
    beta = params.k2 * params.A0 / params.k3 if params.k3 > 0 else np.inf

    return {
        'predicted_cv': cv,
        'predicted_std': std_yield,
        'predicted_mean': mean_yield,
        'predicted_skewness': stats.skew(yields),
        'predicted_kurtosis': stats.kurtosis(yields),
        'n_molecules': params.N_A0,
        'alpha': alpha,
        'beta': beta,
        'n_runs': n_runs,
        'yields': yields,
        'note': 'CV emerges from Gillespie SSA with physical N - no fitting to variance'
    }


def validate_cv_prediction(params: NGEParameters,
                           experimental_cv: float,
                           n_runs: int = 10000,
                           show_progress: bool = True) -> dict:
    """
    Compare predicted CV with experimental CV.

    Per the paper: "The model either predicts the experimental CV or it doesn't."

    This is the key validation step. If the predicted CV matches experimental
    CV, the stochastic framework is validated. If not, the physical model
    may be missing important mechanisms.

    Args:
        params: NGEParameters with fitted k₁, k₂, k₃
        experimental_cv: Experimental CV (in percent)
        n_runs: Number of Gillespie simulations
        show_progress: Show progress bar

    Returns:
        Dictionary with:
            - predicted_cv: CV from Gillespie
            - experimental_cv: Input experimental CV
            - cv_ratio: predicted/experimental
            - relative_error: |predicted - experimental| / experimental (%)
            - prediction_accurate: True if within 20% of experimental
            - interpretation: String describing result
    """
    prediction = predict_cv_from_gillespie(params, n_runs=n_runs, show_progress=show_progress)

    predicted_cv = prediction['predicted_cv']
    cv_ratio = predicted_cv / experimental_cv if experimental_cv > 0 else np.nan
    relative_error = abs(predicted_cv - experimental_cv) / experimental_cv * 100 if experimental_cv > 0 else np.nan

    # Consider prediction accurate if within 20%
    prediction_accurate = relative_error < 20

    if prediction_accurate:
        interpretation = (
            f"Model validated: Predicted CV ({predicted_cv:.1f}%) matches "
            f"experimental CV ({experimental_cv:.1f}%) within 20%."
        )
    elif predicted_cv > experimental_cv:
        interpretation = (
            f"Model overpredicts variability: Predicted CV ({predicted_cv:.1f}%) > "
            f"experimental CV ({experimental_cv:.1f}%). This may indicate "
            "experimental stabilization mechanisms not captured by the model."
        )
    else:
        interpretation = (
            f"Model underpredicts variability: Predicted CV ({predicted_cv:.1f}%) < "
            f"experimental CV ({experimental_cv:.1f}%). This may indicate "
            "additional noise sources (e.g., temperature fluctuations, impurities)."
        )

    return {
        'predicted_cv': predicted_cv,
        'experimental_cv': experimental_cv,
        'cv_ratio': cv_ratio,
        'relative_error': relative_error,
        'prediction_accurate': prediction_accurate,
        'interpretation': interpretation,
        'n_molecules': prediction['n_molecules'],
        'alpha': prediction['alpha'],
        'beta': prediction['beta'],
        'n_runs': n_runs
    }


def compare_seeded_vs_unseeded(params: NGEParameters,
                                seed_levels: List[float],
                                n_runs: int = 1000,
                                show_progress: bool = True) -> dict:
    """
    Reproduce Table 2 from paper: seeding reduces CV.

    Per the paper: "seeding with pre-formed graphene should bypass
    the critical window and dramatically reduce variability"

    Key validation: CV reduction follows model without refitting.
    The same rate constants (k₁, k₂, k₃) are used - only B₀ changes.

    Args:
        params: NGEParameters (B0 will be modified for seeding)
        seed_levels: List of initial B concentrations (M) to test
        n_runs: Simulations per condition
        show_progress: Show progress bar

    Returns:
        Dictionary with:
            - unseeded_cv: CV with B0=0
            - seeded_cvs: Dict mapping seed_level -> CV
            - cv_reductions: Dict mapping seed_level -> fold reduction
            - note: Explanation
    """
    from .stochastic_fast import run_ensemble_gillespie_numba

    results_dict = {}

    # Unseeded (baseline)
    params_unseeded = NGEParameters(
        k1=params.k1, k2=params.k2, k3=params.k3,
        A0=params.A0, B0=0.0, V=params.V, t_max=params.t_max
    )

    if show_progress:
        print("Running unseeded simulations...")
    results_unseeded = run_ensemble_gillespie_numba(
        params_unseeded, n_runs=n_runs, show_progress=show_progress
    )
    yields_unseeded = np.array([r.final_yield for r in results_unseeded])
    cv_unseeded = np.std(yields_unseeded, ddof=1) / np.mean(yields_unseeded) * 100

    results_dict['unseeded'] = {
        'cv': cv_unseeded,
        'mean': np.mean(yields_unseeded),
        'std': np.std(yields_unseeded, ddof=1),
        'B0': 0.0
    }

    # Seeded conditions
    seeded_cvs = {}
    cv_reductions = {}

    for seed_level in seed_levels:
        params_seeded = NGEParameters(
            k1=params.k1, k2=params.k2, k3=params.k3,
            A0=params.A0, B0=seed_level, V=params.V, t_max=params.t_max
        )

        if show_progress:
            print(f"Running seeded simulations (B0={seed_level:.2e} M)...")
        results_seeded = run_ensemble_gillespie_numba(
            params_seeded, n_runs=n_runs, show_progress=show_progress
        )
        yields_seeded = np.array([r.final_yield for r in results_seeded])
        cv_seeded = np.std(yields_seeded, ddof=1) / np.mean(yields_seeded) * 100

        seeded_cvs[seed_level] = cv_seeded
        cv_reductions[seed_level] = cv_unseeded / cv_seeded if cv_seeded > 0 else np.inf

        results_dict[f'seeded_{seed_level}'] = {
            'cv': cv_seeded,
            'mean': np.mean(yields_seeded),
            'std': np.std(yields_seeded, ddof=1),
            'B0': seed_level,
            'cv_reduction': cv_reductions[seed_level]
        }

    return {
        'unseeded_cv': cv_unseeded,
        'seeded_cvs': seeded_cvs,
        'cv_reductions': cv_reductions,
        'detailed_results': results_dict,
        'note': 'Seeding bypasses critical period, reducing CV without refitting parameters'
    }


def compute_critical_period_gillespie(results: List[SimulationResult],
                                       threshold: float = 0.1) -> dict:
    """
    Identify critical sensitivity window from Gillespie trajectories.

    The critical period is when:
    - [B] is small
    - Autocatalytic growth is weak
    - Stochastic etching can destroy nascent nuclei

    This emerges naturally from dynamics - no phenomenological σ(b) model needed.

    Args:
        results: List of SimulationResult from Gillespie simulations
        threshold: Yield threshold defining end of critical period (fraction of A0)

    Returns:
        Dictionary with critical period statistics
    """
    n_runs = len(results)
    t_critical = np.zeros(n_runs)

    for i, result in enumerate(results):
        # Compute yield trajectory
        A0 = result.B_concentration[0] + result.A_concentration[0]
        yields = result.B_concentration / A0 if A0 > 0 else result.B_concentration

        # Find first crossing of threshold
        crossing_idx = np.where(yields >= threshold)[0]
        if len(crossing_idx) > 0:
            t_critical[i] = result.times[crossing_idx[0]]
        else:
            t_critical[i] = result.times[-1]  # Never reached threshold

    return {
        'threshold': threshold,
        'mean_t_critical': np.mean(t_critical),
        'std_t_critical': np.std(t_critical, ddof=1),
        'cv_t_critical': np.std(t_critical, ddof=1) / np.mean(t_critical) * 100 if np.mean(t_critical) > 0 else np.nan,
        'min_t_critical': np.min(t_critical),
        'max_t_critical': np.max(t_critical),
        't_critical_values': t_critical,
        'note': 'Critical period emerges from Gillespie dynamics - no σ(b) model'
    }


def analyze_trajectory_divergence(results: List[SimulationResult],
                                   time_points: Optional[np.ndarray] = None) -> dict:
    """
    Analyze how trajectories diverge during critical period.

    Per paper: "Early differences are amplified by autocatalysis"

    Args:
        results: List of SimulationResult from Gillespie simulations
        time_points: Time points for analysis (default: from first result)

    Returns:
        Dictionary with divergence analysis
    """
    if time_points is None:
        time_points = results[0].times.copy()

    n_runs = len(results)
    n_times = len(time_points)

    # Interpolate all trajectories to common time points
    B_matrix = np.zeros((n_runs, n_times))
    for i, res in enumerate(results):
        B_matrix[i, :] = np.interp(time_points, res.times, res.B_concentration)

    # Compute statistics at each time point
    means = np.mean(B_matrix, axis=0)
    stds = np.std(B_matrix, axis=0, ddof=1)
    cvs = np.zeros(n_times)
    nonzero = means > 1e-15
    cvs[nonzero] = stds[nonzero] / means[nonzero] * 100

    # Find peak CV (maximum divergence point)
    peak_cv_idx = np.argmax(cvs)
    peak_cv_time = time_points[peak_cv_idx]
    peak_cv = cvs[peak_cv_idx]

    # Find time when CV stabilizes (drops to 50% of peak)
    stabilization_idx = np.where(cvs[peak_cv_idx:] < peak_cv * 0.5)[0]
    if len(stabilization_idx) > 0:
        stabilization_time = time_points[peak_cv_idx + stabilization_idx[0]]
    else:
        stabilization_time = time_points[-1]

    return {
        'time_points': time_points,
        'means': means,
        'stds': stds,
        'cvs': cvs,
        'peak_cv': peak_cv,
        'peak_cv_time': peak_cv_time,
        'stabilization_time': stabilization_time,
        'note': 'Early divergence amplified by autocatalysis'
    }
