"""
Analysis functions for NGE simulation results.
"""

from typing import List, Tuple

import numpy as np
from scipy import stats

from .models import SimulationResult


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
