"""
NGE Model Parameter Fitting
============================

Fit the Nucleation-Growth-Etching (NGE) rate constants to experimental
time-resolved yield data.

Workflow:
1. Load experimental data (multiple runs)
2. Compute mean kinetics
3. Fit deterministic NGE to mean
4. Validate fit quality
5. Export parameters for stochastic simulations

Author: [Your Name]
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy import stats
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExperimentalData:
    """Container for experimental yield data"""
    times: np.ndarray  # Time points (s)
    yields: np.ndarray  # 2D array: [n_runs, n_timepoints], yields as fraction (0-1)
    A0: float  # Initial precursor concentration (M)

    @property
    def n_runs(self) -> int:
        return self.yields.shape[0]

    @property
    def n_timepoints(self) -> int:
        return len(self.times)

    @property
    def mean_yield(self) -> np.ndarray:
        """Mean yield across all runs"""
        return np.mean(self.yields, axis=0)

    @property
    def std_yield(self) -> np.ndarray:
        """Standard deviation of yield"""
        return np.std(self.yields, axis=0, ddof=1)

    @property
    def sem_yield(self) -> np.ndarray:
        """Standard error of the mean"""
        return self.std_yield / np.sqrt(self.n_runs)

    @property
    def final_yields(self) -> np.ndarray:
        """Final yields from each run"""
        return self.yields[:, -1]

    @property
    def cv_final(self) -> float:
        """Coefficient of variation of final yields (%)"""
        return np.std(self.final_yields, ddof=1) / np.mean(self.final_yields) * 100


@dataclass
class FitResult:
    """Results from NGE model fitting"""
    k1: float  # Nucleation rate constant (s^-1)
    k2: float  # Growth rate constant (M^-1 s^-1)
    k3: float  # Etching rate constant (s^-1)
    k1_err: float  # Standard error of k1
    k2_err: float  # Standard error of k2
    k3_err: float  # Standard error of k3
    A0: float  # Initial precursor concentration
    r_squared: float  # Coefficient of determination
    rmse: float  # Root mean square error
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    residuals: np.ndarray  # Residuals
    converged: bool  # Did optimization converge?

    def __str__(self):
        return f"""
NGE Fit Results:
================
k1 = {self.k1:.4e} ± {self.k1_err:.4e} s^-1     (nucleation)
k2 = {self.k2:.4e} ± {self.k2_err:.4e} M^-1 s^-1 (growth)
k3 = {self.k3:.4e} ± {self.k3_err:.4e} s^-1     (etching)

Goodness of fit:
  R² = {self.r_squared:.4f}
  RMSE = {self.rmse:.4e}
  AIC = {self.aic:.1f}
  BIC = {self.bic:.1f}

Steady-state yield (predicted): {self.steady_state_yield * 100:.1f}%
"""

    @property
    def steady_state_yield(self) -> float:
        """Predicted steady-state yield as fraction"""
        # At steady state: k1*A + k2*A*B - k3*B = 0
        # With A = A0 - B, solve for B
        # This is approximate - numerical steady state is more accurate
        from scipy.optimize import fsolve
        def ss_eq(B):
            A = self.A0 - B
            return self.k1 * A + self.k2 * A * B - self.k3 * B

        B_ss = fsolve(ss_eq, self.A0 * 0.5)[0]
        return B_ss / self.A0


# =============================================================================
# NGE MODEL
# =============================================================================

def nge_ode(y: np.ndarray, t: float, k1: float, k2: float, k3: float) -> np.ndarray:
    """
    NGE differential equations.

    d[A]/dt = -k1[A] - k2[A][B]
    d[B]/dt = k1[A] + k2[A][B] - k3[B]
    """
    A, B = y
    dA = -k1 * A - k2 * A * B
    dB = k1 * A + k2 * A * B - k3 * B
    return [dA, dB]


def solve_nge(times: np.ndarray, k1: float, k2: float, k3: float,
              A0: float, B0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve NGE model.

    Returns:
        A, B: Concentration arrays
    """
    y0 = [A0, B0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = odeint(nge_ode, y0, times, args=(k1, k2, k3))
    return solution[:, 0], solution[:, 1]


def nge_yield_curve(times: np.ndarray, k1: float, k2: float, k3: float,
                    A0: float) -> np.ndarray:
    """
    Compute yield curve Y(t) = [B](t) / [A]₀
    """
    _, B = solve_nge(times, k1, k2, k3, A0)
    return B / A0


# =============================================================================
# FINKE-WATZKY MODEL (for comparison)
# =============================================================================

def fw_ode(y: np.ndarray, t: float, k1: float, k2: float) -> np.ndarray:
    """
    Finke-Watzky differential equations (no etching).

    d[A]/dt = -k1[A] - k2[A][B]
    d[B]/dt = k1[A] + k2[A][B]
    """
    A, B = y
    dA = -k1 * A - k2 * A * B
    dB = k1 * A + k2 * A * B
    return [dA, dB]


def solve_fw(times: np.ndarray, k1: float, k2: float,
             A0: float, B0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Finke-Watzky model."""
    y0 = [A0, B0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = odeint(fw_ode, y0, times, args=(k1, k2))
    return solution[:, 0], solution[:, 1]


def fw_yield_curve(times: np.ndarray, k1: float, k2: float, A0: float) -> np.ndarray:
    """Compute FW yield curve."""
    _, B = solve_fw(times, k1, k2, A0)
    return B / A0


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def fit_nge_to_mean(data: ExperimentalData,
                    initial_guess: Tuple[float, float, float] = None,
                    bounds: Tuple[Tuple, Tuple] = None,
                    method: str = 'least_squares') -> FitResult:
    """
    Fit NGE model to mean experimental yield curve.

    Args:
        data: ExperimentalData object
        initial_guess: (k1, k2, k3) initial values
        bounds: ((k1_min, k2_min, k3_min), (k1_max, k2_max, k3_max))
        method: 'least_squares', 'differential_evolution', or 'both'

    Returns:
        FitResult object
    """
    times = data.times
    y_exp = data.mean_yield
    A0 = data.A0
    n_points = len(times)

    # Default initial guess (order of magnitude estimates)
    if initial_guess is None:
        # Estimate from data characteristics
        t_half = times[np.argmin(np.abs(y_exp - 0.5 * y_exp[-1]))]
        initial_guess = (1e-4, 0.1, 0.01)

    # Default bounds
    if bounds is None:
        bounds = ((1e-8, 1e-4, 1e-6), (1e-1, 1e3, 1e1))

    # Objective function
    def objective(params):
        k1, k2, k3 = params
        try:
            y_pred = nge_yield_curve(times, k1, k2, k3, A0)
            return np.sum((y_exp - y_pred) ** 2)
        except (ValueError, RuntimeWarning, FloatingPointError):
            return 1e10

    # Fitting function for curve_fit
    def fit_func(t, k1, k2, k3):
        return nge_yield_curve(t, k1, k2, k3, A0)

    # Run optimization
    if method == 'differential_evolution' or method == 'both':
        # Global optimization (slower but more robust)
        de_bounds = list(zip(bounds[0], bounds[1]))
        de_result = differential_evolution(objective, de_bounds,
                                           maxiter=1000, seed=42)
        de_params = de_result.x
        de_cost = de_result.fun

    if method == 'least_squares' or method == 'both':
        # Local optimization (faster, needs good initial guess)
        try:
            if method == 'both':
                p0 = de_params  # Use DE result as starting point
            else:
                p0 = initial_guess

            popt, pcov = curve_fit(fit_func, times, y_exp, p0=p0,
                                   bounds=bounds, maxfev=10000)
            ls_params = popt
            ls_cost = objective(popt)

            # Parameter uncertainties from covariance matrix
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(f"curve_fit failed: {e}")
            ls_params = de_params if method == 'both' else initial_guess
            ls_cost = objective(ls_params)
            perr = np.array([np.nan, np.nan, np.nan])

    # Select best result
    if method == 'both':
        if ls_cost < de_cost:
            best_params = ls_params
        else:
            best_params = de_params
            # Re-estimate errors at DE solution
            try:
                _, pcov = curve_fit(fit_func, times, y_exp, p0=de_params,
                                    bounds=bounds, maxfev=1000)
                perr = np.sqrt(np.diag(pcov))
            except (RuntimeError, ValueError):
                perr = np.array([np.nan, np.nan, np.nan])
    elif method == 'differential_evolution':
        best_params = de_params
        perr = np.array([np.nan, np.nan, np.nan])  # DE doesn't give covariance
    else:
        best_params = ls_params

    k1, k2, k3 = best_params

    # Compute fit statistics
    y_pred = nge_yield_curve(times, k1, k2, k3, A0)
    residuals = y_exp - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n_points)

    # Information criteria
    n_params = 3
    aic = n_points * np.log(ss_res / n_points) + 2 * n_params
    bic = n_points * np.log(ss_res / n_points) + n_params * np.log(n_points)

    return FitResult(
        k1=k1, k2=k2, k3=k3,
        k1_err=perr[0], k2_err=perr[1], k3_err=perr[2],
        A0=A0,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        residuals=residuals,
        converged=True
    )


def fit_fw_to_mean(data: ExperimentalData,
                   initial_guess: Tuple[float, float] = None,
                   bounds: Tuple[Tuple, Tuple] = None) -> dict:
    """
    Fit Finke-Watzky model (no etching) for comparison.
    """
    times = data.times
    y_exp = data.mean_yield
    A0 = data.A0
    n_points = len(times)

    if initial_guess is None:
        initial_guess = (1e-4, 0.1)

    if bounds is None:
        bounds = ((1e-8, 1e-4), (1e-1, 1e3))

    def fit_func(t, k1, k2):
        return fw_yield_curve(t, k1, k2, A0)

    try:
        popt, pcov = curve_fit(fit_func, times, y_exp, p0=initial_guess,
                               bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"FW fit failed: {e}")
        return None

    k1, k2 = popt
    y_pred = fw_yield_curve(times, k1, k2, A0)
    residuals = y_exp - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n_points)

    n_params = 2
    aic = n_points * np.log(ss_res / n_points) + 2 * n_params
    bic = n_points * np.log(ss_res / n_points) + n_params * np.log(n_points)

    return {
        'k1': k1, 'k2': k2,
        'k1_err': perr[0], 'k2_err': perr[1],
        'r_squared': r_squared,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
        'y_pred': y_pred
    }


def compare_nge_vs_fw(data: ExperimentalData, nge_fit: FitResult, fw_fit: dict) -> dict:
    """
    Statistical comparison of NGE vs FW models.
    """
    n = len(data.times)

    # F-test for nested models (FW is nested in NGE with k3=0)
    ss_fw = np.sum(fw_fit['residuals'] ** 2)
    ss_nge = np.sum(nge_fit.residuals ** 2)
    df1 = 1  # Additional parameter in NGE
    df2 = n - 3  # Residual df for NGE

    f_stat = ((ss_fw - ss_nge) / df1) / (ss_nge / df2)
    f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)

    # AIC comparison
    delta_aic = fw_fit['aic'] - nge_fit.aic  # Positive = NGE better

    # BIC comparison
    delta_bic = fw_fit['bic'] - nge_fit.bic

    return {
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        'nge_preferred_ftest': f_pvalue < 0.05,
        'delta_aic': delta_aic,
        'nge_preferred_aic': delta_aic > 2,
        'delta_bic': delta_bic,
        'nge_preferred_bic': delta_bic > 2,
        'fw_r_squared': fw_fit['r_squared'],
        'nge_r_squared': nge_fit.r_squared,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_fit_results(data: ExperimentalData, nge_fit: FitResult,
                     fw_fit: dict = None, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Comprehensive visualization of fitting results.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Generate model predictions
    t_fine = np.linspace(0, data.times[-1] * 1.1, 500)
    y_nge = nge_yield_curve(t_fine, nge_fit.k1, nge_fit.k2, nge_fit.k3, data.A0)

    if fw_fit is not None:
        y_fw = fw_yield_curve(t_fine, fw_fit['k1'], fw_fit['k2'], data.A0)

    # Panel A: Data and fits
    ax_a = fig.add_subplot(gs[0, 0:2])

    # Plot individual runs (faint)
    for i in range(data.n_runs):
        ax_a.plot(data.times, data.yields[i, :] * 100, 'o-',
                  color='gray', alpha=0.3, markersize=3, linewidth=0.5)

    # Plot mean with error bars
    ax_a.errorbar(data.times, data.mean_yield * 100, yerr=data.sem_yield * 100,
                  fmt='ko', markersize=8, capsize=3, label='Mean ± SEM', linewidth=2)

    # Plot NGE fit
    ax_a.plot(t_fine, y_nge * 100, 'b-', linewidth=2.5,
              label=f'NGE (R²={nge_fit.r_squared:.4f})')

    # Plot FW fit if available
    if fw_fit is not None:
        ax_a.plot(t_fine, y_fw * 100, 'r--', linewidth=2,
                  label=f'FW (R²={fw_fit["r_squared"]:.4f})')

    ax_a.set_xlabel('Time (s)', fontsize=12)
    ax_a.set_ylabel('Yield (%)', fontsize=12)
    ax_a.set_title('(A) Model Fits to Mean Kinetics', fontsize=12)
    ax_a.legend(loc='lower right')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, data.times[-1] * 1.05)
    ax_a.set_ylim(0, None)

    # Panel B: Residuals
    ax_b = fig.add_subplot(gs[0, 2])

    ax_b.scatter(data.times, nge_fit.residuals * 100, c='blue', s=50,
                 label='NGE', alpha=0.7)
    if fw_fit is not None:
        ax_b.scatter(data.times, fw_fit['residuals'] * 100, c='red', s=50,
                     marker='x', label='FW', alpha=0.7)

    ax_b.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_b.set_xlabel('Time (s)', fontsize=12)
    ax_b.set_ylabel('Residual (%)', fontsize=12)
    ax_b.set_title('(B) Residuals', fontsize=12)
    ax_b.legend()
    ax_b.grid(True, alpha=0.3)

    # Panel C: Final yield distribution
    ax_c = fig.add_subplot(gs[1, 0])

    ax_c.hist(data.final_yields * 100, bins=15, density=True, alpha=0.7,
              color='steelblue', edgecolor='white')
    ax_c.axvline(np.mean(data.final_yields) * 100, color='blue',
                 linestyle='--', linewidth=2, label='Mean')
    ax_c.axvline(nge_fit.steady_state_yield * 100, color='red',
                 linestyle='--', linewidth=2, label='NGE prediction')

    ax_c.set_xlabel('Final Yield (%)', fontsize=12)
    ax_c.set_ylabel('Density', fontsize=12)
    ax_c.set_title(f'(C) Yield Distribution (CV={data.cv_final:.1f}%)', fontsize=12)
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    # Panel D: Parameter values
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')

    table_data = [
        ['Parameter', 'Value', 'Unit'],
        ['k₁', f'{nge_fit.k1:.3e} ± {nge_fit.k1_err:.1e}', 's⁻¹'],
        ['k₂', f'{nge_fit.k2:.3e} ± {nge_fit.k2_err:.1e}', 'M⁻¹s⁻¹'],
        ['k₃', f'{nge_fit.k3:.3e} ± {nge_fit.k3_err:.1e}', 's⁻¹'],
        ['', '', ''],
        ['[A]₀', f'{data.A0:.4f}', 'M'],
        ['R²', f'{nge_fit.r_squared:.4f}', ''],
        ['RMSE', f'{nge_fit.rmse:.4e}', ''],
        ['AIC', f'{nge_fit.aic:.1f}', ''],
    ]

    table = ax_d.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.35, 0.45, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax_d.set_title('(D) Fitted Parameters', fontsize=12, pad=20)

    # Panel E: Model comparison
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis('off')

    if fw_fit is not None:
        comparison = compare_nge_vs_fw(data, nge_fit, fw_fit)

        comp_data = [
            ['Metric', 'FW', 'NGE', 'Better'],
            ['R²', f'{fw_fit["r_squared"]:.4f}', f'{nge_fit.r_squared:.4f}',
             'NGE' if nge_fit.r_squared > fw_fit['r_squared'] else 'FW'],
            ['RMSE', f'{fw_fit["rmse"]:.4e}', f'{nge_fit.rmse:.4e}',
             'NGE' if nge_fit.rmse < fw_fit['rmse'] else 'FW'],
            ['AIC', f'{fw_fit["aic"]:.1f}', f'{nge_fit.aic:.1f}',
             'NGE' if nge_fit.aic < fw_fit['aic'] else 'FW'],
            ['', '', '', ''],
            ['F-test p', '', f'{comparison["f_pvalue"]:.4f}',
             'NGE sig.' if comparison["f_pvalue"] < 0.05 else 'Not sig.'],
        ]

        table2 = ax_e.table(cellText=comp_data, loc='center', cellLoc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 1.4)

        for j in range(4):
            table2[(0, j)].set_facecolor('#4472C4')
            table2[(0, j)].set_text_props(color='white', weight='bold')

        ax_e.set_title('(E) Model Comparison', fontsize=12, pad=20)

    plt.suptitle('NGE Model Fitting Results', fontsize=14, fontweight='bold', y=1.02)

    return fig


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_data_from_csv(filepath: str, time_col: str = 'time',
                       yield_cols: List[str] = None, A0: float = 0.01) -> ExperimentalData:
    """
    Load experimental data from CSV file.

    Expected format:
        time, run1, run2, run3, ...
        0, 0, 0, 0, ...
        60, 0.02, 0.01, 0.03, ...
        ...

    Yields should be in fraction (0-1) or percentage (0-100).
    If max > 1, assumes percentage and converts.
    """
    df = pd.read_csv(filepath)

    times = df[time_col].values

    if yield_cols is None:
        yield_cols = [c for c in df.columns if c != time_col]

    yields = df[yield_cols].values.T  # Transpose to [n_runs, n_timepoints]

    # Convert percentage to fraction if needed
    if np.max(yields) > 1:
        yields = yields / 100.0

    return ExperimentalData(times=times, yields=yields, A0=A0)


def create_synthetic_data(k1: float, k2: float, k3: float, A0: float,
                          times: np.ndarray, n_runs: int = 20,
                          noise_level: float = 0.05,
                          seed: int = 42) -> ExperimentalData:
    """
    Create synthetic data for testing (useful for validation).
    """
    np.random.seed(seed)

    # Get true curve
    y_true = nge_yield_curve(times, k1, k2, k3, A0)

    # Add noise to create multiple "runs"
    yields = np.zeros((n_runs, len(times)))
    for i in range(n_runs):
        noise = np.random.normal(0, noise_level, len(times))
        yields[i, :] = np.clip(y_true + noise * y_true, 0, 1)

    return ExperimentalData(times=times, yields=yields, A0=A0)


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_parameters_for_simulation(fit_result: FitResult,
                                     output_file: str = 'nge_parameters.py'):
    """
    Export fitted parameters in format ready for stochastic simulations.
    """
    content = f'''"""
NGE Parameters from Experimental Fitting
=========================================
Generated automatically - do not edit manually
"""

# Fitted rate constants
k1 = {fit_result.k1:.6e}  # Nucleation rate (s^-1)
k2 = {fit_result.k2:.6e}  # Growth rate (M^-1 s^-1)
k3 = {fit_result.k3:.6e}  # Etching rate (s^-1)

# Parameter uncertainties (standard error)
k1_err = {fit_result.k1_err:.6e}
k2_err = {fit_result.k2_err:.6e}
k3_err = {fit_result.k3_err:.6e}

# Initial conditions
A0 = {fit_result.A0:.6e}  # Initial precursor concentration (M)

# Fit quality
r_squared = {fit_result.r_squared:.6f}
rmse = {fit_result.rmse:.6e}

# For use with stochastic_nge_simulations.py:
# from nge_parameters import k1, k2, k3, A0
# params = NGEParameters(k1=k1, k2=k2, k3=k3, A0=A0, B0=0.0, V=1e-6, t_max=600)
'''

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Parameters exported to {output_file}")


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

def main():
    """
    Example workflow demonstrating parameter fitting.
    """
    print("=" * 70)
    print("NGE MODEL PARAMETER FITTING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Option 1: Create synthetic data for demonstration
    # -------------------------------------------------------------------------
    print("\n--- Creating synthetic data for demonstration ---")

    # "True" parameters (pretend we don't know these)
    k1_true = 2e-4
    k2_true = 0.15
    k3_true = 0.008
    A0 = 0.01  # 10 mM precursor

    times = np.array([0, 30, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600])

    data = create_synthetic_data(k1_true, k2_true, k3_true, A0,
                                 times, n_runs=25, noise_level=0.03)

    print(f"Created {data.n_runs} synthetic runs")
    print(f"Time points: {data.n_timepoints}")
    print(f"Final yield CV: {data.cv_final:.1f}%")

    # -------------------------------------------------------------------------
    # Option 2: Load real data (uncomment and modify path)
    # -------------------------------------------------------------------------
    # data = load_data_from_csv('your_experimental_data.csv', A0=0.01)

    # -------------------------------------------------------------------------
    # Fit NGE model
    # -------------------------------------------------------------------------
    print("\n--- Fitting NGE model ---")

    nge_fit = fit_nge_to_mean(data, method='both')
    print(nge_fit)

    # -------------------------------------------------------------------------
    # Fit FW model for comparison
    # -------------------------------------------------------------------------
    print("\n--- Fitting FW model (for comparison) ---")

    fw_fit = fit_fw_to_mean(data)
    print(f"FW: k1 = {fw_fit['k1']:.4e}, k2 = {fw_fit['k2']:.4e}")
    print(f"FW R² = {fw_fit['r_squared']:.4f}")

    # -------------------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------------------
    print("\n--- Model comparison ---")

    comparison = compare_nge_vs_fw(data, nge_fit, fw_fit)
    print(f"F-test p-value: {comparison['f_pvalue']:.4f}")
    print(f"NGE preferred (F-test): {comparison['nge_preferred_ftest']}")
    print(f"ΔAIC (FW - NGE): {comparison['delta_aic']:.1f}")
    print(f"NGE preferred (AIC): {comparison['nge_preferred_aic']}")

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------
    print("\n--- Generating figures ---")

    fig = plot_fit_results(data, nge_fit, fw_fit)
    fig.savefig('nge_fitting_results.png', dpi=150, bbox_inches='tight')
    print("Saved: nge_fitting_results.png")

    # -------------------------------------------------------------------------
    # Export parameters for stochastic simulations
    # -------------------------------------------------------------------------
    print("\n--- Exporting parameters ---")

    export_parameters_for_simulation(nge_fit, 'nge_parameters.py')

    # -------------------------------------------------------------------------
    # Compare recovered vs true parameters (synthetic data only)
    # -------------------------------------------------------------------------
    print("\n--- Parameter recovery check (synthetic data) ---")
    print(
        f"k1: true = {k1_true:.4e}, fitted = {nge_fit.k1:.4e}, error = {abs(nge_fit.k1 - k1_true) / k1_true * 100:.1f}%")
    print(
        f"k2: true = {k2_true:.4e}, fitted = {nge_fit.k2:.4e}, error = {abs(nge_fit.k2 - k2_true) / k2_true * 100:.1f}%")
    print(
        f"k3: true = {k3_true:.4e}, fitted = {nge_fit.k3:.4e}, error = {abs(nge_fit.k3 - k3_true) / k3_true * 100:.1f}%")

    plt.show()

    print("\n" + "=" * 70)
    print("FITTING COMPLETE")
    print("=" * 70)

    return data, nge_fit, fw_fit


if __name__ == "__main__":
    data, nge_fit, fw_fit = main()