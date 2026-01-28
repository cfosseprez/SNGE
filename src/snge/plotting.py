"""
Visualization functions for NGE simulation results.
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .models import NGEParameters, SimulationResult
from .deterministic import solve_deterministic
from .analysis import compute_yield_statistics, compute_cv_over_time


def plot_ensemble_trajectories(results: List[SimulationResult],
                               params: NGEParameters,
                               n_show: int = 50,
                               figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot ensemble of stochastic trajectories with deterministic comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: Sample trajectories
    ax1 = axes[0]

    # Plot sample of stochastic trajectories
    alpha = min(0.3, 10 / n_show)
    for i, res in enumerate(results[:n_show]):
        label = 'Individual runs' if i == 0 else None
        ax1.plot(res.times, res.B_concentration * 1000,  # Convert to mM
                 'black', alpha=alpha, linewidth=0.5, label=label)

    # Plot deterministic solution (dotted)
    t_det = np.linspace(0, params.t_max, 500)
    _, B_det = solve_deterministic(params, t_det)
    ax1.plot(t_det, B_det * 1000, 'crimson', linestyle=':', linewidth=2, label='Deterministic')

    # Compute ensemble mean and std for confidence bands
    t_common = np.linspace(0, params.t_max, 200)
    B_matrix = np.zeros((len(results), len(t_common)))
    for i, res in enumerate(results):
        B_matrix[i, :] = np.interp(t_common, res.times, res.B_concentration)
    mean_B = np.mean(B_matrix, axis=0) * 1000
    std_B = np.std(B_matrix, axis=0) * 1000

    # Plot 2-sigma bands (fill + lines)
    ax1.fill_between(t_common, mean_B - 2*std_B, mean_B + 2*std_B,
                     alpha=0.1, color='gray')
    ax1.plot(t_common, mean_B + 2*std_B, 'gray', linestyle='--', linewidth=1, label='±2σ')
    ax1.plot(t_common, mean_B - 2*std_B, 'gray', linestyle='--', linewidth=1)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('[Graphene] (mM)', fontsize=12)
    ax1.set_title('Graphene Synthesis (Gillespie SSA)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, params.t_max)
    ax1.set_ylim(0, None)

    # Right panel: Yield distribution
    ax2 = axes[1]
    yields = np.array([r.final_yield for r in results]) * 100  # Convert to %

    # Calculate statistics
    sim_mean = np.mean(yields)
    sim_std = np.std(yields)
    cv = sim_std / sim_mean * 100

    # More beautiful histogram
    ax2.hist(yields, bins=35, density=True, alpha=0.7, color='#4A90D9',
             edgecolor='white', linewidth=0.5, label='Simulated yields')

    # Add kernel density estimate (blue)
    kde = stats.gaussian_kde(yields)
    x_kde = np.linspace(yields.min() - sim_std, yields.max() + sim_std, 200)
    ax2.plot(x_kde, kde(x_kde), 'crimson', linewidth=1, label='Distribution fit')

    # Mark 2-sigma range
    ax2.axvspan(sim_mean - 2*sim_std, sim_mean + 2*sim_std, alpha=0.15, color='gray',
                label=f'±2σ = ±{2*sim_std:.3f}%')

    # Mark mean
    ax2.axvline(sim_mean, color='darkblue', linestyle='-', linewidth=2,
                label=f'Mean yield = {sim_mean:.2f}%')

    ax2.set_xlabel('Final Yield (%)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)  # Normalized so area = 1
    ax2.set_title(f'Yield Distribution (n={len(results)}, CV={cv:.1f}%)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    return fig


def plot_cv_evolution(results: List[SimulationResult],
                      params: NGEParameters,
                      figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
    """
    Plot how CV evolves during synthesis (identifies critical period).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Compute CV over time
    t_points = np.linspace(0, params.t_max, 200)
    t_points, cv = compute_cv_over_time(results, t_points)

    # Left panel: CV vs time
    ax1 = axes[0]
    ax1.plot(t_points, cv, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax1.set_title('Trajectory Divergence Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Mark critical period: when most variability is generated
    # Better definition: period where CV is still decreasing significantly
    # (i.e., before it stabilizes to final value)
    # Use: when CV reaches within 20% of its final value
    final_cv = cv[-1] if cv[-1] > 0 else np.mean(cv[-10:])
    cv_threshold = final_cv * 1.5  # 50% above final CV
    # Find where CV first drops below threshold (after initial spike)
    start_idx = min(10, len(cv) // 10)  # Skip initial artifact
    below_threshold = np.where(cv[start_idx:] < cv_threshold)[0]
    if len(below_threshold) > 0:
        critical_end_idx = below_threshold[0] + start_idx
        critical_end = t_points[critical_end_idx]
        ax1.axvline(critical_end, color='red', linestyle='--',
                    label=f'Nucleation phase ends ≈ {critical_end:.0f}s')
        ax1.axvspan(0, critical_end, alpha=0.2, color='red')
        ax1.legend()

    # Right panel: Mean trajectory with std bands
    ax2 = axes[1]

    # Interpolate all trajectories
    n_runs = len(results)
    B_matrix = np.zeros((n_runs, len(t_points)))
    for i, res in enumerate(results):
        B_matrix[i, :] = np.interp(t_points, res.times, res.B_concentration)

    mean_B = np.mean(B_matrix, axis=0) * 1000  # mM
    std_B = np.std(B_matrix, axis=0) * 1000

    ax2.fill_between(t_points, mean_B - 2 * std_B, mean_B + 2 * std_B,
                     alpha=0.3, color='blue', label='±2σ')
    ax2.fill_between(t_points, mean_B - std_B, mean_B + std_B,
                     alpha=0.3, color='blue', label='±1σ')
    ax2.plot(t_points, mean_B, 'b-', linewidth=2, label='Mean')

    # Deterministic
    _, B_det = solve_deterministic(params, t_points)
    ax2.plot(t_points, B_det * 1000, 'r--', linewidth=2, label='Deterministic')

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('[Graphene] (mM)', fontsize=12)
    ax2.set_title('Mean Trajectory with Confidence Bands', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distribution_comparison(results_gillespie: List[SimulationResult],
                                 results_em: List[SimulationResult],
                                 experimental_yields: np.ndarray = None,
                                 figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Compare yield distributions from different methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    yields_g = np.array([r.final_yield for r in results_gillespie]) * 100
    yields_em = np.array([r.final_yield for r in results_em]) * 100

    # Left panel: Histograms
    ax1 = axes[0]
    bins = np.linspace(min(yields_g.min(), yields_em.min()),
                       max(yields_g.max(), yields_em.max()), 30)

    ax1.hist(yields_g, bins=bins, density=True, alpha=0.5,
             label=f'Gillespie SSA (CV={np.std(yields_g) / np.mean(yields_g) * 100:.1f}%)',
             color='blue')
    ax1.hist(yields_em, bins=bins, density=True, alpha=0.5,
             label=f'Langevin CLE (CV={np.std(yields_em) / np.mean(yields_em) * 100:.1f}%)',
             color='green')

    if experimental_yields is not None:
        ax1.hist(experimental_yields * 100, bins=bins, density=True, alpha=0.5,
                 label=f'Experimental (CV={np.std(experimental_yields) / np.mean(experimental_yields) * 100:.1f}%)',
                 color='red')

    ax1.set_xlabel('Final Yield (%)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Yield Distribution Comparison', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Q-Q plot
    ax2 = axes[1]

    # Sort both distributions
    yields_g_sorted = np.sort(yields_g)
    yields_em_sorted = np.sort(yields_em)

    # Interpolate to same quantiles
    n_points = min(len(yields_g), len(yields_em))
    quantiles = np.linspace(0, 1, n_points)
    g_quantiles = np.percentile(yields_g, quantiles * 100)
    em_quantiles = np.percentile(yields_em, quantiles * 100)

    ax2.scatter(g_quantiles, em_quantiles, alpha=0.5, s=20)

    # Reference line
    min_val = min(g_quantiles.min(), em_quantiles.min())
    max_val = max(g_quantiles.max(), em_quantiles.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect agreement')

    ax2.set_xlabel('Gillespie SSA Quantiles (%)', fontsize=12)
    ax2.set_ylabel('Langevin CLE Quantiles (%)', fontsize=12)
    ax2.set_title('Q-Q Plot: Method Comparison', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_summary_figure(results: List[SimulationResult],
                        params: NGEParameters,
                        experimental_yields: np.ndarray = None,
                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create comprehensive summary figure for publication.
    """
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Panel A: Sample trajectories
    ax_a = fig.add_subplot(gs[0, 0])
    n_show = 30
    alpha = 0.2
    for i, res in enumerate(results[:n_show]):
        ax_a.plot(res.times, res.B_concentration * 1000, 'b-', alpha=alpha, linewidth=0.5)

    t_det = np.linspace(0, params.t_max, 500)
    _, B_det = solve_deterministic(params, t_det)
    ax_a.plot(t_det, B_det * 1000, 'r-', linewidth=2, label='Deterministic')
    ax_a.set_xlabel('Time (s)')
    ax_a.set_ylabel('[B] (mM)')
    ax_a.set_title('(A) Stochastic Trajectories')
    ax_a.legend(loc='lower right')
    ax_a.grid(True, alpha=0.3)

    # Panel B: CV evolution
    ax_b = fig.add_subplot(gs[0, 1])
    t_points = np.linspace(0.1, params.t_max, 200)
    _, cv = compute_cv_over_time(results, t_points)
    ax_b.plot(t_points, cv, 'b-', linewidth=2)
    ax_b.set_xlabel('Time (s)')
    ax_b.set_ylabel('CV (%)')
    ax_b.set_title('(B) Critical Period')
    ax_b.grid(True, alpha=0.3)

    # Shade nucleation phase (where variability is generated)
    final_cv = cv[-1] if cv[-1] > 0 else np.mean(cv[-10:])
    cv_threshold = final_cv * 1.5
    start_idx = min(10, len(cv) // 10)
    below_threshold = np.where(cv[start_idx:] < cv_threshold)[0]
    if len(below_threshold) > 0:
        critical_end = t_points[below_threshold[0] + start_idx]
        ax_b.axvspan(0, critical_end, alpha=0.2, color='red', label='Nucleation phase')
        ax_b.legend()

    # Panel C: Yield histogram
    ax_c = fig.add_subplot(gs[0, 2])
    yields = np.array([r.final_yield for r in results]) * 100

    ax_c.hist(yields, bins=30, density=True, alpha=0.7, color='steelblue',
              edgecolor='white', label='Simulated')

    if experimental_yields is not None:
        ax_c.hist(experimental_yields * 100, bins=30, density=True, alpha=0.5,
                  color='red', edgecolor='white', label='Experimental')

    ax_c.axvline(np.mean(yields), color='blue', linestyle='--', linewidth=2)
    ax_c.axvline(B_det[-1] / params.A0 * 100, color='red', linestyle='--', linewidth=2)

    ax_c.set_xlabel('Final Yield (%)')
    ax_c.set_ylabel('Density')
    ax_c.set_title(f'(C) Yield Distribution (CV={np.std(yields) / np.mean(yields) * 100:.1f}%)')
    if experimental_yields is not None:
        ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    # Panel D: Mean with confidence bands
    ax_d = fig.add_subplot(gs[1, 0])
    n_runs = len(results)
    B_matrix = np.zeros((n_runs, len(t_points)))
    for i, res in enumerate(results):
        B_matrix[i, :] = np.interp(t_points, res.times, res.B_concentration)

    mean_B = np.mean(B_matrix, axis=0) * 1000
    std_B = np.std(B_matrix, axis=0) * 1000

    ax_d.fill_between(t_points, mean_B - 2 * std_B, mean_B + 2 * std_B,
                      alpha=0.2, color='blue')
    ax_d.fill_between(t_points, mean_B - std_B, mean_B + std_B,
                      alpha=0.3, color='blue')
    ax_d.plot(t_points, mean_B, 'b-', linewidth=2, label='Mean ± σ, 2σ')
    ax_d.plot(t_points, np.interp(t_points, t_det, B_det * 1000), 'r--',
              linewidth=2, label='Deterministic')

    ax_d.set_xlabel('Time (s)')
    ax_d.set_ylabel('[B] (mM)')
    ax_d.set_title('(D) Ensemble Statistics')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    # Panel E: Statistics table
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.axis('off')

    stats_dict = compute_yield_statistics(results)
    table_data = [
        ['Statistic', 'Value'],
        ['N simulations', f'{stats_dict["n"]}'],
        ['Mean yield', f'{stats_dict["mean"] * 100:.2f}%'],
        ['Std deviation', f'{stats_dict["std"] * 100:.2f}%'],
        ['CV', f'{stats_dict["cv"]:.1f}%'],
        ['Skewness', f'{stats_dict["skewness"]:.3f}'],
        ['Min yield', f'{stats_dict["min"] * 100:.2f}%'],
        ['Max yield', f'{stats_dict["max"] * 100:.2f}%'],
    ]

    table = ax_e.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Header styling
    for j in range(2):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    ax_e.set_title('(E) Summary Statistics')

    # Panel F: Q-Q plot against normal
    ax_f = fig.add_subplot(gs[1, 2])
    stats.probplot(yields, dist="norm", plot=ax_f)
    ax_f.set_title('(F) Q-Q Plot (vs Normal)')
    ax_f.grid(True, alpha=0.3)

    plt.suptitle('Stochastic Graphene Synthesis Simulation',
                 fontsize=14, fontweight='bold', y=1.02)

    return fig
