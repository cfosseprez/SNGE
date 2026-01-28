"""
CLI entry point for the SNGE package.

Allows running as:
    python -m snge
    snge (CLI command)
    python src/snge/__main__.py (direct execution)
"""

# Handle direct execution by setting up package imports
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    # Add src directory to path so 'snge' package can be found
    src_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(src_dir))
    __package__ = "snge"

import time

import numpy as np
import matplotlib.pyplot as plt

from .models import NGEParameters, DimensionlessParameters
from .deterministic import solve_deterministic
from .dimensionless import solve_dimensionless, compute_steady_state
from .stochastic_fast import run_ensemble_gillespie_numba
from .langevin_fast import run_ensemble_euler_maruyama_numba, run_ensemble_snge_numba
from .analysis import (
    compute_yield_statistics,
    compare_distributions,
    compute_dimensionless_parameters,
    compute_snge_yield_statistics,
    analyze_noise_sensitivity,
    compare_cle_vs_snge,
)
from .plotting import (
    plot_ensemble_trajectories,
    plot_cv_evolution,
    plot_distribution_comparison,
    plot_summary_figure,
)


def run_snge_demo():
    """
    Demonstrate the SNGE model with phenomenological noise.

    Compares CLE (volume-dependent noise) vs SNGE (state-dependent noise).
    """
    print("\n" + "=" * 70)
    print("SNGE MODEL DEMONSTRATION")
    print("Phenomenological State-Dependent Noise")
    print("=" * 70)

    # Define NGE parameters
    params = NGEParameters(
        k1=2e-4,
        k2=0.15,
        k3=0.008,
        A0=0.01,
        B0=0.0,
        V=1e-12,  # 1 pL volume
        t_max=500.0
    )

    print(f"\nNGE Parameters:")
    print(f"  k1 = {params.k1:.2e} s^-1")
    print(f"  k2 = {params.k2:.2e} M^-1 s^-1")
    print(f"  k3 = {params.k3:.2e} s^-1")
    print(f"  A0 = {params.A0:.4f} M")
    print(f"  V  = {params.V:.2e} L")
    print(f"  N_A0 = {params.N_A0:.2e} molecules")

    # Convert to dimensionless parameters
    dim_params = compute_dimensionless_parameters(params)
    print(f"\nDimensionless Parameters:")
    print(f"  α (nucleation/etching) = {dim_params['alpha']:.4f}")
    print(f"  β (growth/etching)     = {dim_params['beta']:.4f}")
    print(f"  τ_max                  = {dim_params['tau_max']:.2f}")
    print(f"  Steady-state yield     = {dim_params['b_steady_state']*100:.1f}%")

    # Create SNGE parameters
    snge_params = DimensionlessParameters.from_nge_parameters(
        params,
        sigma0=0.05,   # Noise intensity
        epsilon=0.001,  # Regularization
        noise_model="inverse"
    )

    print(f"\nSNGE Noise Model:")
    print(f"  σ(b) = σ₀ / (b + ε)")
    print(f"  σ₀ = {snge_params.sigma0}")
    print(f"  ε  = {snge_params.epsilon}")

    # Analyze noise sensitivity
    sensitivity = analyze_noise_sensitivity(snge_params)
    print(f"\nNoise Sensitivity Analysis:")
    print(f"  σ(b=0.001) = {sensitivity['sigma_max']:.2f} (early stage)")
    print(f"  σ(b=0.5)   = {sensitivity['sigma_at_b_half']:.4f} (mid-stage)")
    print(f"  Amplification ratio: {sensitivity['amplification_ratio']:.1f}x")

    N_RUNS = 100

    # -------------------------------------------------------------------------
    # Run CLE simulations (volume-dependent noise)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(f"Running {N_RUNS} CLE simulations (volume-dependent noise)...")
    start_time = time.time()

    results_cle = run_ensemble_euler_maruyama_numba(params, n_runs=N_RUNS, dt=0.1)

    cle_time = time.time() - start_time
    print(f"CLE completed in {cle_time:.1f} seconds")

    stats_cle = compute_yield_statistics(results_cle)
    print(f"\nCLE Statistics:")
    print(f"  Mean yield: {stats_cle['mean'] * 100:.2f}%")
    print(f"  Std dev:    {stats_cle['std'] * 100:.2f}%")
    print(f"  CV:         {stats_cle['cv']:.1f}%")

    # -------------------------------------------------------------------------
    # Run SNGE simulations (state-dependent noise)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(f"Running {N_RUNS} SNGE simulations (state-dependent noise)...")
    start_time = time.time()

    results_snge = run_ensemble_snge_numba(snge_params, n_runs=N_RUNS, dtau=0.01)

    snge_time = time.time() - start_time
    print(f"SNGE completed in {snge_time:.1f} seconds")

    stats_snge = compute_snge_yield_statistics(results_snge)
    print(f"\nSNGE Statistics:")
    print(f"  Mean yield: {stats_snge['mean'] * 100:.2f}%")
    print(f"  Std dev:    {stats_snge['std'] * 100:.2f}%")
    print(f"  CV:         {stats_snge['cv']:.1f}%")

    # -------------------------------------------------------------------------
    # Compare methods
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("KEY COMPARISON: CLE vs SNGE")
    print("-" * 40)

    comparison = compare_cle_vs_snge(results_cle, results_snge, params)

    print(f"\n{'Metric':<20} {'CLE':<15} {'SNGE':<15}")
    print("-" * 50)
    print(f"{'Mean yield':<20} {comparison['cle_mean']*100:>12.2f}% {comparison['snge_mean']*100:>12.2f}%")
    print(f"{'Std deviation':<20} {comparison['cle_std']*100:>12.2f}% {comparison['snge_std']*100:>12.2f}%")
    print(f"{'CV':<20} {comparison['cle_cv']:>12.1f}% {comparison['snge_cv']:>12.1f}%")

    print(f"\nStatistical Tests:")
    print(f"  KS test p-value: {comparison['ks_pvalue']:.4f}")
    print(f"  Same distribution: {comparison['same_distribution']}")
    print(f"  Variances equal: {comparison['variances_equal']}")

    print(f"\nKey Insight:")
    print(f"  CLE noise scales with 1/√V (volume-dependent)")
    print(f"  SNGE noise σ(b) = σ₀/(b+ε) (state-dependent, captures critical period)")

    return results_cle, results_snge, params, snge_params


def main():
    """
    Main function demonstrating the stochastic simulation workflow.
    """
    print("=" * 70)
    print("STOCHASTIC NGE SIMULATIONS")
    print("=" * 70)

    # Define parameters (REPLACE WITH YOUR FITTED VALUES)
    params = NGEParameters(
        k1=1e-6,
        k2=0.5,
        k3=0.01,
        A0=0.01,
        B0=0.0,
        V=1e-12,  # 1000x larger → ~600,000 molecules
        t_max=1000
    )

    print(f"\nParameters:")
    print(f"  k1 = {params.k1:.2e} s^-1")
    print(f"  k2 = {params.k2:.2e} M^-1 s^-1")
    print(f"  k3 = {params.k3:.2e} s^-1")
    print(f"  A0 = {params.A0:.4f} M")
    print(f"  V  = {params.V:.2e} L")
    print(f"  N_A0 = {params.N_A0:.2e} molecules")

    # Deterministic solution for reference
    print("\n" + "-" * 40)
    print("Computing deterministic solution...")
    t_det = np.linspace(0, params.t_max, 500)
    A_det, B_det = solve_deterministic(params, t_det)
    det_yield = B_det[-1] / params.A0
    print(f"Deterministic final yield: {det_yield * 100:.2f}%")

    # Number of simulations
    N_RUNS = 10

    # -------------------------------------------------------------------------
    # Gillespie SSA
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(f"Running {N_RUNS} Gillespie SSA simulations...")
    start_time = time.time()

    results_gillespie = run_ensemble_gillespie_numba(params, n_runs=N_RUNS, dt_record=1.0)

    gillespie_time = time.time() - start_time
    print(f"Gillespie completed in {gillespie_time:.1f} seconds")

    stats_g = compute_yield_statistics(results_gillespie)
    print(f"\nGillespie Statistics:")
    print(f"  Mean yield: {stats_g['mean'] * 100:.2f}%")
    print(f"  Std dev:    {stats_g['std'] * 100:.2f}%")
    print(f"  CV:         {stats_g['cv']:.1f}%")
    print(f"  Skewness:   {stats_g['skewness']:.3f}")
    print(f"  Range:      [{stats_g['min'] * 100:.2f}%, {stats_g['max'] * 100:.2f}%]")

    # -------------------------------------------------------------------------
    # Euler-Maruyama CLE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print(f"Running {N_RUNS} Euler-Maruyama CLE simulations...")
    start_time = time.time()

    results_em = run_ensemble_euler_maruyama_numba(params, n_runs=N_RUNS, dt=0.1)

    em_time = time.time() - start_time
    print(f"Euler-Maruyama completed in {em_time:.1f} seconds")

    stats_em = compute_yield_statistics(results_em)
    print(f"\nEuler-Maruyama Statistics:")
    print(f"  Mean yield: {stats_em['mean'] * 100:.2f}%")
    print(f"  Std dev:    {stats_em['std'] * 100:.2f}%")
    print(f"  CV:         {stats_em['cv']:.1f}%")
    print(f"  Skewness:   {stats_em['skewness']:.3f}")
    print(f"  Range:      [{stats_em['min'] * 100:.2f}%, {stats_em['max'] * 100:.2f}%]")

    # -------------------------------------------------------------------------
    # Compare methods
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Comparing methods...")

    comparison = compare_distributions(
        stats_g['yields'], stats_em['yields'],
        'Gillespie', 'Euler-Maruyama'
    )

    print(f"\nKolmogorov-Smirnov test:")
    print(f"  Statistic: {comparison['ks_statistic']:.4f}")
    print(f"  p-value:   {comparison['ks_pvalue']:.4f}")
    print(f"  Same distribution: {comparison['ks_same_distribution']}")

    print(f"\nVariance comparison (Levene's test):")
    print(f"  p-value:   {comparison['levene_pvalue']:.4f}")
    print(f"  Equal variances: {comparison['variances_equal']}")

    # -------------------------------------------------------------------------
    # SNGE Demonstration
    # -------------------------------------------------------------------------
    results_cle, results_snge, _, snge_params = run_snge_demo()

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Generating figures...")

    # Figure 1: Gillespie trajectories and distribution
    fig1 = plot_ensemble_trajectories(results_gillespie, params, n_show=50)
    fig1.savefig('figure_gillespie_ensemble.png', dpi=150, bbox_inches='tight')
    print("  Saved: figure_gillespie_ensemble.png")

    # Figure 2: CV evolution (critical period)
    fig2 = plot_cv_evolution(results_gillespie, params)
    fig2.savefig('figure_cv_evolution.png', dpi=150, bbox_inches='tight')
    print("  Saved: figure_cv_evolution.png")

    # Figure 3: Method comparison
    fig3 = plot_distribution_comparison(results_gillespie, results_em)
    fig3.savefig('figure_method_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: figure_method_comparison.png")

    # Figure 4: Summary figure
    fig4 = plot_summary_figure(results_gillespie, params)
    fig4.savefig('figure_summary.png', dpi=150, bbox_inches='tight')
    print("  Saved: figure_summary.png")

    # Figure 5: SNGE vs CLE comparison
    fig5, axes = plt.subplots(1, 3, figsize=(15, 5))

    # SNGE trajectories
    ax1 = axes[0]
    for i, res in enumerate(results_snge[:20]):
        ax1.plot(res.tau, res.b * 100, alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Dimensionless time (τ)')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title('SNGE Trajectories')
    ax1.grid(True, alpha=0.3)

    # Noise sensitivity
    ax2 = axes[1]
    sensitivity = analyze_noise_sensitivity(snge_params)
    ax2.semilogy(sensitivity['b'] * 100, sensitivity['sigma'])
    ax2.set_xlabel('Yield (%)')
    ax2.set_ylabel('Noise intensity σ(b)')
    ax2.set_title('State-Dependent Noise\nσ(b) = σ₀/(b+ε)')
    ax2.grid(True, alpha=0.3)

    # Distribution comparison
    ax3 = axes[2]
    cle_yields = [r.final_yield * 100 for r in results_cle]
    snge_yields = [r.final_yield * 100 for r in results_snge]
    ax3.hist(cle_yields, bins=20, alpha=0.5, label='CLE', density=True)
    ax3.hist(snge_yields, bins=20, alpha=0.5, label='SNGE', density=True)
    ax3.set_xlabel('Final Yield (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('CLE vs SNGE Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig5.savefig('figure_snge_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: figure_snge_comparison.png")

    plt.show()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return results_gillespie, results_em, params


if __name__ == "__main__":
    results_g, results_em, params = main()
