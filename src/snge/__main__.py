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

from .models import NGEParameters
from .deterministic import solve_deterministic
from .stochastic_fast import run_ensemble_gillespie_numba
from .langevin_fast import run_ensemble_euler_maruyama_numba
from .analysis import compute_yield_statistics, compare_distributions
from .plotting import (
    plot_ensemble_trajectories,
    plot_cv_evolution,
    plot_distribution_comparison,
    plot_summary_figure,
)


def main():
    """
    Main function demonstrating the stochastic simulation workflow.
    """
    print("=" * 70)
    print("STOCHASTIC NGE SIMULATIONS")
    print("=" * 70)

    # Define parameters (REPLACE WITH YOUR FITTED VALUES)
    params = NGEParameters(
        k1=1e-6,  # Nucleation rate (s^-1) - REPLACE WITH FITTED VALUE
        k2=0.5,  # Growth rate (M^-1 s^-1) - REPLACE WITH FITTED VALUE
        k3=0.01,  # Etching rate (s^-1) - REPLACE WITH FITTED VALUE
        A0=0.01,  # Initial precursor concentration (M)
        B0=0.0,  # Initial graphene concentration (M)
        V=1e-15,  # System volume (L) - ~600 molecules, shows clear variation
        t_max=600  # Simulation time (s)
    )
    # INITIAL PARAMETERS
    # params = NGEParameters(
    #     k1=1e-5,  # Nucleation rate (s^-1) - REPLACE WITH FITTED VALUE
    #     k2=0.01,  # Growth rate (M^-1 s^-1) - REPLACE WITH FITTED VALUE
    #     k3=0.005,  # Etching rate (s^-1) - REPLACE WITH FITTED VALUE
    #     A0=0.01,  # Initial precursor concentration (M)
    #     B0=0.0,  # Initial graphene concentration (M)
    #     V=1e-16,  # System volume (L) - ~600 molecules, shows clear variation
    #     t_max=600  # Simulation time (s)
    # )

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
    N_RUNS = 10000

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

    plt.show()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return results_gillespie, results_em, params


if __name__ == "__main__":
    results_g, results_em, params = main()
