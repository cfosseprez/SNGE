"""
Ensemble simulation runners for the NGE model.
"""

from typing import List

from tqdm import tqdm

from .models import NGEParameters, SimulationResult
from .stochastic import gillespie_ssa_fast
from .langevin import euler_maruyama_cle_simple


def run_ensemble_gillespie(params: NGEParameters,
                           n_runs: int = 1000,
                           dt_record: float = 1.0,
                           show_progress: bool = True) -> List[SimulationResult]:
    """
    Run ensemble of Gillespie SSA simulations.

    Args:
        params: NGE parameters
        n_runs: Number of independent simulations
        dt_record: Recording interval
        show_progress: Show progress bar

    Returns:
        List of SimulationResult objects
    """
    results = []
    iterator = tqdm(range(n_runs), desc="Gillespie SSA") if show_progress else range(n_runs)

    for _ in iterator:
        result = gillespie_ssa_fast(params, dt_record=dt_record)
        results.append(result)

    return results


def run_ensemble_euler_maruyama(params: NGEParameters,
                                n_runs: int = 1000,
                                dt: float = 0.1,
                                show_progress: bool = True) -> List[SimulationResult]:
    """
    Run ensemble of Euler-Maruyama CLE simulations.

    Args:
        params: NGE parameters
        n_runs: Number of independent simulations
        dt: Integration time step
        show_progress: Show progress bar

    Returns:
        List of SimulationResult objects
    """
    results = []
    iterator = tqdm(range(n_runs), desc="Euler-Maruyama") if show_progress else range(n_runs)

    for i in iterator:
        result = euler_maruyama_cle_simple(params, dt=dt, seed=None)
        results.append(result)

    return results
