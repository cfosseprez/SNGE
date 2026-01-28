# SNGE - Stochastic Nucleation-Growth-Etching Model

Stochastic simulations of the Nucleation-Growth-Etching (NGE) model for graphene synthesis kinetics.

## Installation

```bash
pip install -e .
```

For GPU acceleration (optional):
```bash
pip install cupy-cuda12x  # Adjust for your CUDA version
```

## The NGE Model

The NGE reaction scheme describes graphene synthesis:

```
A → B           (nucleation, rate k₁)
A + B → 2B      (growth, rate k₂)
B → C           (etching, rate k₃)
```

Where: A = precursor, B = graphene, C = etched products

## Quick Start

```python
from snge import (
    NGEParameters,
    run_ensemble_gillespie_numba,
    run_ensemble_euler_maruyama_numba,
    compute_yield_statistics,
)

# Define parameters
params = NGEParameters(
    k1=1e-4,    # Nucleation rate (s⁻¹)
    k2=0.1,     # Growth rate (M⁻¹s⁻¹)
    k3=0.01,    # Etching rate (s⁻¹)
    A0=0.01,    # Initial precursor concentration (M)
    B0=0.0,     # Initial graphene concentration (M)
    V=1e-15,    # System volume (L) - see note below
    t_max=600   # Simulation time (s)
)

# Run 1000 stochastic simulations (Numba-accelerated)
results = run_ensemble_euler_maruyama_numba(params, n_runs=1000)

# Analyze results
stats = compute_yield_statistics(results)
print(f"Mean yield: {stats['mean']*100:.2f}%")
print(f"CV: {stats['cv']:.1f}%")
```

## Simulation Methods

| Method | Use Case | Speed |
|--------|----------|-------|
| `run_ensemble_gillespie_numba` | Exact stochastic, small systems (< 10⁶ molecules) | Fast |
| `run_ensemble_euler_maruyama_numba` | Approximate, any system size | Very fast |
| `run_ensemble_euler_maruyama_gpu` | Large ensembles (10,000+ runs) | Fastest |

### Choosing Volume (V)

The volume parameter controls the number of molecules and thus noise magnitude:

| Volume (L) | Molecules | Gillespie | Euler-Maruyama | Relative Noise |
|------------|-----------|-----------|----------------|----------------|
| `1e-6` | ~6×10¹⁵ | Too slow | Fast | Very low |
| `1e-12` | ~6×10⁹ | Too slow | Fast | Low |
| `1e-15` | ~6×10³ | Fast | Fast | High |

**Rule of thumb:**
- For realistic volumes: Use **Euler-Maruyama only**
- For method validation: Use `V=1e-15` (both methods comparable)

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPERIMENTAL DATA                               │
│   Multiple runs of time-resolved yield measurements                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: FIT DETERMINISTIC NGE                    │
│   Fit d[B]/dt = k₁[A] + k₂[A][B] - k₃[B] to mean kinetics          │
│   → Get fitted k₁, k₂, k₃                                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 2: STOCHASTIC SIMULATIONS                   │
│   Use fitted k₁, k₂, k₃ (NO new free parameters!)                   │
│   Run 1000+ Monte Carlo simulations                                 │
│   → Get predicted CV, distribution shape                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: VALIDATE                                 │
│   Compare experimental CV vs predicted CV                           │
│   If they match → Stochastic framework validated!                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Parameter Fitting

```python
from snge.fitting import load_data_from_csv, fit_nge_to_mean, plot_fit_results

# Load experimental data (CSV with columns: time, run1, run2, ...)
data = load_data_from_csv('your_data.csv', A0=0.01)

# Fit NGE model to mean kinetics
fit_result = fit_nge_to_mean(data, method='both')
print(fit_result)

# Visualize results
fig = plot_fit_results(data, fit_result)
fig.savefig('fit_results.png')
```

### Expected Data Format

```csv
time,run1,run2,run3,...,runN
0,0.00,0.00,0.00,...,0.00
60,0.02,0.015,0.025,...,0.018
120,0.12,0.09,0.15,...,0.11
180,0.30,0.25,0.35,...,0.28
...
```

Yields can be fractions (0-1) or percentages (0-100, auto-detected).

## API Reference

### Data Structures

- `NGEParameters` - Model parameters (k₁, k₂, k₃, A₀, B₀, V, t_max)
- `SimulationResult` - Single simulation result (times, concentrations, yield)
- `ExperimentalData` - Experimental data container (from fitting module)
- `FitResult` - Fitting results with uncertainties

### Simulation Functions

```python
# Single trajectory
gillespie_ssa(params)              # Exact stochastic
gillespie_ssa_numba(params)        # Numba-accelerated
euler_maruyama_cle(params)         # Chemical Langevin Equation
euler_maruyama_numba(params)       # Numba-accelerated

# Ensemble (parallel)
run_ensemble_gillespie_numba(params, n_runs=1000)
run_ensemble_euler_maruyama_numba(params, n_runs=1000)
run_ensemble_euler_maruyama_gpu(params, n_runs=1000)  # Requires CuPy
```

### Analysis Functions

```python
compute_yield_statistics(results)      # Mean, std, CV, skewness, etc.
compare_distributions(yields1, yields2) # Statistical comparison
compute_cv_over_time(results, t_points) # CV evolution
```

### Plotting Functions

```python
plot_ensemble_trajectories(results, params)
plot_cv_evolution(results, params)
plot_distribution_comparison(results_g, results_em)
plot_summary_figure(results, params)
```

## CLI Usage

```bash
# Run demo simulation
python -m snge

# Or use the installed command
snge
```

## Performance

First run includes JIT compilation (~2-5s). Subsequent runs:

| Method | 1000 runs | Notes |
|--------|-----------|-------|
| Pure Python Gillespie | Minutes | Don't use |
| Numba Gillespie | 1-10s | For small V only |
| Pure Python E-M | 30-60s | Don't use |
| Numba E-M | 1-5s | Recommended |
| GPU E-M | <1s | For 10,000+ runs |

## License

GPL-3.0
