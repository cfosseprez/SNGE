# SNGE - Stochastic Nucleation-Growth-Etching Model

Stochastic simulations of the Nucleation-Growth-Etching (NGE) model for graphene synthesis kinetics.

## The NGE Framework

### Key Insight: CV Emerges from Physics

Unlike phenomenological models that fit noise parameters to variance, the NGE framework **predicts CV from physical parameters alone**:

1. **Fit rate constants** (k₁, k₂, k₃) to mean kinetics only
2. **Run Gillespie simulations** with physical N = [A]₀ × V × Nₐ
3. **CV emerges naturally** - no fitting to variance data

This is a **TRUE PREDICTION**: the model either matches experimental CV or it doesn't.

> "The CV is not imposed or fitted—it emerges from running stochastic
> simulations with physical parameters. The model either predicts the
> experimental CV or it doesn't."

### Seeding Validation

The framework predicts seeding reduces CV by bypassing the critical period. This prediction requires **NO new parameters** - only changing B₀ > 0.

### Dimensionless Parameters

- α = k₁/k₃ (nucleation-to-etching ratio)
- β = k₂[A]₀/k₃ (growth-to-etching ratio)

For plasma graphene synthesis: α ~ 0.01-0.1, β ~ 0.5-2 (competitive regime)

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
    compute_yield_statistics,
    predict_cv_from_gillespie,
    validate_cv_prediction,
)

# Define parameters (fitted from mean kinetics)
params = NGEParameters(
    k1=1e-4,    # Nucleation rate (s⁻¹)
    k2=0.1,     # Growth rate (M⁻¹s⁻¹)
    k3=0.01,    # Etching rate (s⁻¹)
    A0=0.01,    # Initial precursor concentration (M)
    B0=0.0,     # Initial graphene concentration (M)
    V=1e-15,    # System volume (L)
    t_max=600   # Simulation time (s)
)

# Predict CV from Gillespie SSA (paper's methodology)
prediction = predict_cv_from_gillespie(params, n_runs=1000)
print(f"Predicted CV: {prediction['predicted_cv']:.1f}%")
print(f"Molecule count: {prediction['n_molecules']}")

# Validate against experimental CV
experimental_cv = 25.0  # Example experimental CV
validation = validate_cv_prediction(params, experimental_cv=experimental_cv)
print(validation['interpretation'])
```

## Workflow: The Paper's Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXPERIMENTAL DATA                               │
│   Multiple runs of time-resolved yield measurements                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: FIT DETERMINISTIC NGE                    │
│   Fit d[B]/dt = k₁[A] + k₂[A][B] - k₃[B] to MEAN kinetics          │
│   → Get fitted k₁, k₂, k₃                                           │
│   NOTE: NO parameters fitted to variance!                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 2: GILLESPIE SSA SIMULATIONS                │
│   Use fitted k₁, k₂, k₃ with physical N = [A]₀ × V × Nₐ            │
│   Run 1000+ Monte Carlo simulations                                 │
│   → CV emerges naturally from Poisson statistics                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: VALIDATE                                 │
│   Compare experimental CV vs predicted CV                           │
│   If they match → Stochastic framework validated!                   │
│   The model either predicts the experimental CV or it doesn't.      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 4: SEEDING PREDICTION                       │
│   Change B₀ > 0 (NO other parameter changes!)                       │
│   Predict reduced CV due to bypassing critical period               │
│   This is a TRUE prediction with no free parameters                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Simulation Methods

| Method | Use Case | Speed |
|--------|----------|-------|
| `run_ensemble_gillespie_numba` | **CV prediction** (exact stochastic) | Fast |
| `run_ensemble_euler_maruyama_numba` | Large systems (CLE approximation) | Very fast |
| `run_ensemble_euler_maruyama_gpu` | Large ensembles (10,000+ runs) | Fastest |

### Choosing Volume (V)

The volume parameter determines the molecule count N = [A]₀ × V × Nₐ:

| Volume (L) | Molecules | Gillespie | Euler-Maruyama | Relative Noise |
|------------|-----------|-----------|----------------|----------------|
| `1e-6` | ~6×10¹⁵ | Too slow | Fast | Very low |
| `1e-12` | ~6×10⁹ | Too slow | Fast | Low |
| `1e-15` | ~6×10³ | Fast | Fast | High |

**Rule of thumb:**
- For realistic volumes: Use Euler-Maruyama
- For CV prediction (small systems): Use Gillespie SSA

## Seeding Analysis

```python
from snge import NGEParameters, compare_seeded_vs_unseeded

# Parameters (from mean kinetics fitting)
params = NGEParameters(k1=1e-4, k2=0.1, k3=0.01, A0=0.01, B0=0.0, V=1e-15, t_max=600)

# Test seeding levels
seed_levels = [1e-4, 5e-4, 1e-3]  # M

# Compare seeded vs unseeded (same k1, k2, k3 - only B0 changes)
comparison = compare_seeded_vs_unseeded(params, seed_levels=seed_levels, n_runs=1000)

print(f"Unseeded CV: {comparison['unseeded_cv']:.1f}%")
for level, cv in comparison['seeded_cvs'].items():
    reduction = comparison['cv_reductions'][level]
    print(f"Seeded ({level:.0e} M) CV: {cv:.1f}% ({reduction:.1f}x reduction)")
```

## Parameter Fitting

```python
from snge.fitting import load_data_from_csv, fit_nge_to_mean, plot_fit_results

# Load experimental data (CSV with columns: time, run1, run2, ...)
data = load_data_from_csv('your_data.csv', A0=0.01)

# Fit NGE model to MEAN kinetics (NOT variance!)
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

## API Reference

### Data Structures

- `NGEParameters` - Model parameters (k₁, k₂, k₃, A₀, B₀, V, t_max)
- `SimulationResult` - Single simulation result (times, concentrations, yield)
- `DimensionlessParameters` - Dimensionless parameters (α, β, τ_max)
- `ExperimentalData` - Experimental data container
- `FitResult` - Fitting results with uncertainties

### CV Prediction Functions (Paper's Methodology)

```python
# Predict CV from physical parameters (no fitting to variance)
predict_cv_from_gillespie(params, n_runs=10000)

# Validate prediction against experimental CV
validate_cv_prediction(params, experimental_cv=25.0)

# Compare seeded vs unseeded CV
compare_seeded_vs_unseeded(params, seed_levels=[1e-4, 1e-3])

# Analyze critical period from Gillespie results
compute_critical_period_gillespie(results, threshold=0.1)

# Analyze trajectory divergence
analyze_trajectory_divergence(results)
```

### Simulation Functions

```python
# Single trajectory (Gillespie SSA - exact)
gillespie_ssa(params)
gillespie_ssa_numba(params)

# Single trajectory (CLE - approximation)
euler_maruyama_cle(params)
euler_maruyama_numba(params)

# Ensemble (parallel) - RECOMMENDED
run_ensemble_gillespie_numba(params, n_runs=1000)
run_ensemble_euler_maruyama_numba(params, n_runs=1000)
run_ensemble_euler_maruyama_gpu(params, n_runs=1000)  # Requires CuPy
```

### Analysis Functions

```python
compute_yield_statistics(results)      # Mean, std, CV, skewness, etc.
compare_distributions(yields1, yields2) # Statistical comparison
compute_cv_over_time(results, t_points) # CV evolution
compute_dimensionless_parameters(params) # α, β, τ_max
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
| Numba Gillespie | 1-10s | **Recommended for CV prediction** |
| Pure Python E-M | 30-60s | Don't use |
| Numba E-M | 1-5s | Large systems only |
| GPU E-M | <1s | For 10,000+ runs |

## Verification

After installation, verify the implementation:

```bash
# Run tests
pytest

# Verify CV emergence (no fitting)
python -c "
from snge import NGEParameters, run_ensemble_gillespie_numba, compute_yield_statistics

params = NGEParameters(k1=1e-4, k2=0.1, k3=0.01, A0=0.01, V=1e-15, t_max=600, B0=0.0)
results = run_ensemble_gillespie_numba(params, n_runs=1000)
stats = compute_yield_statistics(results)
print(f'CV emerges naturally: {stats[\"cv\"]:.1f}%')
"

# Verify seeding reduces CV
python -c "
from snge import NGEParameters, run_ensemble_gillespie_numba, compute_yield_statistics

# Unseeded
params_unseeded = NGEParameters(k1=1e-4, k2=0.1, k3=0.01, A0=0.01, V=1e-15, t_max=600, B0=0.0)
results_u = run_ensemble_gillespie_numba(params_unseeded, n_runs=1000)
cv_unseeded = compute_yield_statistics(results_u)['cv']

# Seeded
params_seeded = NGEParameters(k1=1e-4, k2=0.1, k3=0.01, A0=0.01, V=1e-15, t_max=600, B0=0.001)
results_s = run_ensemble_gillespie_numba(params_seeded, n_runs=1000)
cv_seeded = compute_yield_statistics(results_s)['cv']

print(f'Unseeded CV: {cv_unseeded:.1f}%')
print(f'Seeded CV: {cv_seeded:.1f}%')
print(f'CV reduction: {cv_unseeded/cv_seeded:.1f}x')
"
```

## License

GPL-3.0
