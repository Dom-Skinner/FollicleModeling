# FollicleModeling

Bayesian compartmental modeling of ovarian follicle dynamics in mice. The project fits competing Markov models to longitudinal follicle count data (2, 4, 6, 9, 12 months) and uses posterior predictive checks to compare model adequacy. Full mathematical derivations are in `docs/notes.tex`.

## Biological context

Mouse ovarian follicles progress through distinct developmental stages — Primordial → Primary → Secondary — before either dying or maturing further. Each model variant represents a different biological hypothesis about what governs that progression: simple exponential residence times (Faddy), the possibility of paused/dormant states (Pausing), or non-exponential (Erlang-distributed) residence times arising from hidden sub-stages (Queuing/Erlang).

## Statistical approach

All models share the same likelihood: a **Negative Multinomial** observation model (equivalent to a Gamma-Poisson mixture with Multinomial splitting). This accounts for overdispersion in count data while preserving the conjugate structure under linear Markov dynamics. The transition probabilities at each timepoint come from the matrix exponential of the generator matrix W:

```
θ(t) = exp(W · (t − 2)) · π₀
```

where t = 2 months is the initial reference timepoint and π₀ is the initial state distribution. Parameters are inferred with NUTS (Turing.jl).

## Models

### Faddy model (`FaddyModel.jl`)
The baseline model from Faddy et al. (1976): four states (Primordial, Primary, Secondary, Dead/Lost), exponential residence times, 4 rate parameters. Analysis proceeds in two stages: first fitting only the initial conditions from 2-month data with rates fixed at literature values, then jointly fitting all parameters to the 4–12 month data.

Parameters: `w1` (mean primordial lifetime), `w2` (mean primary lifetime), `w3` (mean secondary lifetime), `θ12` (probability of reaching primary vs dying directly).

### Pausing model (`PausingModel.jl`)
Extends Faddy to include hidden paused states for primary and secondary follicles. The paused state represents first wave follicles thar are in a dormant/paused sub-state and will later resume normal dynamics by entering the regular primary and secondary sub-state. Six states total; observations aggregate the active and paused sub-states for each visible stage.

Additional parameters: `θ34` (probability of reaching secondary from primary), `θ6`, `θ7` (pause/resume rates for primary and secondary respectively).

### Queuing (Erlang) model (`QueuingModel.jl`)
Models non-exponential residence times by expanding each biological stage into a chain of hidden sub-stages, producing Erlang-distributed waiting times. This captures biological "maturation clocks" where a follicle must complete multiple sequential steps before progressing. See `docs/notes.tex` §Erlang for the derivation.

Additional parameters: tbd.

## File structure

Library code lives in `src/`; everything at the top level is a runnable script.

```
FollicleModeling/
├── src/
│   ├── Models.jl           # Turing model definitions: total_model and its sampling
│   │                       # functions; transition matrix constructors; model builders
│   ├── NegativeMultinomial.jl # Vendored NegativeMultinomial distribution (from
│   │                       # AugmentedGPLikelihoods, MIT); included by Models.jl
│   ├── Utils.jl            # Helper functions: data loading, credible intervals,
│   │                       # multinomial approximation, empirical statistics
│   ├── PlotUtils.jl        # Plotting helpers: posterior credible ribbons, corner plots,
│   │                       # calibration plots
│   ├── ModelConfigs.jl     # Model registry: topology + priors for every candidate model,
│   │                       # the single source of truth used by scripts and CV alike
│   └── CrossValidation.jl  # Pointwise log-likelihood, elpd_cv, fold construction
├── FaddyModel.jl           # Full Faddy analysis: fit, predictive checks, parameter posteriors
├── PausingModel.jl         # Full pausing-model analysis
├── QueuingModel.jl         # Full queuing (Erlang) model analysis
├── TimeDepModel.jl         # Faddy with a time-dependent rate μ(t) (ODE propagator)
├── QueuingTimeDepModel.jl  # Queuing model with time-dependent μ(t)
├── PausedTimeDepModel.jl   # Pausing model with time-dependent μ(t)
├── InitialCondition.jl     # Focused analysis of 2-month data only (prior/posterior for IC)
├── PriorPredictive.jl      # Minimal prior predictive example (old)
├── RunLOO.jl               # Serial leave-one-out cross-validation: fit folds, then plot
├── loo/                    # Parallel LOO via Snakemake: fit_fold.jl, combine.jl, plot_loo.jl
│                           # (driven by the Snakefile; see docs/LOO_instructions.md)
├── test/runtests.jl        # Unit tests for the model and utility code
├── data/                   # Raw CSV/Excel follicle count data
├── models/                 # Saved MCMC chains (.jld2)
├── plots/                  # Output figures
└── docs/                   # notes.tex (full math) + References.bib
```

## Running the analyses

Each analysis script is self-contained and should be run from the project root (data and plot paths are relative to it). They pull in the library code with `include("src/Models.jl")`, `include("src/Utils.jl")`, and so on, so no package installation step is needed beyond the standard `Pkg.instantiate()`.

```julia
# Install dependencies (first time only)
using Pkg; Pkg.instantiate()
```

## Key functions

| Function | File | Description |
|---|---|---|
| `extract_data()` | src/Utils.jl | Load CSV → 5 count matrices (one per timepoint) |
| `finite_transition_matrix(W, t)` | src/Models.jl | Compute exp(W·(t−2)) with numerical safeguards |
| `probability_flow(π₀, W, times)` | src/Models.jl | Trajectory: [θ(t₁), θ(t₂), …] |
| `confidence_intervals(f, t)` | src/Utils.jl | Posterior predictive quantiles for any scalar function |
| `chain_stats_sample(sample_fun, ...)` | src/Utils.jl | Replicated posterior predictive mean/cov quantiles |
| `plot_empirical_stats(...)` | src/PlotUtils.jl | Calibration scatter: empirical vs model mean/covariance |
| `params_logn(mean, var)` | src/Utils.jl | LogNormal(μ,σ) parameters matching given mean and variance |
| `model_config(name)` | src/ModelConfigs.jl | Fetch one model's topology + priors from the registry |
| `elpd_cv(...)` | src/CrossValidation.jl | Cross-validated expected log predictive density |

## Dependencies

Core: `Turing`, `ExponentialUtilities`, `LinearAlgebra`, `NaNMath`, `OrdinaryDiffEq` (time-dependent models). `NegativeMultinomial` is vendored in `src/NegativeMultinomial.jl` (from AugmentedGPLikelihoods, MIT) and needs only `Distributions` and `SpecialFunctions`.

Data/IO: `DataFrames`, `CSV`, `JLD2`

Plotting: `Plots`, `StatsPlots`, `Measures`

See `Project.toml` for pinned versions.

## References

- Faddy MJ, Gosden RG (1996). A model conforming the decline in follicle numbers to the age of menopause in women.
- Full bibliography: `docs/References.bib`
