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

```
FollicleModeling/
├── Models.jl          # Turing model definitions: total_model, pausing_model, queuing_model
│                      # and their sampling functions; transition matrix constructors
├── Utils.jl           # Helper functions: data loading, credible intervals, multinomial
│                      # approximation, empirical statistics
├── PlotUtils.jl       # Plotting helpers: posterior credible ribbons, calibration plots
├── FaddyModel.jl      # Full Faddy analysis: fit, predictive checks, parameter posteriors
├── PausingModel.jl    # Full pausing-model analysis
├── QueuingModel.jl    # Full queuing-model analysis
├── InitialCondition.jl # Focused analysis of 2-month data only (prior/posterior for IC)
├── PriorPredictive.jl # Minimal prior predictive example (old)
├── data/              # Raw CSV/Excel follicle count data
├── models/            # Saved MCMC chains (.jld2)
├── plots/             # Output figures
└── docs/              # notes.tex (full math) + References.bib
```

## Running the analyses

Each analysis script is self-contained and can be run from the project root. They all call `include("Models.jl")`, `include("Utils.jl")`, and `include("PlotUtils.jl")` at the top, so no package installation step is needed beyond the standard `Pkg.instantiate()`.

```julia
# Install dependencies (first time only)
using Pkg; Pkg.instantiate()
```

## Key functions

| Function | File | Description |
|---|---|---|
| `extract_data()` | Utils.jl | Load CSV → 5 count matrices (one per timepoint) |
| `finite_transition_matrix(W, t)` | Models.jl | Compute exp(W·(t−2)) with numerical safeguards |
| `probability_flow(π₀, W, times)` | Models.jl | Trajectory: [θ(t₁), θ(t₂), …] |
| `confidence_intervals(f, t)` | Utils.jl | Posterior predictive quantiles for any scalar function |
| `chain_stats_sample(sample_fun, ...)` | Utils.jl | Replicated posterior predictive mean/cov quantiles |
| `plot_empirical_stats(...)` | PlotUtils.jl | Calibration scatter: empirical vs model mean/covariance |
| `params_logn(mean, var)` | Utils.jl | LogNormal(μ,σ) parameters matching given mean and variance |

## Dependencies

Core: `Turing`, `ExponentialUtilities`, `AugmentedGPLikelihoods` (for `NegativeMultinomial`), `LinearAlgebra`, `NaNMath`

Data/IO: `DataFrames`, `CSV`, `JLD2`

Plotting: `Plots`, `StatsPlots`, `Measures`

See `Project.toml` for pinned versions.

## References

- Faddy MJ, Gosden RG (1996). A model conforming the decline in follicle numbers to the age of menopause in women.
- Full bibliography: `docs/References.bib`
