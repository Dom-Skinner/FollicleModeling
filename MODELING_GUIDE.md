# Modeling guide: math, code, and how to experiment

This guide pairs the math in `docs/notes.tex` with the code that implements it. **Part I** gives the big picture and is meant for people. **Part II** gives the full technical detail (conventions, parameter indices, gotchas) for anyone, human or agent, changing the code. `README.md` covers file layout and setup.

---

# Part I: The idea

## What the code does

We take whole-ovary follicle counts at 2, 4, 6, 9 and 12 months, each a vector `[primordial, primary, secondary]` per ovary. We ask which stochastic models of follicle dynamics fit them. Each model is a hypothesis about the dynamics:

| Model | Hypothesis |
|---|---|
| `Faddy` | One exponential compartment per stage (classic Faddy). |
| `Queuing` | Primary and secondary stages are chains of hidden substages, so residence times are Erlang (less variable than exponential). |
| `Paused` | Queuing, plus a hidden **paused** reservoir in primary and secondary. It holds only follicles present at 2 months (residual first wave), which drain into the active chain. |
| `*TimeDep` | Any of the above, with a primordial exit rate that changes smoothly with age. |

All six (`Faddy`, `FaddyTimeDep`, `Queuing`, `QueuingTimeDep`, `Paused`, `PausedTimeDep`) are defined in one place: `src/ModelConfigs.jl`.

## The modeling assumptions

1. **Follicles are independent.** Each follicle moves through a finite set of states as a continuous-time Markov chain with generator $W_\theta$ (possibly time dependent). Some states are hidden substages. One absorbing state stands for "no longer counted" (atresia, or growth past secondary).
2. **Ovaries differ in size.** Each ovary has its own latent intensity $\lambda_i \sim \mathrm{Gamma}$, which scales how many follicles it starts with. Integrating out $\lambda_i$ gives a closed-form **Negative Multinomial** likelihood: a negative-binomial total, split multinomially across stages. This captures the large ovary-to-ovary spread, which a Poisson model cannot.
3. **The reference time is 2 months.** The 2-month data pins down the initial state distribution and the overdispersion. The 4 to 12 month data then constrains the dynamics.

## Every answer is a posterior, so report credible intervals

The models are fit by Bayesian inference (NUTS in Turing.jl). A fit therefore gives a **joint posterior distribution** over every parameter (initial size, overdispersion, initial state mix, residence times, survival probabilities, pause durations, early and late primordial rates), not one best-fit value. A saved chain (`models/<Model>_chain.jld2`) is a set of draws from that joint distribution.

**The consequence for any experiment:** any quantity you compute from the model comes with an uncertainty for free. Examples are a predicted count at 18 months, the fraction of secondaries that are paused, the time to deplete the reserve, or the number of follicles that would leave secondary under a hypothetical intervention. To get the uncertainty:

1. Write a function that takes **one posterior draw** and returns your quantity.
2. Evaluate it on many draws.
3. Report quantiles, for example the median and 95% credible interval (2.5%, 97.5%).

```julia
include("src/ModelConfigs.jl"); include("src/CrossValidation.jl")   # chain_draws lives here
using JLD2, Statistics
cfg   = model_config("PausedTimeDep")
chain = load("models/PausedTimeDep_chain.jld2", "chain")

# Example: expected number of primordials in an average ovary at 18 months (an extrapolation)
function my_quantity(draw)
    μN = extract_array(draw, "ic")[1]
    a  = probability_flow(extract_array(draw, "π_vals"),
                          cfg.transition_fcn(extract_array(draw, "rate_params")), [18.0])[1]
    return μN * (cfg.coarse_grain * a[1:end-1])[1]
end
vals = [my_quantity(d) for d in chain_draws(chain)]
quantile(vals, [0.025, 0.5, 0.975])
```

Things to keep in mind:

- **Do not collapse to a point estimate early.** Push every draw through the full calculation and summarize at the end. The value computed at the posterior mean is not the posterior mean of the value, especially for nonlinear quantities such as residence times or ratios.
- **Separate the two kinds of uncertainty.**
  - *Parameter uncertainty* asks "what is the expected count?". Compute deterministically per draw, as in the example above.
  - *Posterior predictive* asks "what would a new ovary look like?". It also includes ovary-to-ovary and stochastic noise. Use `make_sample_fun` or `simulate_ovary`, which draw random counts per posterior draw. Predictive intervals are much wider. Use them when comparing to individual ovaries.
- **Wide intervals are a result.** Several parameters are only weakly identified (see the corner plots: $\mu_2$, $\mu_3$ and the primordial survival trade off). A wide credible interval on a prediction tells you the data does not constrain it, which often points to the next experiment worth doing. The `LabelingExperiment.jl` script was built in this spirit.
- **Compare models on held-out data, not on posterior fit.** Use `src/CrossValidation.jl` (leave-one-ovary-out and leave-one-age-out ELPD). It also reports uncertainty: a standard error on each ΔELPD.

## Typical workflow

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()          # once
# Fit a model + make all diagnostic plots (writes plots/, models/<Model>_chain.jld2):
julia --project=. --threads=2 PausedTimeDepModel.jl
# Simulate individual ovaries from posterior draws and animate them:
julia --project=. SimulateOvary.jl PausedTimeDep 3
julia --project=. VisualizeOvary.jl simulations/PausedTimeDep_sim_1.jld2 plots/out.mp4
# In-silico lineage tracing:
julia --project=. LabelingExperiment.jl QueuingTimeDep primordial3pct 50
# Model comparison (parallel, via Snakemake; or serially with RunLOO.jl):
snakemake --cores N all
```

Scripts are run from the project root, since paths are relative. Constant-rate models fit in minutes. TimeDep models solve an ODE at every gradient step and are noticeably slower.

---

# Part II: Technical reference

## 1. Data

- Source: `data/WT C57B6 mouse oocyte counts for Dominic.xlsx - Aging WM Tracker.csv`. Columns 4, 6 and 7 are Non-growing NOBOX+ (primordial), Primary and Secondary. Column 8 (Tertiary+) is used only for display.
- `extract_data()` returns five count matrices, one per age (rows are ovaries, columns are the three stages). Ages are split by the non-missing `Condition` entries.
- `load_training_data()` returns `(; counts_2_month, ..., input_data, times_unique, times_vec)`:
  - `input_data` stacks the 4, 6, 9 and 12 month ovaries (27 rows in total).
  - `times_unique = [4, 6, 9, 12]`.
  - `times_vec[i]` is an **index into `times_unique`**, not an age.
- Ovaries are treated as independent. Sister ovaries are ignored, and all mice share one background.

## 2. State space and generator conventions

- Hidden states are $k = 1,\dots,n_h$, plus one final absorbing state $n_h + 1$, the **unobserved bin** (atresia, or exit past secondary). Some wrappers add further trailing absorbing states.
- Generator $W$ is $(n_h+1)\times(n_h+1)$ with **column = source, row = destination**, so $\dot p = W p$ and columns sum to zero.
- The **coarse-graining matrix** $M$ (`coarse_grain`, size $C \times n_h$, with $C = 3$) sums hidden states into observed classes. Each hidden state maps to exactly one class, and the unobserved bin is dropped before $M$ is applied.
- Hidden-state layout produced by `build_queuing_model(k; paused)`: all active substates come first, compartment by compartment, then the paused reservoirs, then the unobserved bin. For the registry's `k = [1, 8, 8]`:

  | index | state |
  |---|---|
  | 1 | primordial |
  | 2–9 | primary substates $S_{2,1..8}$ |
  | 10–17 | secondary substates $S_{3,1..8}$ |
  | 18, 19 | paused primary, paused secondary (Paused models only) |
  | last | unobserved bin |

  For Faddy (`k = [1,1,1]`) the states are 1, 2 and 3, and state 4 is unobserved.

## 3. Propagation

The occupancy of a single follicle at time $T$ is $p(T) = \Phi_\theta(T)\,p^0$, with the initial time **$t_0 = 2$ months (hard-coded)**.

- **Constant $W$:** $\Phi(T) = \exp\big(W (T - 2)\big)$, computed with `ExponentialUtilities.exponential!`. This is `probability_flow(π_vals, W::AbstractMatrix, times)`.
- **Time-dependent $W(t)$:** `transition_fcn(rate_params)` returns a *callable* `t -> W(t)`. Julia then dispatches to `probability_flow(π_vals, Wfun::Function, times)`, which integrates $\dot p = W(t)p$ from $t_0 = 2$ with `Tsit5` (reltol 1e-6, abstol 1e-8).
- Both methods return the full occupancy vector, including trailing absorbing states, for each requested time. Initial mass exists only on the $n_h$ hidden states.
- **Numerical guards:** non-finite or huge ($>10^{10}$) generators, and failed ODE solves, return zero occupancy instead of throwing. This makes NUTS reject the proposal rather than crash. Negative, NaN and Inf entries are zeroed (`_zero_bad!`).

## 4. Observation model (Negative Multinomial)

**Derivation (notes, "Set up").** Suppose the initial count in hidden state $k$ is $\mathrm{Poisson}(\lambda p^0_k)$ with $\sum_k p^0_k = 1$. Because follicles are independent, observed class $s$ at time $T$ is then $\mathrm{Poisson}(\lambda\,\eta_s(T))$, with $\eta(T) = M\,\Phi(T)\,p^0$.

Now let $\lambda \sim \mathrm{Gamma}(a, b)$ (shape $a$, rate $b$) be **ovary specific**. Integrating it out gives

$$
\mathbb P(\mathbf y) = \frac{\Gamma(y_+ + a)}{\Gamma(a)\prod_s y_s!}\; p_0^{\,a} \prod_s q_s^{\,y_s},
\qquad q_s = \frac{\eta_s}{b + H},\quad p_0 = 1-\sum_s q_s = \frac{b}{b+H},\quad H=\sum_s\eta_s .
$$

Equivalent factorization:

- $Y_+ \sim \mathrm{NB}\big(a,\ b/(b+H)\big)$
- $\mathbf Y \mid Y_+ \sim \mathrm{Mult}(Y_+, \eta/H)$

Moments:

- $\mathbb E[Y_s] = a\eta_s/b$
- $\mathrm{Var}(Y_s) = a\eta_s/b + a\eta_s^2/b^2$
- $\mathrm{Cov}(Y_s, Y_u) = a\eta_s\eta_u/b^2$ for $s \neq u$. The classes are positively correlated through the shared $\lambda$.

**Parameterization in code (`total_model`).** The sampled parameters are `ic = [μ_N, p]`, and the code sets

$$
r \equiv a = \frac{p\,\mu_N}{1-p},\qquad b = \frac{p}{1-p}.
$$

At $t = 2$ we have $H = 1$ (all hidden states are observed and $p^0$ sums to 1), so:

- **$\mu_N = r/b$ is the mean total follicle count at 2 months.**
- The 2-month total has variance $\mu_N / p$.

The shape $r$ and rate $b$ are the same at all ages. Only $\eta(t)$ changes. So overdispersion evolves only through the dynamics, which notes.tex flags as a limitation (see "heterogeneity" below).

**Code details:**

- Vendored distribution: `NegativeMultinomial(x₀ = r, p = q)`, with $p_0 = 1 - \sum q$ (`src/NegativeMultinomial.jl`).
- In `total_model`, `q = max.(M*a ./ (sum(a) + b), 1e-10)`, renormalized if $\sum q \ge 1 - 10^{-10}$. Here `a` is the hidden occupancy with the unobserved bin dropped.
- Scoring in `CrossValidation.pointwise_loglik` repeats this exactly. Any change to the likelihood must be mirrored there.
- **The 2-month rows are scored directly against $p^0$ (`π_vals`), with no propagation.** The 4 to 12 month rows are scored against $\eta(t_i)$.

## 5. Erlang (queuing) compartments

Compartment $S$ is expanded into $k_S$ hidden substates in series. Each substate progresses at rate $\alpha_S$ and dies at rate $\delta_S$, with the same death rate in every substate:

$$
S_m \xrightarrow{\alpha_S} S_{m+1},\qquad S_m \xrightarrow{\delta_S} D .
$$

The model is parameterized by:

- $\mu_S$: the mean residence time **conditional on successful progression**.
- $\theta_S$ (written $\rho_S$ in the notes): the probability of progressing rather than dying.

$$
\theta_S = \Big(\tfrac{\alpha_S}{\alpha_S+\delta_S}\Big)^{k_S},\quad
\mu_S = \tfrac{k_S}{\alpha_S+\delta_S}
\;\Longleftrightarrow\;
\alpha_S = \tfrac{k_S}{\mu_S}\theta_S^{1/k_S},\quad
\delta_S = \tfrac{k_S}{\mu_S}\big(1-\theta_S^{1/k_S}\big).
$$

Conditional on progressing, $\tau_S \sim \mathrm{Erlang}(k_S, k_S/\mu_S)$, with coefficient of variation $1/\sqrt{k_S}$. With $k_S = 1$ this is exponential, and then $\alpha = \theta/\mu$, $\delta = (1-\theta)/\mu$.

**The last compartment (secondary).** Both progression and death go to the unobserved bin, so only the total rate $k/\mu$ matters for counts. As a result:

- $\theta_3$ is only weakly identified for $k > 1$.
- $\theta_3$ is **completely unidentified for $k = 1$**, so Faddy pins it to 1.

Primordial always has $k = 1$. The registry uses `k = [1, 8, 8]`.

Initial mass in $S_m$ with $m > 1$ represents follicles already partway through a stage at 2 months. The Dirichlet prior on `π_vals` covers every hidden state.

## 6. Paused states

For each flagged compartment $c$ (`paused = [false, true, true]`) there is a reservoir $P_c$:

$$
P_c \xrightarrow{\ \gamma_c = 1/\mu_{\text{pause},c}\ } S_{c,1}.
$$

- $P_c$ has **no inflow and no death**. It is filled only through the initial distribution `π_vals` (the residual first-wave cohort).
- After leaving $P_c$, a follicle behaves like any other follicle entering compartment $c$.
- $M$ counts $P_c$ as part of observed class $c$, so paused follicles are invisible in the data.

Posterior quantities of interest (see `PausedTimeDepModel.jl`):

- Expected number of paused follicles: $\mu_N\, a_{P_c}(t)$.
- Paused fraction of a class: $a_{P_c}(t) / (M a(t))_c$.

## 7. Time-dependent primordial exit

`build_timedep(base; t0=2, τ=2)` replaces $\mu_1$ by

$$
\mu(t) = \mu_{\text{late}} + \frac{2(\mu_{\text{early}} - \mu_{\text{late}})}{1 + e^{(t - t_0)/\tau}},
$$

so that $\mu(t_0) = \mu_{\text{early}}$ and $\mu(\infty) = \mu_{\text{late}}$.

- The primordial survival $\theta_1$, i.e. the atretic fraction, stays constant.
- $\tau$ is **fixed at 2 months**, not inferred.
- `rate_params` changes from `[μ1, rest...]` to `[μ_early, μ_late, rest...]`, which **shifts every later index by +1**.
- This works for any base model whose compartment 1 has $k = 1$.

## 8. Exit counter

`with_exit_counter(transition_fcn, coarse_grain)` adds an absorbing "exited secondary" state just before the unobserved bin. It counts follicles that leave secondary for tertiary or atresia, separately from earlier deaths.

- Its `coarse_grain` is a 1 × (n_h+1) matrix that selects this state.
- `exited_follicles_plot` and `simulate_ovary` use it.
- It works with both constant and callable generators.

## 9. Priors (`src/ModelConfigs.jl`, shared across models)

Units are **months**. The baseline timescales come from a Faddy fit (per-day rates × 30.4).

| Parameter | Prior | Notes |
|---|---|---|
| `ic[1]` = $\mu_N$ | LogNormal, mean 1750, var 35 000 | mean total count at 2 mo |
| `ic[2]` = $p$ | Beta(3, 750), truncated > 1e-8 | 2-mo total variance = $\mu_N/p$ |
| `π_vals` | Dirichlet(ones($n_h$)) | initial hidden-state distribution |
| $\mu_1$ (and $\mu_{\text{early}}$, $\mu_{\text{late}}$) | LogNormal, mean ≈ 5.48, var 3 | primordial residence |
| $\mu_2$ | LogNormal, mean ≈ 0.765, var 0.008 | primary, conditional on progression |
| $\mu_3$ | LogNormal, mean ≈ 0.577, var 0.008 | secondary, conditional on progression |
| $\theta_c$ | Beta(4, 4) | survival through compartment |
| $\mu_{\text{pause},c}$ | Exponential(5.0) (mean 5 mo) | mean pause duration |

`params_logn(mean, var)` converts a mean and variance into LogNormal $(\mu, \sigma)$. Improving how the priors are parameterized is an open TODO (see CLAUDE.md).

## 10. `rate_params` index tables (check these before indexing a chain)

| Model | `rate_params` |
|---|---|
| Faddy | `[μ1, μ2, μ3, θ1, θ2]` (θ3 ≡ 1 pinned) |
| FaddyTimeDep | `[μ_early, μ_late, μ2, μ3, θ1, θ2]` |
| Queuing | `[μ1, μ2, μ3, θ1, θ2, θ3]` |
| QueuingTimeDep | `[μ_early, μ_late, μ2, μ3, θ1, θ2, θ3]` |
| Paused | `[μ1, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary]` |
| PausedTimeDep | `[μ_early, μ_late, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary]` |

The chain variable names are `ic[1]`, `ic[2]`, `π_vals[j]` and `rate_params[j]`. Use `extract_array(draw, "rate_params")` to get them as a vector.

## 11. The Turing model

```julia
total_model(initial_values,   # counts_2_month (n × 3)
            observations,     # Int.(input_data) for 4–12 mo, or [] for IC-only fit
            times,            # times_vec (indices into times_unique)
            times_unique,     # [4,6,9,12]; pass [] to skip dynamics
            init_priors, π_priors, rate_priors,
            transition_fcn, coarse_grain)
```

Typical call (see any `*Model.jl` script):

```julia
cfg = model_config("Paused")
chain = sample(total_model(counts_2_month, Int.(input_data), times_vec, times_unique,
                           cfg.init_priors, cfg.π_priors, cfg.rate_priors,
                           cfg.transition_fcn, cfg.coarse_grain),
               NUTS(), MCMCThreads(), 300, 2)
```

When `times_unique` is empty, only the initial condition is fitted. Every script does this first ("fixed rates"), which effectively gives the prior predictive for the dynamics.

## 12. Posterior tools

| Function | File | What it gives |
|---|---|---|
| `rand_draw(chain)` | Utils | one random posterior draw as a NamedTuple |
| `chain_draws(chain)` | CrossValidation | iterator over **all** draws |
| `extract_array(draw, "prefix")` | Models | vector of `prefix[i]` values |
| `probability_flow(π, W, times)` | Models | deterministic occupancy $p(t)$ for one draw |
| `sample_model(chain, t, transition_fcn)` | Models | one posterior-predictive **hidden-state** count vector at time t |
| `make_sample_fun(chain, tf; coarse_grain)` | Utils | `t -> M * sample_model(...)`: predictive observed counts |
| `compute_quantiles(sample_fun, t_vals)` | Utils | quantile array for `credible_ribbon_plots` |
| `confidence_intervals(f, t)` | Utils | quantiles of any random scalar `f(t)` (despite the name, these are credible/predictive intervals) |
| `chain_stats_sample` / `calibration_plots` | Utils / PlotUtils | posterior predictive check of per-age mean and covariance |
| `posterior_sojourn_times(chain, tf, M, c)` | Utils | Gillespie residence times in compartment c, conditional on progression, integrated over the posterior |
| `simulate_ovary(draw, tf, M)` | Utils | full per-follicle stochastic trajectory of one ovary (thinning handles time-dependent W) |
| `plot_param_posteriors`, `corner_plot`, `plot_π_posterior` | PlotUtils | prior vs posterior, pairwise correlations |

`sample_model` draws a Negative Multinomial over hidden states, which `make_sample_fun` then coarse-grains. This is equivalent in distribution to the likelihood because Poisson splitting is consistent. But it uses `clamp` instead of the floor and renormalization in `total_model`, so **never use it for scoring**. Use `pointwise_loglik` instead.

## 13. Model comparison (`src/CrossValidation.jl`)

- The unit held out is one ovary's full 3-vector at 4, 6, 9 or 12 months. The 2-month data is **always** in training.
- `elpd_cv_fold` refits on the remaining data. It then scores $\log \frac1S\sum_s p(y_i\mid\theta^{(s)})$ (`logmeanexp`) for each held-out ovary.
- There are two fold schemes:
  - leave-one-ovary-out (the default)
  - `age_folds(data)`: leave one age out, a stricter test of the dynamics
- `elpd_delta(results; by, ref)` gives paired differences against the best model: $\Delta = \sum_i d_i$, $\mathrm{se} = \sqrt{n\,\mathrm{Var}(d_i)}$.
- Running it: in parallel with the `Snakefile` (6 models × 31 folds, see `loo/` and `docs/LOO_instructions.md`; `slurm_script_rusty.sh` runs it on the cluster), or serially with `RunLOO.jl`.
- Current result (notes, Fig. LOO): the paused model has the best ELPD, but the margins are modest.

## 14. Recipes

**Add a new model.**

1. Build `(transition_fcn, coarse_grain, n_hidden)`. Reuse `build_queuing_model`, `build_timedep` and `with_exit_counter` where possible. `transition_fcn(rate_params)` must return either an $(n_h+1)^2$ generator (column = source) whose last state is the unobserved bin, or a callable `t -> W(t)`.
2. Add an entry to `model_registry()` with priors whose length matches `rate_params`.
3. Copy the closest `*Model.jl` script and update the parameter indices and labels.
4. To include it in model comparison, add its name to `MODELS` in the `Snakefile`.

**Get a credible interval for a new quantity.** Write `f(draw)`, map it over `chain_draws(chain)` (or call `rand_draw` many times), and take quantiles. For a predictive version, add the noise, either with `make_sample_fun` or with `simulate_ovary`.

**In-silico experiments.**

- Add a labelling scheme to the `label_schemes` dict in `LabelingExperiment.jl`. A scheme is a rule `(category, hidden_state) -> probability labelled at t0`.
- To test an intervention (for example a perturbed rate after some age), wrap `transition_fcn` so it returns a modified `W(t)`, then reuse `probability_flow` or `simulate_ovary`. Evaluate across posterior draws so that the effect comes with a credible interval.

**Fitted chains** are saved to `models/<Model>_chain.jld2` by the analysis scripts, and `SimulateOvary.jl` and `LabelingExperiment.jl` load them from there. Short runs (300 draws × 2 chains) are the default. Check the R-hat and ESS values in `describe(chain)` before trusting tail quantiles.

## 15. Gotchas

- `t0 = 2.0` is hard-coded in `probability_flow` (`exponential!(W*(t-2))`) and in `build_timedep`.
- `times_vec` holds indices, not ages. `elpd_cv_fold` converts with `times_unique[times_vec]`.
- The final state is always dropped as unobserved (`[1:end-1]`). Wrappers that add absorbing states must put them **before** that final bin.
- In TimeDep models every posterior-predictive draw needs an ODE solve, so keep `N_samples` modest.
- `VisualizeOvary.jl` defines its own `mean` helper. Load it inside a module, as `LabelingExperiment.jl` does.
- `src/ModelConfigs.jl` is the single source of truth for topology and priors. Do not redefine priors in scripts.
- `docs/` is synced from Overleaf. Do not edit it here.

## 16. Open directions (from notes.tex / CLAUDE.md)

- **Between-ovary heterogeneity in primordial exit:** $\mu_i(t) = \mu(t)\,e^{\sigma_r z_i}$ with $z_i \sim \mathcal N(0,1)$ and a half-normal prior on $\sigma_r$. The likelihood would be evaluated conditional on each ovary's rates. This would also let overdispersion change with age.
- **Synthetic recovery:** simulate data from one model, then check that cross-validation picks that model.
- **Tertiary follicles** as an observed compartment.
- **Interactions** (for example AMH feedback). The current $\mu(t)$ is the simplest effective stand-in.
- **Non-first-wave pausing:** a paused state that later follicles can also enter and leave.
- **Practical identifiability:** what the posterior can and cannot pin down, studied through corner plots and prior/posterior overlap.
- **Better prior parameterizations** tied to literature measurements.
