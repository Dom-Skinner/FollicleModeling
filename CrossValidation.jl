# Leave-one-ovary-out (and K-fold) cross-validation for the follicle models.
#
# The held-out unit is one ovary's observed 3-vector [Primordial, Primary,
# Secondary] at age 4/6/9/12 months. The 2-month counts are always kept in
# training as the initial condition (see docs/LOO_instructions.md). For each
# held-out ovary we compute the log posterior-predictive density (ELPD) of its
# count vector under the model refit without it.
#
# The pointwise log-likelihood below replicates total_model's scoring EXACTLY
# (Models.jl total_model, the observation loop) — NOT sample_model, which drops
# the coarse-graining, uses clamp instead of the max-floor, and skips the
# renormalization branch. Getting this wrong silently corrupts the ELPD.

isdefined(@__MODULE__, :total_model)        || include("Models.jl")
isdefined(@__MODULE__, :load_training_data) || include("Utils.jl")

using DataFrames
using Turing
using Random
using Statistics
using AugmentedGPLikelihoods


# Log-likelihood of one observed ovary `y_i` at `age` under one posterior draw,
# for a given model `config` (a NamedTuple from model_registry). Mirrors the
# NegativeMultinomial scoring inside total_model.
function pointwise_loglik(draw::NamedTuple, config, y_i, age)
    ic = extract_array(draw, "ic")
    μ, p = ic[1], ic[2]
    r = p * μ / (1 - p)
    b = p / (1 - p)

    π_vals      = extract_array(draw, "π_vals")
    rate_params = extract_array(draw, "rate_params")
    W = config.transition_fcn(rate_params)

    a_k = probability_flow(π_vals, W, [age])[1][1:end-1]  # drop final unobserved state
    A = sum(a_k)
    π_k = max.(config.coarse_grain * a_k ./ (A + b), 1e-10)
    if sum(π_k) > 1 - 1e-10
        π_k = π_k ./ (sum(π_k) + 1e-9)
    end
    return logpdf(AugmentedGPLikelihoods.SpecialDistributions.NegativeMultinomial(r, π_k), Int.(y_i))
end


# Numerically stable log(mean(exp.(v))).
function logmeanexp(v)
    m = maximum(v)
    return m + log(sum(exp, v .- m)) - log(length(v))
end


# Iterate a Turing chain's draws as NamedTuples (parameter values only).
chain_draws(chain) = (NamedTuple(row) for row in eachrow(DataFrame(chain)[:, Not([:chain, :iteration])]))


# Log posterior-predictive density of one ovary, averaging over all draws of `chain`.
function pointwise_elpd(chain, config, y_i, age)
    return logmeanexp([pointwise_loglik(d, config, y_i, age) for d in chain_draws(chain)])
end


# S×N matrix of pointwise log-likelihoods (draws × ovaries) for a fitted chain.
# Reused for the in-sample first-pass ranking and the replica verification;
# also the exact input a PSIS-LOO package would need later.
function loglik_matrix(chain, config, Y, ages)
    draws = collect(chain_draws(chain))
    S, N = length(draws), size(Y, 1)
    return [pointwise_loglik(draws[s], config, Y[i, :], ages[i]) for s in 1:S, i in 1:N]
end


# Exact cross-validation ELPD for a SINGLE fold of one model.
#
# `fold` is a vector of held-out row indices into data.input_data (a singleton for
# leave-one-ovary-out, a whole age group for leave-one-age-out). Refits total_model
# on every non-held-out row (plus all of counts_2_month) and scores each held-out
# ovary. Returns a tidy DataFrame(model, ovary, age, elpd) with one row PER HELD-OUT
# ovary (ovary = global row index into data.input_data). Seeding is done here so a
# single fold is reproducible on its own — this is the shared unit of work for both
# the serial elpd_cv loop below and the parallel Snakemake worker (loo/fit_fold.jl),
# guaranteeing they score through identical code.
function elpd_cv_fold(config, data, fold; n_samples = 300, n_chains = 2, seed = 1)
    Y = Int.(data.input_data)
    ages = data.times_unique[data.times_vec]   # actual age (months) per ovary row
    N = size(Y, 1)

    keep = setdiff(1:N, fold)
    Random.seed!(seed)
    model = total_model(data.counts_2_month, Y[keep, :], data.times_vec[keep],
                        data.times_unique, config.init_priors, config.π_priors,
                        config.rate_priors, config.transition_fcn, config.coarse_grain)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)
    return DataFrame(model = config.name, ovary = collect(fold), age = ages[fold],
                     elpd = [pointwise_elpd(chain, config, Y[i, :], ages[i]) for i in fold])
end


# Exact cross-validation ELPD for one model, across a set of folds.
#
# `data` is the NamedTuple from load_training_data(). `folds` is a vector of
# held-out row-index groups into data.input_data; the default gives exact
# leave-one-ovary-out, and passing grouped indices (e.g. one group per age) gives
# K-fold from the same code path. Every fold keeps all of counts_2_month plus all
# non-held-out input_data rows. Returns a tidy long DataFrame(model, ovary, age, elpd),
# one row per held-out ovary, sorted by ovary. Thin loop over elpd_cv_fold.
function elpd_cv(config, data; folds = [[i] for i in 1:size(data.input_data, 1)],
                 n_samples = 300, n_chains = 2, seed = 1)
    rows = reduce(vcat,
        [elpd_cv_fold(config, data, fold; n_samples = n_samples, n_chains = n_chains, seed = seed)
         for fold in folds])
    return sort!(rows, :ovary)
end


# Fold vector that holds out each age group as a whole (leave-one-age-out), for
# elpd_cv's `folds` argument. Contrast the default (one ovary per fold): here the
# model is refit with an entire age missing, so scoring the withheld ovaries at
# that age is a stricter test of the dynamics (no same-age siblings in training).
function age_folds(data)
    ages = data.times_unique[data.times_vec]
    return [findall(==(a), ages) for a in sort(unique(ages))]
end


# ---- Model comparison via paired ELPD differences ----------------------------

# ΔELPD of each model relative to the best model (best = highest total ELPD) —
# the standard leave-one-out comparison. For each held-out ovary i the contrast
#     d_i = elpd_{i,model} - elpd_{i,best}
# is a *paired* difference, so the large shared ovary-to-ovary count variance
# cancels and its spread reflects only how consistently a model beats (or trails)
# the best one. Within each group we report the summed difference and its paired
# standard error,
#     Δelpd = Σ_i d_i,   se = sqrt(n · Var_i(d_i))
# (the usual loo elpd_diff / se_diff); the best model sits at Δelpd = 0 with se 0.
#
# `by` sets the grouping: [:model] for the overall comparison, [:model, :age] for
# the age-resolved one. `ref` fixes the reference model (its name); by default the
# best model within `results` (highest total ELPD) is used. Passing an explicit
# `ref` keeps the zero line on the same model across figures built from different
# CV runs. Returns (ref_model_name, summary_dataframe with the `by` columns plus
# :Δelpd, :se, :n).
function elpd_delta(results; by = [:model], ref = nothing)
    refmodel = ref
    if refmodel === nothing
        totals = combine(groupby(results, :model), :elpd => sum => :total)
        refmodel = totals.model[argmax(totals.total)]
    end
    ref_elpd = Dict(r.ovary => r.elpd for r in eachrow(results[results.model .== refmodel, :]))
    d = copy(results)
    d.d = [e - ref_elpd[o] for (o, e) in zip(results.ovary, results.elpd)]   # paired difference
    summary = combine(groupby(d, by),
                      :d => sum => :Δelpd,
                      :d => (x -> sqrt(length(x) * var(x))) => :se,
                      nrow => :n)
    return refmodel, summary
end
