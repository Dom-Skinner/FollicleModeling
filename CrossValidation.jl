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


# Exact cross-validation ELPD for one model.
#
# `data` is the NamedTuple from load_training_data(). `folds` is a vector of
# held-out row-index groups into data.input_data; the default gives exact
# leave-one-ovary-out, and passing grouped indices (e.g. one group per age) gives
# K-fold from the same code path. Every fold keeps all of counts_2_month plus all
# non-held-out input_data rows. Returns a tidy long DataFrame(model, ovary, age, elpd).
function elpd_cv(config, data; folds = [[i] for i in 1:size(data.input_data, 1)],
                 n_samples = 300, n_chains = 2, seed = 1)
    Y = Int.(data.input_data)
    ages = data.times_unique[data.times_vec]   # actual age (months) per ovary row
    N = size(Y, 1)
    elpd = fill(NaN, N)

    for fold in folds
        keep = setdiff(1:N, fold)
        Random.seed!(seed)
        model = total_model(data.counts_2_month, Y[keep, :], data.times_vec[keep],
                            data.times_unique, config.init_priors, config.π_priors,
                            config.rate_priors, config.transition_fcn, config.coarse_grain)
        chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)
        for i in fold
            elpd[i] = pointwise_elpd(chain, config, Y[i, :], ages[i])
        end
    end
    return DataFrame(model = config.name, ovary = 1:N, age = ages, elpd = elpd)
end


# ---- Summaries over a stacked results DataFrame (all models, from elpd_cv) ----

# Per-model total ELPD, difference from the best model (best = 0, others ≤ 0),
# and the standard error of that difference from the pointwise contrasts
# d_i = elpd_{i,best} - elpd_{i,m}: SE ≈ sqrt(n · Var_i(d_i)).
function elpd_compare(results)
    models  = unique(results.model)
    ovaries = sort(unique(results.ovary))
    E = Dict(m => [only(results.elpd[(results.model .== m) .& (results.ovary .== o)])
                   for o in ovaries] for m in models)
    totals = Dict(m => sum(E[m]) for m in models)
    best = models[argmax([totals[m] for m in models])]
    rows = [(; model = m,
               elpd  = totals[m],
               Δelpd = totals[m] - totals[best],
               se    = sqrt(length(ovaries) * var(E[best] .- E[m])))
            for m in models]
    return sort(DataFrame(rows), :elpd, rev = true)
end

# Per-(model, age) total ELPD and ovary count, for the age-resolved comparison.
function elpd_by_age(results)
    return sort(combine(groupby(results, [:model, :age]),
                        :elpd => sum => :elpd, nrow => :n), [:model, :age])
end
