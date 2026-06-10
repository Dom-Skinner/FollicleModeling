using Plots
using AugmentedGPLikelihoods
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")
include("PlotUtils.jl")

(; counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month,
   input_data, times_unique, times_vec) = load_training_data()


θ_fixed = [0.0043, 0.0017, 0.043, 0.057]*30.4 # fixed values from Faddy converted into 1/month
w1_fixed = 1/(θ_fixed[1] + θ_fixed[2])
w2_fixed = 1/θ_fixed[3]
w3_fixed = 1/θ_fixed[4]
θ12_fixed = θ_fixed[1]/(θ_fixed[1] + θ_fixed[2])


# ============================================================
# Priors shared across both fitting steps
# ============================================================

init_priors = [LogNormal(params_logn(1750,35_000)...),
                Truncated(Beta(3, 750), 1e-8, Inf)]

# 7-element rate prior for [w1, w2, w3, θ12, θ34, θ6, θ7]
rate_priors_paused = [
    LogNormal(params_logn(w1_fixed,3.0)...),
    LogNormal(params_logn(w2_fixed,0.02)...),
    LogNormal(params_logn(w3_fixed,0.02)...),
    Beta(4,4),        # θ12
    Beta(4,4),        # θ34
    Gamma(2.0, 0.3),  # θ6: resume rate for paused primary
    Gamma(2.0, 0.3),  # θ7: resume rate for paused secondary
]

# coarse-graining matrix: maps 5 non-absorbing states to 3 observed categories
# [Primordial | Primary-active + Primary-paused | Secondary-active + Secondary-paused]
coarse_grain_paused = Float64[1 0 0 0 0;
                               0 1 1 0 0;
                               0 0 0 1 1]

function transition_matrix_paused(params)
    w1, w2, w3, θ12, θ34, θ6, θ7 = params
    θ = [θ12/w1, (1-θ12)/w1, θ34/w2, (1-θ34)/w2, 1/w3, θ6, θ7]
    return [
        -(θ[1]+θ[2])    0.0        0.0     0.0   0.0   0.0
        θ[1]      -(θ[3]+θ[4])     θ[6]    0.0   0.0   0.0
        0.0             0.0       -θ[6]    0.0   0.0   0.0
        0.0             θ[4]       0.0    -θ[5]  θ[7]  0.0
        0.0             0.0        0.0     0.0  -θ[7]  0.0
        θ[2]            θ[3]       0.0     θ[5]  0.0   0.0
    ]
end



# ============================================================
# Step 1: Fit initial conditions only (prior predictive for dynamics)
# ============================================================
# Passing empty observations/times mirrors the fixed-rate step in FaddyModel.jl:
# rates are drawn from their priors but not informed by the 4-12 month data.

@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors, Dirichlet(ones(5)), rate_priors_paused,
    transition_matrix_paused, coarse_grain_paused),
    NUTS(), MCMCThreads(), 1000, 2);

N_samples = 400
t_vals = 2:0.25:12

sample_fun_prior = make_sample_fun(prior_chain, transition_matrix_paused; coarse_grain=coarse_grain_paused)
quantiles = compute_quantiles(sample_fun_prior, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fixed_rates.pdf")

mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun_prior, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/pausing_predictive_checks_fixed_rates.pdf")


# ============================================================
# Step 2: Full fit (rates + initial conditions)
# ============================================================

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique, init_priors, Dirichlet(ones(5)), rate_priors_paused,
    transition_matrix_paused, coarse_grain_paused),
    NUTS(), MCMCThreads(), 2000, 2);

# Posterior predictive credible interval ribbons

t_vals = 2:0.5:12
sample_fun = make_sample_fun(chain, transition_matrix_paused; coarse_grain=coarse_grain_paused)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fitted_rates.pdf")

# Posterior predictive calibration (mean and covariance)
plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/pausing_predictive_checks_fitted_rates.pdf")

# Prior/posterior comparison for each parameter
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]",
     "rate_params[4]", "rate_params[5]", "rate_params[6]", "rate_params[7]"],
    [init_priors..., rate_priors_paused...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:3, 0:0.01:2,
     0:0.01:1, 0:0.01:1, 0:0.01:1, 0:0.01:1],
    ["μ", "p", "w1", "w2", "w3", "θ12", "θ34", "θ6", "θ7"])
p_π = plot_π_posterior(chain, Dirichlet(ones(5)))
plot(p_π..., param_plots..., layout=(4,4), size=(1000,600), margin=4mm)
savefig("plots/pausing_model_prior_posterior.pdf")

# Presentation-quality parameter plots
pres_plots = plot_param_posteriors(chain,
    ["rate_params[2]", "rate_params[3]", "rate_params[4]", "rate_params[5]"],
    rate_priors_paused[2:5],
    [0:0.01:3, 0:0.01:2, 0:0.01:1, 0:0.01:1],
    ["Avg time as Primary (months)", "Avg time as Secondary (months)",
     "Probability of reaching primary", "Probability of reaching \n secondary from primary"];
    ylabel="Density")
plot(pres_plots..., layout=(1,4), size=(1000,300), margin=6mm, xguidefontsize=8, guidefontsize=8)
savefig("plots/PosteriorPredsPaused.pdf")
