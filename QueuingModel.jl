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


# ===== Erlang shapes per observed compartment [Primordial, Primary, Secondary] =====
# k = [1,1,1] recovers the exponential (Faddy-style) model; larger k gives a more
# clock-like (less dispersed) maturation time (CV = 1/√k). Swap freely to compare.
k = [1, 2, 2]
(; transition_fcn, coarse_grain, n_hidden) = build_queuing_model(k)


# Ballpark timescales from the Faddy fit (converted to 1/month), used to set priors.
θ_fixed = [0.0043, 0.0017, 0.043, 0.057]*30.4
w1_fixed = 1/(θ_fixed[1] + θ_fixed[2])
w2_fixed = 1/θ_fixed[3]
w3_fixed = 1/θ_fixed[4]


init_priors = [LogNormal(params_logn(1750,35_000)...),
                Truncated(Beta(3, 750), 1e-8,Inf)]
π_priors = Dirichlet(ones(n_hidden))      # full Dirichlet over all hidden substates

# rate_params = [w1, w2, w3, θ1, θ2]:
#   w_c : mean residence time in compartment c (total time spent there)
#   θ1  : survival probability Primordial -> Primary
#   θ2  : survival probability Primary    -> Secondary
rate_priors = [ LogNormal(params_logn(w1_fixed,3.0)...),
    LogNormal(params_logn(w2_fixed,0.008)...),
    LogNormal(params_logn(w3_fixed,0.008)...),
    Beta(4,4),
    Beta(4,4)]


################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),1000,2);
#jldsave("models/QueuingModel_fixed.jld2"; prior_chain)


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_fcn; coarse_grain=coarse_grain)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Queuing_model_fixed_rates.pdf")


mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/Queuing_predictive_checks_fixed_rates.pdf")


# ========== Now fit everything, not just initial conditions ==========

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),300,2);


sample_fun = make_sample_fun(chain, transition_fcn; coarse_grain=coarse_grain)

N_samples = 20_0
t_vals = 2:0.5:12
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Queuing_model_fitted_rates.pdf")


plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/Queuing_predictive_checks_fitted_rates.pdf")

# prior/posterior check
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]", "rate_params[4]", "rate_params[5]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:3, 0:0.01:2, 0:0.01:1, 0:0.01:1],
    ["μ", "p", "w1", "w2", "w3", "θ1", "θ2"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., size=(1400,800), margin=4mm)
savefig("plots/Queuing_model_fitted_params.pdf")
