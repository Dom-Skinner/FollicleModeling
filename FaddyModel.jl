using Plots
using AugmentedGPLikelihoods
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")
include("PlotUtils.jl")
include("ModelConfigs.jl")

(; counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month,
   input_data, times_unique, times_vec) = load_training_data()

# Model definition (topology + priors) comes from the shared registry so it stays
# in sync with the other scripts and the cross-validation code. Faddy =
# build_queuing_model([1,1,1]) with Primary survival pinned to 1; rate_params =
# [μ1, μ2, μ3, θ12].
(; transition_fcn, coarse_grain, init_priors, π_priors, rate_priors) = model_config("Faddy")


################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),1000,2);
#jldsave("models/FaddyModel_fixed.jld2"; prior_chain)


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_fcn)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Faddy_model_fixed_rates.pdf")




mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/predictive_checks_fixed_rates.pdf")

# ========== Now fit everything, not just initial conditions ==========
    

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),300,2);

    
sample_fun = make_sample_fun(chain, transition_fcn)

N_samples = 10_000
t_vals = 2:0.5:12
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Faddy_model_fitted_rates.pdf")


plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/predictive_checks_fitted_rates.pdf")

# prior/posterior check
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]", "rate_params[4]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:3, 0:0.01:2, 0:0.01:1],
    ["μ_N", "p", "μ1", "μ2", "μ3", "θ12"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., layout=(3,3), size=(1000,400), margin=4mm)
savefig("plots/Faddy_model_fitted_params.pdf")

# Presentation-quality parameter plots
pres_plots = plot_param_posteriors(chain,
    ["rate_params[2]", "rate_params[3]", "rate_params[4]"],
    rate_priors[2:4],
    [0:0.01:1.2, 0:0.01:1.2, 0:0.01:1],
    ["Avg time as Primary", "Avg time as Secondary", "Probability of reaching primary"];
    ylabel="Density")
plot(pres_plots..., layout=(1,3), size=(1000,300), margin=5mm)
savefig("plots/PosteriorPredsFaddy.pdf")

# ===== Conditional residence-time distributions =====
# Time a follicle spends in Primary / Secondary GIVEN it successfully progresses
# out. In the Faddy model Primary and Secondary have no death channel, so every
# follicle progresses and the residence times are simple exponentials with means
# μ2 and μ3 (k=1, so conditional and unconditional coincide). Integrates over
# posterior uncertainty.
primary_times   = posterior_sojourn_times(chain, transition_fcn, coarse_grain, 2; N=50_000)
secondary_times = posterior_sojourn_times(chain, transition_fcn, coarse_grain, 3; N=50_000)

p_soj = density(primary_times, label="Primary", lw=2, fill=(0,0.15), grid=false,
                xlabel="Time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary", lw=2, fill=(0,0.15),xlims=(0,5))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/Faddy_conditional_sojourn_times.pdf")