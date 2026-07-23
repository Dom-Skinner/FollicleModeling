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
# in sync with the other scripts and the cross-validation code. Erlang shapes
# k = [1,8,8] per compartment [Primordial, Primary, Secondary]; larger k gives a
# more clock-like (less dispersed) maturation time (CV = 1/√k). rate_params =
# [μ1, μ2, μ3, θ1, θ2, θ3] (μ_c = conditional mean residence, θ_c = survival).
(; k, transition_fcn, coarse_grain, init_priors, π_priors, rate_priors) = model_config("Queuing")


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

savefig(plot(chain), "plots/Queuing_model_chain.pdf")

sample_fun = make_sample_fun(chain, transition_fcn; coarse_grain=coarse_grain)

N_samples = 4_000
t_vals = 2:0.25:12
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
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]", "rate_params[4]", "rate_params[5]", "rate_params[6]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:3, 0:0.01:2, 0:0.01:1, 0:0.01:1, 0:0.01:1],
    ["μ_N", "p", "μ1", "μ2", "μ3", "θ1", "θ2", "θ3"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., size=(1400,800), margin=4mm)
savefig("plots/Queuing_model_fitted_params.pdf")

# Corner plot: pairwise posterior correlations (off-diagonal) + marginals (diagonal)
# for the physical parameters. The 17 initial hidden-state fractions π_vals are
# omitted (too many); add "π_vals[k]" keys to include specific ones.
savefig(corner_plot(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]",
     "rate_params[4]", "rate_params[5]", "rate_params[6]"];
    labels=["μ_N", "p", "μ1", "μ2", "μ3", "θ1", "θ2", "θ3"], size=(1500, 1500)),
    "plots/Queuing_model_corner.pdf")

pres_plots = plot_param_posteriors(chain,
    ["rate_params[2]", "rate_params[3]", "rate_params[4]"],
    rate_priors[2:4],
    [0:0.01:1.2, 0:0.01:1.2, 0:0.01:1],
    ["Avg time as Primary", "Avg time as Secondary", "Probability of reaching primary"];
    ylabel="Density")
plot(pres_plots..., layout=(1,3), size=(1000,300), margin=5mm)
savefig("plots/PosteriorPredsQueuing.pdf")

# ===== Conditional residence-time distributions =====
# Time a follicle spends in Primary / Secondary GIVEN it successfully progresses
# out (rather than dying). For this Erlang model that conditional time is exactly
# Erlang(k_c, k_c/μ_c), independent of the death rate δ_c. We therefore sample it
# directly rather than via the dynamics: this is robust to the weakly identified
# survival probabilities (in particular θ3, where graduation and death from
# Secondary both flow to the unobserved bin and are indistinguishable). One
# Erlang draw per posterior sample integrates over posterior uncertainty in μ_c.
chain_df = DataFrame(chain)
conditional_sojourn(c; N=50_000) =
    [rand(Erlang(k[c], extract_array(rand_draw(chain_df), "rate_params")[c] / k[c])) for _ in 1:N]
primary_times   = conditional_sojourn(2)
secondary_times = conditional_sojourn(3)

p_soj = density(primary_times, label="Primary", lw=2, fill=(0,0.15), grid=false,
                xlabel="Time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary", lw=2, fill=(0,0.15),xlims=(0,5))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/Queuing_conditional_sojourn_times.pdf")

# ===== Illustrative Erlang waiting-time distributions (mean 1) =====
# Erlang(k, 1/k) has mean 1 for any k; increasing k sharpens the distribution
# around its mean (CV = 1/√k). k=1 is the memoryless exponential.
ts = 0:0.005:1.5
p_erlang = plot(grid=false, xlabel="Time (months)", ylabel="Density", legend=:topright)
for kk in (1, 8, 16)
    plot!(p_erlang, ts, pdf.(Erlang(kk, 0.5/kk), ts), lw=2, label="k = $kk")
end
plot(p_erlang, size=(600,400), margin=4mm)
savefig("plots/Erlang_waiting_times.pdf")
