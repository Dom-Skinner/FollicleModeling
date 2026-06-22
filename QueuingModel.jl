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
k = [1, 8, 8]
(; transition_fcn, coarse_grain, n_hidden) = build_queuing_model(k)


# Ballpark timescales from the Faddy fit (converted to 1/month), used to set priors.
θ_fixed = [0.0043, 0.0017, 0.043, 0.057]*30.4
μ1_fixed = 1/(θ_fixed[1] + θ_fixed[2])
μ2_fixed = 1/θ_fixed[3]
μ3_fixed = 1/θ_fixed[4]


init_priors = [LogNormal(params_logn(1750,35_000)...),
                Truncated(Beta(3, 750), 1e-8,Inf)]
π_priors = Dirichlet(ones(n_hidden))      # full Dirichlet over all hidden substates

# rate_params = [μ1, μ2, μ3, θ1, θ2, θ3]:
#   μ_c : mean residence time in compartment c *conditional on successful
#         progression* (Erlang(k_c, k_c/μ_c) among surviving follicles)
#   θ1  : survival probability Primordial -> Primary
#   θ2  : survival probability Primary    -> Secondary
#   θ3  : survival probability Secondary  -> growing/dead bin. Weakly identified:
#         progression and death from Secondary are both unobserved, so θ3 is
#         informed mainly by the prior (and only weakly by the Secondary
#         residence-time shape). Kept for a uniform parameterization.
rate_priors = [ LogNormal(params_logn(μ1_fixed,3.0)...),
    LogNormal(params_logn(μ2_fixed,0.008)...),
    LogNormal(params_logn(μ3_fixed,0.008)...),
    Beta(4,4),
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
