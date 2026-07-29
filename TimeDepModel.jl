using Plots
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

# Faddy topology with a time-dependent primordial exit μ(t) (sigmoid, τ fixed at 2
# months). The generator is now a callable t -> W(t), so total_model routes through
# the ODE method of probability_flow (dp/dt = W(t) p) instead of a matrix exponential.
# rate_params = [μ_early, μ_late, μ2, μ3, θ12, θ23]. See build_timedep_faddy.
# NOTE: each posterior-predictive draw now costs one ODE solve, so N_samples for the
# ribbon plots is smaller than in the matrix-exponential scripts.
(; transition_fcn, coarse_grain, init_priors, π_priors, rate_priors) = model_config("FaddyTimeDep")


################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),1000,2);


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_fcn)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/TimeDep_model_fixed_rates.pdf")


mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/TimeDep_predictive_checks_fixed_rates.pdf")


# ========== Now fit everything, not just initial conditions ==========

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),300,2);

savefig(plot(chain), "plots/TimeDep_model_chain.pdf")

sample_fun = make_sample_fun(chain, transition_fcn)

N_samples = 2_000        # reduced: each draw is an ODE solve (see note above)
t_vals = 2:0.5:12
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/TimeDep_model_fitted_rates.pdf")


plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/TimeDep_predictive_checks_fitted_rates.pdf")

# prior/posterior check
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]",
     "rate_params[4]", "rate_params[5]", "rate_params[6]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:9, 0:0.01:3, 0:0.01:2, 0:0.01:1, 0:0.01:1],
    ["μ_N", "p", "μ_early", "μ_late", "μ2", "μ3", "θ12", "θ23"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., layout=(4,3), size=(1000,600), margin=4mm)
savefig("plots/TimeDep_model_fitted_params.pdf")

# Corner plot: pairwise posterior correlations (off-diagonal) + marginals (diagonal).
savefig(corner_plot(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]",
     "rate_params[4]", "rate_params[5]", "rate_params[6]"];
    labels=["μ_N", "p", "μ_early", "μ_late", "μ2", "μ3", "θ12", "θ23"], size=(1500, 1500)),
    "plots/TimeDep_model_corner.pdf")

# ===== Posterior of the time-dependent primordial residence time μ(t) =====
# μ(t) = μ_late + 2(μ_early - μ_late)/(1 + exp((t - 2)/τ)), τ = 2 months.
chain_df = DataFrame(chain)
μ_of_t(t) = begin
    rp = extract_array(rand_draw(chain_df), "rate_params")
    μ_early, μ_late = rp[1], rp[2]
    μ_late + 2 * (μ_early - μ_late) / (1 + exp((t - 2.0) / 2.0))
end
μ_q = stack([confidence_intervals(μ_of_t, t; N_samples=4000) for t in t_vals])   # (n_q × n_t)
p_μ = credible_ribbon_plots(reshape(μ_q, size(μ_q, 1), size(μ_q, 2), 1), t_vals)[1]
plot(p_μ, xlabel="Age (months)", ylabel="Primordial mean residence μ(t)",
     title="Time-dependent primordial exit", size=(600,400), margin=4mm)
savefig("plots/TimeDep_mu_of_t.pdf")

# ===== Conditional residence-time distributions (primary/secondary only) =====
# The primordial residence is time-varying (μ(t)), so it has no single sojourn
# distribution and is omitted. Primary/secondary rates are constant, so we score
# them with the generator frozen at the (arbitrary) reference time t=2.
ref_transition = rp -> transition_fcn(rp)(2.0)
primary_times   = posterior_sojourn_times(chain, ref_transition, coarse_grain, 2; N=50_000)
secondary_times = posterior_sojourn_times(chain, ref_transition, coarse_grain, 3; N=50_000)

p_soj = density(primary_times, label="Primary", lw=2, fill=(0,0.15), grid=false,
                xlabel="Time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary", lw=2, fill=(0,0.15),xlims=(0,5))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/TimeDep_conditional_sojourn_times.pdf")
