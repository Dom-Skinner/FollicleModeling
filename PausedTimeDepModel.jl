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

# Paused + Erlang model (k=[1,8,8], paused=[false,true,true]) with a time-dependent
# primordial exit μ(t) (sigmoid, τ fixed at 2 months). The generator is a callable
# t -> W(t), so total_model routes through the ODE method of probability_flow. rate_params
# = [μ_early, μ_late, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary] — μ1 is split
# into (μ_early, μ_late), so residence/pause indices shift by +1 vs PausingModel.jl.
# NOTE: each posterior-predictive draw costs one ODE solve, so N_samples is reduced.
(; k, paused, transition_fcn, coarse_grain, init_priors, π_priors, rate_priors) = model_config("PausedTimeDep")


################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),1000,2);


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_fcn; coarse_grain=coarse_grain)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/PausedTimeDep_model_fixed_rates.pdf")


mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/PausedTimeDep_predictive_checks_fixed_rates.pdf")


# ========== Now fit everything, not just initial conditions ==========

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),300,2);

savefig(plot(chain), "plots/PausedTimeDep_model_chain.pdf")

sample_fun = make_sample_fun(chain, transition_fcn; coarse_grain=coarse_grain)

N_samples = 2_000        # reduced: each draw is an ODE solve (see note above)
t_vals = 2:0.5:12
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/PausedTimeDep_model_fitted_rates.pdf")


plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/PausedTimeDep_predictive_checks_fitted_rates.pdf")

# prior/posterior check
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]", "rate_params[4]",
     "rate_params[5]", "rate_params[6]", "rate_params[7]", "rate_params[8]", "rate_params[9]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:9, 0:0.01:3, 0:0.01:2,
     0:0.01:1, 0:0.01:1, 0:0.01:1, 0:0.1:20, 0:0.1:20],
    ["μ_N", "p", "μ_early", "μ_late", "μ2", "μ3", "θ1", "θ2", "θ3", "μ_pause_primary", "μ_pause_secondary"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., size=(1700,1000), margin=4mm)
savefig("plots/PausedTimeDep_model_fitted_params.pdf")

# Corner plot: pairwise posterior correlations (off-diagonal) + marginals (diagonal).
savefig(corner_plot(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]", "rate_params[4]",
     "rate_params[5]", "rate_params[6]", "rate_params[7]", "rate_params[8]", "rate_params[9]"];
    labels=["μ_N", "p", "μ_early", "μ_late", "μ2", "μ3", "θ1", "θ2", "θ3",
            "μ_pause_primary", "μ_pause_secondary"], size=(1800, 1800)),
    "plots/PausedTimeDep_model_corner.pdf")

pres_plots = plot_param_posteriors(chain,
    ["rate_params[3]", "rate_params[4]", "rate_params[8]", "rate_params[9]"],
    vcat(rate_priors[3:4], rate_priors[8:9]),
    [0:0.01:1.2, 0:0.01:1.2, 0:0.1:30, 0:0.1:30],
    ["Avg time as Primary", "Avg time as Secondary", "Avg time as paused primary", "Avg time as paused secondary"];
    ylabel="Density")
# ----- Paused-follicle summaries: number and within-compartment fraction -----
# Paused follicles are latent (counted inside the observed Primary/Secondary totals).
# Per posterior draw, expected number = μ_N * occupancy, paused fraction = paused
# occupancy / compartment occupancy (compartment total via coarse_grain). 2 vs 12 months.
# transition_fcn returns t -> W(t), so probability_flow dispatches to the ODE method.
chain_df = DataFrame(chain)
paused_state(c) = sum(k) + findfirst(==(c), findall(paused))   # hidden index of P_c
ip, is = paused_state(2), paused_state(3)

function paused_draw(t)
    samp = rand_draw(chain_df)
    μN   = extract_array(samp, "ic")[1]
    a    = probability_flow(extract_array(samp, "π_vals"),
                            transition_fcn(extract_array(samp, "rate_params")), [t])[1][1:end-1]
    (num_p  = μN * a[ip],  num_s  = μN * a[is],
     frac_p = a[ip] / dot(coarse_grain[2, :], a),
     frac_s = a[is] / dot(coarse_grain[3, :], a))
end

Ndraw   = 2_000        # reduced: each paused_draw is an ODE solve
draws2  = [paused_draw(2.0)  for _ in 1:Ndraw]
draws12 = [paused_draw(12.0) for _ in 1:Ndraw]

paused_panel(getter, xlabel) = begin
    p = density([getter(d) for d in draws2],  label="2 mo",  lw=2, fill=(0,0.15), grid=false,
                xlabel=xlabel, ylabel="Density")
    density!(p, [getter(d) for d in draws12], label="12 mo", lw=2, fill=(0,0.15))
    p
end
p_num_p  = paused_panel(d -> d.num_p,  "# paused primary")
p_num_s  = paused_panel(d -> d.num_s,  "# paused secondary")
p_frac_p = paused_panel(d -> d.frac_p, "paused fraction of primary")
p_frac_s = paused_panel(d -> d.frac_s, "paused fraction of secondary")

plot(pres_plots..., p_num_p, p_num_s, p_frac_p, p_frac_s,
     layout=(2,4), size=(1600,600), margin=5mm)
savefig("plots/PosteriorPredsPausedTimeDep.pdf")

# ===== Posterior of the time-dependent primordial residence time μ(t) =====
μ_of_t(t) = begin
    rp = extract_array(rand_draw(chain_df), "rate_params")
    μ_early, μ_late = rp[1], rp[2]
    μ_late + 2 * (μ_early - μ_late) / (1 + exp((t - 2.0) / 2.0))
end
μ_q = stack([confidence_intervals(μ_of_t, t; N_samples=4000) for t in t_vals])   # (n_q × n_t)
p_μ = credible_ribbon_plots(reshape(μ_q, size(μ_q, 1), size(μ_q, 2), 1), t_vals)[1]
plot(p_μ, xlabel="Age (months)", ylabel="Primordial mean residence μ(t)",
     title="Time-dependent primordial exit", size=(600,400), margin=4mm)
savefig("plots/PausedTimeDep_mu_of_t.pdf")

# ===== Conditional residence-time distributions (primary/secondary, active only) =====
# Erlang(k_c, k_c/μ_c) conditional on progression, excluding dormancy. μ2, μ3 live at
# rate_params[3], [4] (shifted +1 because μ1 -> μ_early, μ_late).
conditional_sojourn(c; N=50_000) =
    [rand(Erlang(k[c], extract_array(rand_draw(chain_df), "rate_params")[c+1] / k[c])) for _ in 1:N]
primary_times   = conditional_sojourn(2)
secondary_times = conditional_sojourn(3)

p_soj = density(primary_times, label="Primary (active)", lw=2, fill=(0,0.15), grid=false,
                xlabel="Active time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary (active)", lw=2, fill=(0,0.15),xlims=(0,2))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/PausedTimeDep_conditional_sojourn_times.pdf")
