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


# ===== Paused + queuing (Erlang) model =====
# Each observed compartment is expanded into k_c Erlang substates (exactly the
# queuing topology). Primary and Secondary additionally carry a dormant reservoir
# state P_c that resumes into their first substate S_{c,1}: an unpaused follicle
# sees the plain queuing chain, an initially paused one joins it at S_{c,1}. The
# reservoirs have no inflow (first-wave folliculogenesis leftovers), populated only
# by the 2-month initial condition.
k      = [1, 8, 8]
paused = [false, true, true]
(; transition_fcn, coarse_grain, n_hidden) = build_queuing_model(k; paused=paused)


# Ballpark timescales from the Faddy fit (converted to 1/month), used to set priors.
θ_fixed = [0.0043, 0.0017, 0.043, 0.057]*30.4
μ1_fixed = 1/(θ_fixed[1] + θ_fixed[2])
μ2_fixed = 1/θ_fixed[3]
μ3_fixed = 1/θ_fixed[4]


init_priors = [LogNormal(params_logn(1750,35_000)...),
                Truncated(Beta(3, 750), 1e-8,Inf)]
π_priors = Dirichlet(ones(n_hidden))      # full Dirichlet over all hidden states (active + paused)

# rate_params = [μ1, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary]:
#   μ_c        : mean residence time in compartment c, conditional on successful
#                progression (Erlang(k_c, k_c/μ_c) among surviving follicles)
#   θ1, θ2, θ3 : survival probabilities (θ3 weakly identified, as in QueuingModel)
#   μ_pause_*  : mean dormancy time in the Primary / Secondary paused reservoir
#                (resume rate = 1/μ_pause). Informed only via how the initial
#                paused mass drains over 4-12 months — priors matter; tune freely.
rate_priors = [ LogNormal(params_logn(μ1_fixed,3.0)...),
    LogNormal(params_logn(μ2_fixed,0.008)...),
    LogNormal(params_logn(μ3_fixed,0.008)...),
    Beta(4,4),
    Beta(4,4),
    Beta(4,4),
    Exponential(5.0),
    Exponential(5.0)]


################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),1000,2);
#jldsave("models/PausingModel_fixed.jld2"; prior_chain)


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_fcn; coarse_grain=coarse_grain)
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fixed_rates.pdf")


mean_data, cov_data = empirical_stats(input_data, times_vec)

plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
    ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean, plt_cov)
savefig("plots/Pausing_predictive_checks_fixed_rates.pdf")


# ========== Now fit everything, not just initial conditions ==========

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,init_priors,π_priors,rate_priors,transition_fcn,coarse_grain),NUTS(),  MCMCThreads(),300,2);

savefig(plot(chain), "plots/Pausing_model_chain.pdf")

sample_fun = make_sample_fun(chain, transition_fcn; coarse_grain=coarse_grain)

N_samples = 4_000
t_vals = 2:0.25:12
quantiles = compute_quantiles(sample_fun, t_vals; N_samples)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fitted_rates.pdf")


plt_mean, plt_cov = calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data)
plot(plt_mean, plt_cov)
savefig("plots/Pausing_predictive_checks_fitted_rates.pdf")

# prior/posterior check
param_plots = plot_param_posteriors(chain,
    ["ic[1]", "ic[2]", "rate_params[1]", "rate_params[2]", "rate_params[3]",
     "rate_params[4]", "rate_params[5]", "rate_params[6]", "rate_params[7]", "rate_params[8]"],
    [init_priors..., rate_priors...],
    [1000:5:2500, 0:0.0001:0.015, 0:0.01:9, 0:0.01:3, 0:0.01:2,
     0:0.01:1, 0:0.01:1, 0:0.01:1, 0:0.1:20, 0:0.1:20],
    ["μ_N", "p", "μ1", "μ2", "μ3", "θ1", "θ2", "θ3", "μ_pause_primary", "μ_pause_secondary"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., size=(1600,900), margin=4mm)
savefig("plots/Pausing_model_fitted_params.pdf")


pres_plots = plot_param_posteriors(chain,
    ["rate_params[2]", "rate_params[3]", "rate_params[7]", "rate_params[8]"],
    vcat(rate_priors[2:3], rate_priors[7:8]),
    [0:0.01:1.2, 0:0.01:1.2, 0:0.1:30, 0:0.1:30],
    ["Avg time as Primary", "Avg time as Secondary", "Avg time as paused primary", "Avg time as paused secondary"];
    ylabel="Density")
# ----- Paused-follicle summaries: number and within-compartment fraction -----
# Paused follicles are latent (counted inside the observed Primary/Secondary
# totals). Per posterior draw, expected number = μ_N * occupancy and the paused
# fraction = paused occupancy / compartment occupancy (compartment total via
# coarse_grain). Compared at 2 vs 12 months.
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

Ndraw   = 5_000
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
savefig("plots/PosteriorPredsPausing.pdf")

# ===== Conditional residence-time distributions =====
# Active time a follicle spends in Primary / Secondary GIVEN it successfully
# progresses out (rather than dying), EXCLUDING any dormancy. The reservoir only
# adds pause *before* S_{c,1}, so the active residence is the same Erlang as in the
# queuing model: Erlang(k_c, k_c/μ_c), independent of the death rate. We sample it
# directly, one draw per posterior sample.
chain_df = DataFrame(chain)
conditional_sojourn(c; N=50_000) =
    [rand(Erlang(k[c], extract_array(rand_draw(chain_df), "rate_params")[c] / k[c])) for _ in 1:N]
primary_times   = conditional_sojourn(2)
secondary_times = conditional_sojourn(3)

p_soj = density(primary_times, label="Primary (active)", lw=2, fill=(0,0.15), grid=false,
                xlabel="Active time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary (active)", lw=2, fill=(0,0.15),xlims=(0,2))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/Pausing_conditional_sojourn_times.pdf")
