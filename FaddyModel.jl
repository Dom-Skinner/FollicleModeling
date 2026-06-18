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


init_priors = [LogNormal(params_logn(1750,35_000)...),
                Truncated(Beta(3, 750), 1e-8,Inf)]
π_priors = Dirichlet(ones(3))

rate_priors = [ LogNormal(params_logn(w1_fixed,3.0)...), # set priors based on Faddy values or ballpark magnitude estimates
    LogNormal(params_logn(w2_fixed,0.008)...),
    LogNormal(params_logn(w3_fixed,0.008)...),
    Beta(4,4)]
coarse_grain_arr = I(3)    



function transition_matrix_faddy(params)
    w1, w2, w3, θ12 = params
    θ = [θ12/w1, (1-θ12)/w1, 1/w2, 1/w3]
    return [
        -(θ[1]+θ[2])  0.0      0.0       0.0  
        θ[1]      -(θ[3])      0.0       0.0
        0.0         θ[3]      -θ[4]      0.0
        θ[2]          0        θ[4]      0.0
    ]
end



################ First we fit with fixed rates, i.e. initial conditions only
@time prior_chain = sample(total_model(counts_2_month, [],[],[],
    init_priors,π_priors,rate_priors,transition_matrix_faddy,coarse_grain_arr),NUTS(),  MCMCThreads(),1000,2);
#jldsave("models/FaddyModel_fixed.jld2"; prior_chain)


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = make_sample_fun(prior_chain, transition_matrix_faddy)
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
    times_unique,init_priors,π_priors,rate_priors,transition_matrix_faddy,coarse_grain_arr),NUTS(),  MCMCThreads(),300,2);

    
sample_fun = make_sample_fun(chain, transition_matrix_faddy)

N_samples = 20_0
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
    ["μ", "p", "w1", "w2", "w3", "θ12"])
p_π = plot_π_posterior(chain, π_priors)
plot(p_π..., param_plots..., layout=(3,3), size=(1000,400), margin=4mm)
savefig("plots/Faddy_model_fitted_params.pdf")

# Presentation-quality parameter plots
pres_plots = plot_param_posteriors(chain,
    ["rate_params[2]", "rate_params[3]", "rate_params[4]"],
    rate_priors[2:4],
    [0:0.01:3, 0:0.01:2, 0:0.01:1],
    ["Avg time as Primary", "Avg time as Secondary", "Probability of reaching primary"];
    ylabel="Density")
plot(pres_plots..., layout=(1,3), size=(1000,300), margin=5mm)
savefig("plots/PosteriorPredsFaddy.pdf")

# ===== Conditional residence-time distributions =====
# Posterior-predictive distribution of the time a follicle spends in Primary /
# Secondary GIVEN it successfully progresses out (rather than dying). In the
# Faddy model there is no death from Primary/Secondary, so every transition is a
# success and these are simple exponentials. Integrates over posterior uncertainty.
primary_times   = posterior_sojourn_times(chain, transition_matrix_faddy, coarse_grain_arr, 2; N=50_000)
secondary_times = posterior_sojourn_times(chain, transition_matrix_faddy, coarse_grain_arr, 3; N=50_000)

p_soj = density(primary_times, label="Primary", lw=2, fill=(0,0.15), grid=false,
                xlabel="Time spent in compartment (months)", ylabel="Density")
density!(p_soj, secondary_times, label="Secondary", lw=2, fill=(0,0.15),xlims=(0,5))
vline!(p_soj, [mean(primary_times)],   ls=:dash, lc=1, label="")
vline!(p_soj, [mean(secondary_times)], ls=:dash, lc=2, label="")
plot(p_soj, size=(600,400), margin=4mm)
savefig("plots/Faddy_conditional_sojourn_times.pdf")