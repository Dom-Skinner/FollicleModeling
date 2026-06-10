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
q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975]
quantiles = compute_quantiles(sample_fun_prior, t_vals; N_samples, q_levels)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fixed_rates.pdf")

mean_data,cov_data = empirical_stats(input_data,times_vec)

mean_quantiles_prior, cov_quantiles_prior = chain_stats_sample(sample_fun_prior, input_data, times_vec, times_unique;
    N=5000, probs=[0.025, 0.5, 0.975])
plt_mean_prior, plt_cov_prior = plot_empirical_stats(mean_data, cov_data, mean_quantiles_prior,
    cov_quantiles_prior; ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean_prior, plt_cov_prior)
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
q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975]
quantiles = compute_quantiles(sample_fun, t_vals; N_samples, q_levels)

p_arr = credible_ribbon_plots(quantiles, t_vals)
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fitted_rates.pdf")

# Posterior predictive calibration (mean and covariance)
mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun, input_data, times_vec, times_unique;
                                  N=5000, probs=[0.025, 0.5, 0.975])
plt_mean, plt_cov = plot_empirical_stats(mean_data, cov_data, mean_quantiles,
    cov_quantiles; ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
plot(plt_mean, plt_cov)
savefig("plots/pausing_predictive_checks_fitted_rates.pdf")

# Prior/posterior comparison for each parameter
p_μ = histogram(vec(chain["ic[1]"]),normalize=:pdf,xlabel="μ",label="Posterior", grid=false)
plot!(p_μ, 1000:5:2500,pdf(init_priors[1],1000:5:2500),label = "Prior")

p_p = histogram(vec(chain["ic[2]"]),normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!(p_p, 0:0.0001:0.015,pdf(init_priors[2],0:0.0001:0.015),label = "Prior")

p_w1 = histogram(vec(chain["rate_params[1]"]),normalize=:pdf,xlabel="w1",label="Posterior", grid=false)
plot!(p_w1, 0:0.01:9,pdf(rate_priors_paused[1],0:0.01:9),label = "Prior")

p_w2 = histogram(vec(chain["rate_params[2]"]),normalize=:pdf,xlabel="w2",label="Posterior", grid=false)
plot!(p_w2, 0:0.01:3,pdf(rate_priors_paused[2],0:0.01:3),label = "Prior")

p_w3 = histogram(vec(chain["rate_params[3]"]),normalize=:pdf,xlabel="w3",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:2,pdf(rate_priors_paused[3],0:0.01:2),label = "Prior")

p_θ12 = histogram(vec(chain["rate_params[4]"]),normalize=:pdf,xlabel="θ12",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(rate_priors_paused[4],0:0.01:1),label = "Prior")

p_θ34 = histogram(vec(chain["rate_params[5]"]),normalize=:pdf,xlabel="θ34",label="Posterior", grid=false)
plot!(p_θ34, 0:0.01:1,pdf(rate_priors_paused[5],0:0.01:1),label = "Prior")

p_θ6 = histogram(vec(chain["rate_params[6]"]),normalize=:pdf,xlabel="θ6",label="Posterior", grid=false)
plot!(p_θ6, 0:0.01:1,pdf(rate_priors_paused[6],0:0.01:1),label = "Prior")

p_θ7 = histogram(vec(chain["rate_params[7]"]),normalize=:pdf,xlabel="θ7",label="Posterior", grid=false)
plot!(p_θ7, 0:0.01:1,pdf(rate_priors_paused[7],0:0.01:1),label = "Prior")

p_π = plot_π_posterior(chain, Dirichlet(ones(5)))
plot(p_π...,p_μ,p_p,p_w1,p_w2,p_w3,p_θ12,p_θ34,p_θ6,p_θ7, layout = (4,4), size=(1000,600),
    margin = 4mm)
savefig("plots/pausing_model_prior_posterior.pdf")


# Presentation-quality parameter plots
p_w2 = histogram(vec(chain["rate_params[2]"]),normalize=:pdf,xlabel="Avg time as Primary (months)",label="Posterior", grid=false)
plot!(p_w2, 0:0.01:3,pdf(rate_priors_paused[2],0:0.01:3),label = "Prior",ylabel="Density")

p_w3 = histogram(vec(chain["rate_params[3]"]),normalize=:pdf,xlabel="Avg time as Secondary (months)",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:2,pdf(rate_priors_paused[3],0:0.01:2),label = "Prior",ylabel="Density")

p_θ12 = histogram(vec(chain["rate_params[4]"]),normalize=:pdf,xlabel="Probability of reaching primary",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(rate_priors_paused[4],0:0.01:1),label = "Prior",ylabel="Density")

p_θ34 = histogram(vec(chain["rate_params[5]"]),normalize=:pdf,xlabel="Probability of reaching \n secondary from primary",label="Posterior", grid=false)
plot!(p_θ34, 0:0.01:1,pdf(rate_priors_paused[5],0:0.01:1),label = "Prior",ylabel="Density")

plot(p_w2,p_w3,p_θ12,p_θ34, layout = (1,4), size=(1000,300),
    margin = 6mm,xguidefontsize=8,guidefontsize=8)
savefig("plots/PosteriorPredsPaused.pdf")
