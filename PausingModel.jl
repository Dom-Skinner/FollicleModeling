using Plots
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")
include("PlotUtils.jl")

counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month = extract_data()
input_data = Float64.(vcat(counts_4_month,counts_6_month,counts_9_month,counts_12_month))
input_times = vcat(4*ones(size(counts_4_month,1)),6*ones(size(counts_6_month,1)),9*ones(size(counts_9_month,1)),12*ones(size(counts_12_month,1)))

times_unique = unique(input_times)
times_vec = [findfirst(isequal(t), times_unique) for t in input_times]


θ_fixed = [0.0043, 0.0017, 0.043, 0.057]*30.4 # fixed values from Faddy converted into 1/month
w1_fixed = 1/(θ_fixed[1] + θ_fixed[2])
w2_fixed = 1/θ_fixed[3]
w3_fixed = 1/θ_fixed[4]
θ12_fixed = θ_fixed[1]/(θ_fixed[1] + θ_fixed[2])

in_priors = Dict(
    "μ" => LogNormal(params_logn(1750,35_000)...), 
    "p" => Beta(3, 750), 
    "π_vals" => Dirichlet(ones(5)),
    "w1" => LogNormal(params_logn(w1_fixed,3.0)...), # set priors based on Faddy values or ballpark magnitude estimates
    "w2" => LogNormal(params_logn(w2_fixed,0.005)...),
    "w3" => LogNormal(params_logn(w3_fixed,0.005)...),
    "θ12" => Beta(4,4),
    "θ34" => Beta(4,4),
    "θ6" => Gamma(2.0, 0.3), 
    "θ7" => Gamma(2.0, 0.3),
)


# Fit the model

@time chain = sample(pausing_model(sum(counts_2_month,dims=2),counts_2_month, input_data, times_vec,
    times_unique,in_priors),NUTS(),  MCMCThreads(),2000,2);


# Plot credible intervals for the model

N_samples = 20_000
t_vals = 2:0.5:12
chain_df = DataFrame(chain)
quantiles_N0 = stack([confidence_intervals(t->sample_model_paused(chain_df,t)[1],t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N1 = stack([confidence_intervals(t->sum(sample_model_paused(chain_df,t)[2:3]),t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N2 = stack([confidence_intervals(t->sum(sample_model_paused(chain_df,t)[4:5]),t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles = stack([quantiles_N0,quantiles_N1,quantiles_N2])


p_arr = [plot(grid=false) for _ in 1:3]
nbands = (size(quantiles, 1)-1) >> 1
for i = 1:length(p_arr), j = 1:nbands
    plot!(p_arr[i],t_vals,quantiles[nbands+1,:,i],ribbon=(quantiles[nbands+1,:,i] .- quantiles[nbands+1-j,:,i], quantiles[nbands+j+1,:,i] .- quantiles[nbands+1,:,i]),fillalpha=0.2,fc=:blue,lc=:black)
end
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Pausing_model_fitted_rates.pdf")

# Compare posterior statistics
mean_data,cov_data = empirical_stats(input_data,times_vec)
hidden_states = x -> [x[1], x[2] + x[3], x[4] + x[5]]
sample_fun  = t-> hidden_states(sample_model_exact_paused(chain_df,t))
mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                                  N=5000, probs=[0.025, 0.5, 0.975])
plt_mean, plt_cov = plot_empirical_stats(mean_data, cov_data, mean_quantiles,
    cov_quantiles; ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
plot(plt_mean, plt_cov)
savefig("plots/pausing_predictive_checks_fitted_rates.pdf")

# prior/posterior check for parameters

p_μ = histogram(chain_df.μ,normalize=:pdf,xlabel="μ",label="Posterior", grid=false)
plot!(p_μ, 1000:5:2500,pdf(in_priors["μ"] ,1000:5:2500),label = "Prior")

p_p = histogram(chain_df.p,normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!(p_p, 0:0.0001:0.015,pdf(in_priors["p"],0:0.0001:0.015),label = "Prior")


p_w1 = histogram(chain_df.w1,normalize=:pdf,xlabel="w1",label="Posterior", grid=false)
plot!(p_w1, 0:0.01:9,pdf(in_priors["w1"],0:0.01:9),label = "Prior")

p_w2 = histogram(chain_df.w2,normalize=:pdf,xlabel="w2",label="Posterior", grid=false)    
plot!(p_w2, 0:0.01:3,pdf(in_priors["w2"],0:0.01:3),label = "Prior")

p_w3 = histogram(chain_df.w3,normalize=:pdf,xlabel="w3",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:2,pdf(in_priors["w3"],0:0.01:2),label = "Prior")

p_θ12 = histogram(chain_df.θ12,normalize=:pdf,xlabel="θ12",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(in_priors["θ12"],0:0.01:1),label = "Prior")

p_θ34 = histogram(chain_df.θ34,normalize=:pdf,xlabel="θ34",label="Posterior", grid=false)
plot!(p_θ34, 0:0.01:1,pdf(in_priors["θ34"],0:0.01:1),label = "Prior")

p_θ6 = histogram(chain_df.θ6,normalize=:pdf,xlabel="θ6",label="Posterior", grid=false)
plot!(p_θ6, 0:0.01:1,pdf(in_priors["θ6"],0:0.01:1),label = "Prior")

p_θ7 = histogram(chain_df.θ7,normalize=:pdf,xlabel="θ7",label="Posterior", grid=false)
plot!(p_θ7, 0:0.01:1,pdf(in_priors["θ7"],0:0.01:1),label = "Prior")

p = plot_π_posterior(chain,in_priors)
plot(p...,p_μ,p_p,p_w1,p_w2,p_w3,p_θ12,p_θ34,p_θ6,p_θ7, layout = (4,4), size=(1000,600),
    margin = 4mm)
savefig("plots/pausing_model_prior_posterior.pdf")