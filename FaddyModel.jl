using Plots
using AugmentedGPLikelihoods
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


innit_priors = [LogNormal(params_logn(1750,35_000)...),
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
@time faddy_chain = sample(total_model(counts_2_month, [],[],[],
    innit_priors,π_priors,rate_priors,transition_matrix_faddy,coarse_grain_arr),NUTS(),  MCMCThreads(),1000,2);
#jldsave("models/FaddyModel_fixed.jld2"; faddy_chain)


N_samples = 40_0
t_vals = 2:0.25:12

sample_fun = t -> sample_model(faddy_chain,t, transition_matrix_faddy)
q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975]
quantiles_N0 = stack([confidence_intervals(t->sample_fun(t)[1],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles_N1 = stack([confidence_intervals(t->sample_fun(t)[2],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles_N2 = stack([confidence_intervals(t->sample_fun(t)[3],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles = stack([quantiles_N0,quantiles_N1,quantiles_N2])

p_arr = [plot(grid=false) for _ in 1:3]
nbands = (size(quantiles, 1)-1) >> 1
for i = 1:length(p_arr), j = 1:nbands
    plot!(p_arr[i],t_vals,quantiles[nbands+1,:,i],ribbon=(quantiles[nbands+1,:,i] .- quantiles[nbands+1-j,:,i], quantiles[nbands+j+1,:,i] .- quantiles[nbands+1,:,i]),fillalpha=0.2,fc=:blue,lc=:black)
end
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Faddy_model_fixed_rates.pdf")




mean_data,cov_data = empirical_stats(input_data,times_vec)

mean_quantiles_post, cov_quantiles_post = chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                                  N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_post, plt_cov_post = plot_empirical_stats(mean_data, cov_data, mean_quantiles_post,
    cov_quantiles_post; ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean_post, plt_cov_post)
savefig("plots/predictive_checks_fixed_rates.pdf")

# ========== Now fit everything, not just initial conditions ==========
    

@time chain = sample(total_model(counts_2_month, Int64.(input_data), times_vec,
    times_unique,innit_priors,π_priors,rate_priors,transition_matrix_faddy,coarse_grain_arr),NUTS(),  MCMCThreads(),300,2);

    
sample_fun = t -> sample_model(chain,t, transition_matrix_faddy)

N_samples = 20_0
t_vals = 2:0.5:12
q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975]
quantiles_N0 = stack([confidence_intervals(t->sample_fun(t)[1],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles_N1 = stack([confidence_intervals(t->sample_fun(t)[2],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles_N2 = stack([confidence_intervals(t->sample_fun(t)[3],t,q_levels=q_levels, N_samples=N_samples) for t in t_vals])
quantiles = stack([quantiles_N0,quantiles_N1,quantiles_N2])

p_arr = [plot(grid=false) for _ in 1:3]
nbands = (size(quantiles, 1)-1) >> 1
for i = 1:length(p_arr), j = 1:nbands
    plot!(p_arr[i],t_vals,quantiles[nbands+1,:,i],ribbon=(quantiles[nbands+1,:,i] .- quantiles[nbands+1-j,:,i], quantiles[nbands+j+1,:,i] .- quantiles[nbands+1,:,i]),fillalpha=0.2,fc=:blue,lc=:black)
end
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Faddy_model_fitted_rates.pdf")


mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                                  N=5000, probs=[0.025, 0.5, 0.975])
plt_mean, plt_cov = plot_empirical_stats(mean_data, cov_data, mean_quantiles,
    cov_quantiles; ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
plot(plt_mean, plt_cov)
savefig("plots/predictive_checks_fitted_rates.pdf")

# prior/posterior check
p_μ = histogram(vec(chain["inpriors[1]"]),normalize=:pdf,xlabel="μ",label="Posterior", grid=false)
plot!(p_μ, 1000:5:2500,pdf(innit_priors[1] ,1000:5:2500),label = "Prior")

p_p = histogram(vec(chain["inpriors[2]"]),normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!(p_p, 0:0.0001:0.015,pdf(innit_priors[2],0:0.0001:0.015),label = "Prior")



p_w1 = histogram(vec(chain["rate_params[1]"]),normalize=:pdf,xlabel="w1",label="Posterior", grid=false)
plot!(p_w1, 0:0.01:9,pdf(rate_priors[1],0:0.01:9),label = "Prior")

p_w2 = histogram(vec(chain["rate_params[2]"]),normalize=:pdf,xlabel="w2",label="Posterior", grid=false)    
plot!(p_w2, 0:0.01:3,pdf(rate_priors[2],0:0.01:3),label = "Prior")

p_w3 = histogram(vec(chain["rate_params[3]"]),normalize=:pdf,xlabel="w3",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:2,pdf(rate_priors[3],0:0.01:2),label = "Prior")

p_θ12 = histogram(vec(chain["rate_params[4]"]),normalize=:pdf,xlabel="θ12",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(rate_priors[4],0:0.01:1),label = "Prior")

p = plot_π_posterior(chain,π_priors)

plot(p...,p_μ,p_p,p_w1,p_w2,p_w3,p_θ12, layout = (3,3), size=(1000,400),
    margin = 4mm)


savefig("plots/Faddy_model_fitted_params.pdf")


#[rand(Exponential(t)) for t in chain_df.w2]
# Make a plot for presentation
p_w2 = histogram(vec(chain["rate_params[2]"]),normalize=:pdf,xlabel="Avg time as Primary",label="Posterior", grid=false)    
plot!(p_w2, 0:0.01:3,pdf(rate_priors[2],0:0.01:3),label = "Prior",ylabel="Density")

p_w3 = histogram(vec(chain["rate_params[3]"]),normalize=:pdf,xlabel="Avg time as Secondary",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:2,pdf(rate_priors[3],0:0.01:2),label = "Prior",ylabel="Density")

p_θ12 = histogram(vec(chain["rate_params[4]"]),normalize=:pdf,xlabel="Probability of reaching primary",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(rate_priors[4],0:0.01:1),label = "Prior",ylabel="Density")

plot(p_w2,p_w3,p_θ12, layout = (1,3), size=(1000,300),
    margin = 5mm)
savefig("plots/PosteriorPredsFaddy.pdf")