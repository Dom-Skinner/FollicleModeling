using Plots
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")

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

σ_ln =(m,v) -> sqrt(log(1 + v/m^2))
μ_ln = (m,v) -> log(m) - 0.5*log(1 + v/m^2)
#"θ" => filldist(Gamma(2.0, 0.3), 4),.
in_priors = Dict(
    "r" => LogNormal(2, 0.5), 
    "p" => Beta(2, 500), 
    "π_vals" => Dirichlet(ones(3)),
    "w1" => LogNormal(μ_ln(w1_fixed,2.0), σ_ln(w1_fixed,2.0)), # set priors based on Faddy values or ballpark magnitude estimates
    "w2" => LogNormal(μ_ln(w2_fixed,2.0), σ_ln(w2_fixed,2.0)),
    "w3" => LogNormal(μ_ln(w3_fixed,2.0), σ_ln(w3_fixed,2.0)),
    "θ12" => Uniform(0, 1)
)


################ First we fit with fixed rates, i.e. initial conditions only
@time faddy_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, [],[],[],in_priors),NUTS(),  MCMCThreads(),10000,2);
#jldsave("models/FaddyModel_fixed.jld2"; faddy_chain)
#faddy_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, [],[],[],in_priors), Prior(), 1_0000)

N_samples = 40_000
t_vals = 2:0.25:12
faddy_chain_df = DataFrame(faddy_chain)
faddy_chain_df.var"w1" .= w1_fixed
faddy_chain_df.var"w2" .= w2_fixed
faddy_chain_df.var"w3" .= w3_fixed
faddy_chain_df.var"θ12" .= θ12_fixed
quantiles_N0 = stack([confidence_intervals(t->sample_model_faddy(faddy_chain_df,t)[1],t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N1 = stack([confidence_intervals(t->sum(sample_model_faddy(faddy_chain_df,t)[2]),t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N2 = stack([confidence_intervals(t->sample_model_faddy(faddy_chain_df,t)[3],t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
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

sample_fun_prior = t -> sample_model_faddy(faddy_chain_df, t)[1:3]
mean_quantiles_post, cov_quantiles_post = chain_stats_sample(sample_fun_prior, input_data, times_vec, times_unique; 
                                  N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_post, plt_cov_post = plot_empirical_stats(mean_data, cov_data, mean_quantiles_post,
    cov_quantiles_post; ylabel_mean="Prior mean", ylabel_cov="Prior covariance")
plot(plt_mean_post, plt_cov_post)
savefig("plots/predictive_checks_fixed_rates.pdf")

# ========== Now fit everything, not just initial conditions ==========
    
@time chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, input_data, times_vec,
    times_unique,in_priors),NUTS(),  MCMCThreads(),1000,2);

#jldsave("models/FaddyModelFittedRates.jld2"; chain)

N_samples = 20_000
t_vals = 2:0.5:12
chain_df = DataFrame(chain)
quantiles_N0 = stack([confidence_intervals(t->sample_model_exact_faddy(chain_df,t)[1],t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N1 = stack([confidence_intervals(t->sum(sample_model_exact_faddy(chain_df,t)[2]),t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles_N2 = stack([confidence_intervals(t->sample_model_exact_faddy(chain_df,t)[3],t,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975],N_samples=N_samples) for t in t_vals])
quantiles = stack([quantiles_N0,quantiles_N1,quantiles_N2])


p_arr = [plot(grid=false) for _ in 1:3]
nbands = (size(quantiles, 1)-1) >> 1
for i = 1:length(p_arr), j = 1:nbands
    plot!(p_arr[i],t_vals,quantiles[nbands+1,:,i],ribbon=(quantiles[nbands+1,:,i] .- quantiles[nbands+1-j,:,i], quantiles[nbands+j+1,:,i] .- quantiles[nbands+1,:,i]),fillalpha=0.2,fc=:blue,lc=:black)
end
plot_exp_data!(p_arr...,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

plot(p_arr...,layout=(1,3),size=(1000,450), margin = 4mm)
savefig("plots/Faddy_model_fitted_rates.pdf")


sample_fun  = t->sample_model_faddy(chain_df,t)[1:3]
mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                                  N=5000, probs=[0.025, 0.5, 0.975])
plt_mean, plt_cov = plot_empirical_stats(mean_data, cov_data, mean_quantiles,
    cov_quantiles; ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
plot(plt_mean, plt_cov)
savefig("plots/predictive_checks_fitted_rates.pdf")

# prior/posterior check
p_r = histogram(chain_df.r,normalize=:pdf,xlabel="r",label="Posterior", grid=false)
plot!(p_r, 0:0.01:16,pdf(in_priors["r"] ,0:0.01:16),label = "Prior")

p_p = histogram(chain_df.p,normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!(p_p, 0:0.0002:0.04,pdf(in_priors["p"],0:0.0002:0.04),label = "Prior")


p_w1 = histogram(chain_df.w1,normalize=:pdf,xlabel="w1",label="Posterior", grid=false)
plot!(p_w1, 0:0.01:9,pdf(in_priors["w1"],0:0.01:9),label = "Prior")

p_w2 = histogram(chain_df.w2,normalize=:pdf,xlabel="w2",label="Posterior", grid=false)    
plot!(p_w2, 0:0.01:9,pdf(in_priors["w2"],0:0.01:9),label = "Prior")

p_w3 = histogram(chain_df.w3,normalize=:pdf,xlabel="w3",label="Posterior", grid=false)
plot!(p_w3, 0:0.01:9,pdf(in_priors["w3"],0:0.01:9),label = "Prior")

p_θ12 = histogram(chain_df.θ12,normalize=:pdf,xlabel="θ12",label="Posterior", grid=false)
plot!(p_θ12, 0:0.01:1,pdf(in_priors["θ12"],0:0.01:1),label = "Prior")
p = plot_π_posterior(chain,in_priors)
plot(p...,p_r,p_p,p_w1,p_w2,p_w3,p_θ12, layout = (3,3), size=(1000,400),
    margin = 4mm)


savefig("plots/Faddy_model_fitted_params.pdf")