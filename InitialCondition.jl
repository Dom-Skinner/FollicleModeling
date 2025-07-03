using Plots
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")


counts_2_month,_,_,_,_ = extract_data() # here only care about 2 month data


in_priors_initial = Dict(
    "r" => LogNormal(2, 0.5), 
    "p" => Beta(2, 500), 
    "π_vals" => Dirichlet([2.5,0.5,0.5]), # we know 1 is more likely so slightly weight towards that.
    # none of the ones below are used in the initial fit
    "w1" => LogNormal(1, 1),
    "w2" => LogNormal(1, 1),
    "w3" => LogNormal(1, 1),
    "θ12" => Uniform(0, 1)
)

prior_predictive_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, [],[],[],in_priors_initial), Prior(), 10_000)

ppc_df = DataFrame(prior_predictive_chain)

mean_data,cov_data = empirical_stats(counts_2_month,ones(Int64,size(counts_2_month,1)))

sample_fun_prior = t -> sample_model_faddy(ppc_df, t)[1:3]
mean_quantiles_prior, cov_quantiles_prior = chain_stats_sample(sample_fun_prior, counts_2_month, ones(Int64,size(counts_2_month,1)), [2.0]; 
                                  N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_prior = scatter(vec(vcat(mean_data...)), vec(mean_quantiles_prior[:,:,2]), yerr = (vec(mean_quantiles_prior[:,:,2]) .- vec(mean_quantiles_prior[:,:,1]), vec(mean_quantiles_prior[:,:,3]) .- vec(mean_quantiles_prior[:,:,2])),
    xlabel="Empirical mean", ylabel="Prior mean", xaxis=:log10, yaxis=:log10,label=false)                                  
plot!(plt_mean_prior,[1, 1e6], [1, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_mean_prior),
    xlims=xlims(plt_mean_prior), grid=false)

plt_cov_prior = scatter(vec(vcat(cov_data...)), vec(cov_quantiles_prior[:,:,:,2]), yerr = (vec(cov_quantiles_prior[:,:,:,2]) .- vec(cov_quantiles_prior[:,:,:,1]), vec(cov_quantiles_prior[:,:,:,3]) .- vec(cov_quantiles_prior[:,:,:,2])),
    xlabel="Empirical covariance", ylabel="Prior covariance", label=false)    
plot!(plt_cov_prior,[1, 1e6], [1, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_cov_prior),xlims=xlims(plt_cov_prior), grid=false)
plot(plt_mean_prior, plt_cov_prior)


# ========= Now we actually fit

initial_condition_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, [],[],[],in_priors_initial),NUTS(),  MCMCThreads(),10_000,2);
#summarize(initial_condition_chain)
#plot(initial_condition_chain)

init_cond_df = DataFrame(initial_condition_chain)

sample_fun_post = t -> sample_model_faddy(init_cond_df, t)[1:3]
mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun_post, counts_2_month, ones(Int64,size(counts_2_month,1)), [2.0]; 
                                  N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_post = scatter(vec(vcat(mean_data...)), vec(mean_quantiles[:,:,2]), yerr = (vec(mean_quantiles[:,:,2]) .- vec(mean_quantiles[:,:,1]), vec(mean_quantiles[:,:,3]) .- vec(mean_quantiles[:,:,2])),
    xlabel="Empirical mean", ylabel="Posterior mean", xaxis=:log10, yaxis=:log10,label=false)                                  
plot!(plt_mean_post,[1, 1e6], [1, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_mean_post),
    xlims=xlims(plt_mean_post), grid=false)

plot_cov_post = scatter(vec(vcat(cov_data...)), vec(cov_quantiles[:,:,:,2]), yerr = (vec(cov_quantiles[:,:,:,2]) .- vec(cov_quantiles[:,:,:,1]), vec(cov_quantiles[:,:,:,3]) .- vec(cov_quantiles[:,:,:,2])),
    xlabel="Empirical covariance", ylabel="Posterior covariance", label=false,yaxis=:log10,xaxis=:log10)    
plot!(plot_cov_post,[1, 1e6], [1, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plot_cov_post),xlims=xlims(plot_cov_post), grid=false)
plot(plt_mean_post, plot_cov_post)


plot(plt_mean_prior,plt_cov_prior,plt_mean_post,plot_cov_post,
    layout = (2,2), size=(600,600), margin = 4mm)


# prior/posterior check
histogram(init_cond_df.r,normalize=:pdf,xlabel="r",label="Posterior", grid=false)
plot!( 0:0.01:8,pdf(in_priors_initial["r"] ,0:0.01:8),label = "Prior")

histogram(init_cond_df.p,normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!( 0:0.0002:0.04,pdf(in_priors_initial["p"],0:0.0002:0.04),label = "Prior")


p = plot_π_posterior(initial_condition_chain,in_priors_initial)
plot(p..., layout = (1,3), size=(1000,400))