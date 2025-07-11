using Plots
using StatsPlots
using Measures
using Random
using JLD2

include("Models.jl")
include("Utils.jl")
include("PlotUtils.jl")


counts_2_month,_,_,_,_ = extract_data() # here only care about 2 month data

in_priors_initial = Dict(
    "μ" => LogNormal(params_logn(1750,35_000)...), 
    "p" => Beta(3, 750), 
    "π_vals" => Dirichlet([2.5,0.5,0.5]), # we know 1 is more likely so slightly weight towards that.
    # none of the ones below are used in the initial fit
    "w1" => LogNormal(1, 1),
    "w2" => LogNormal(1, 1),
    "w3" => LogNormal(1, 1),
    "θ12" => Uniform(0, 1)
)

# ========= First do prior predictive checks
prior_predictive_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, [],[],[],in_priors_initial), Prior(), 10_000)

ppc_df = DataFrame(prior_predictive_chain)

mean_data,cov_data = empirical_stats(counts_2_month,ones(Int64,size(counts_2_month,1)))

sample_fun_prior = t -> sample_model_faddy(ppc_df, t)[1:3]
mean_quantiles_prior, cov_quantiles_prior = chain_stats_sample(sample_fun_prior, counts_2_month, 
                            ones(Int64,size(counts_2_month,1)), [2.0]; N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_prior, plt_cov_prior = plot_empirical_stats(mean_data, cov_data, mean_quantiles_prior, 
    cov_quantiles_prior; ylabel_mean="Prior mean", ylabel_cov="Prior covariance", logscale=false)
plot(plt_mean_prior, plt_cov_prior)

# ========= Now we actually fit

initial_condition_chain = sample(faddy_model(sum(counts_2_month,dims=2),counts_2_month, 
    [],[],[],in_priors_initial),NUTS(),  MCMCThreads(),10_000,2);
#summarize(initial_condition_chain)
#plot(initial_condition_chain) # checks on convergence

init_cond_df = DataFrame(initial_condition_chain)

sample_fun_post = t -> sample_model_faddy(init_cond_df, t)[1:3]
mean_quantiles, cov_quantiles = chain_stats_sample(sample_fun_post, counts_2_month, 
        ones(Int64,size(counts_2_month,1)), [2.0]; N=5000, probs=[0.025, 0.5, 0.975])

plt_mean_post, plt_cov_post = plot_empirical_stats(mean_data, cov_data, mean_quantiles,
    cov_quantiles; ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
plot(plt_mean_post, plt_cov_post)

plot(plt_mean_prior,plt_cov_prior,plt_mean_post,plt_cov_post,
    layout = (2,2), size=(500,500), margin = 4mm)
savefig("plots/predictive_checks_initial.pdf")

# ========= Compare prior and posterior
# prior/posterior check
p_μ = histogram(init_cond_df.μ,normalize=:pdf,xlabel="μ",label="Posterior", grid=false)
plot!(p_μ, 1000:5:2500,pdf(in_priors_initial["μ"] ,1000:5:2500),label = "Prior")

p_p = histogram(init_cond_df.p,normalize=:pdf,xlabel="p",label="Posterior", grid=false)
plot!(p_p, 0:0.0001:0.025,pdf(in_priors_initial["p"],0:0.0001:0.025),label = "Prior")

p = plot_π_posterior(initial_condition_chain,in_priors_initial)
plot(p...,p_μ,p_p, layout = (2,3), size=(1000,400),
    margin = 4mm)
savefig("plots/initial_condition_posterior.pdf")


# Just to highlight how non-Poisson the original data is
histogram(counts_2_month[:,1],normalize=:pdf,bins=10,label="Data at 2 months",xlabel="Primordial follicles",
    ylabel="Density", grid=false)
plot!(0:3000,pdf(Poisson(mean(counts_2_month[:,1])),0:3000),label="Poisson", grid=false)
plot!(0:3000,pdf(NegativeBinomial(median( init_cond_df.p .*init_cond_df.μ ./ (1 .-init_cond_df.p) ),
    median(init_cond_df.p)),0:3000),label="Negative Binomial", grid=false)
savefig("plots/initial_condition_data.pdf")    