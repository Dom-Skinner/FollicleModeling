using Plots
using StatsPlots


function plot_exp_data!(p1,p2,p3,counts_2_month,counts_4_month,counts_6_month,counts_9_month,counts_12_month)

    plot!(p1, 2*ones(size(counts_2_month,1)), counts_2_month[:,1], seriestype = :scatter, 
    label = "2 month", legend = false, title = "Primordial", xlabel = "Age (months)", ylabel = "Counts")
    plot!(p1, 4*ones(size(counts_4_month,1)), counts_4_month[:,1], seriestype = :scatter, label = "4 month")
    plot!(p1, 6*ones(size(counts_6_month,1)), counts_6_month[:,1], seriestype = :scatter, label = "6 month")
    plot!(p1, 9*ones(size(counts_9_month,1)), counts_9_month[:,1], seriestype = :scatter, label = "9 month")
    plot!(p1, 12*ones(size(counts_12_month,1)), counts_12_month[:,1], seriestype = :scatter, label = "12 month")



    plot!(p2, 2*ones(size(counts_2_month,1)), counts_2_month[:,2], seriestype = :scatter, label = "2 month", 
    legend = false, title = "Primary", xlabel = "Age (months)", ylabel = "Counts")
    plot!(p2, 4*ones(size(counts_4_month,1)), counts_4_month[:,2], seriestype = :scatter, label = "4 month")
    plot!(p2, 6*ones(size(counts_6_month,1)), counts_6_month[:,2], seriestype = :scatter, label = "6 month")
    plot!(p2, 9*ones(size(counts_9_month,1)), counts_9_month[:,2], seriestype = :scatter, label = "9 month")
    plot!(p2, 12*ones(size(counts_12_month,1)), counts_12_month[:,2], seriestype = :scatter, label = "12 month")

    plot!(p3, 2*ones(size(counts_2_month,1)), counts_2_month[:,3], seriestype = :scatter, label = "2 month", 
    legend = false, title = "Secondary", xlabel = "Age (months)", ylabel = "Counts")
    plot!(p3, 4*ones(size(counts_4_month,1)), counts_4_month[:,3], seriestype = :scatter, label = "4 month")
    plot!(p3, 6*ones(size(counts_6_month,1)), counts_6_month[:,3], seriestype = :scatter, label = "6 month")
    plot!(p3, 9*ones(size(counts_9_month,1)), counts_9_month[:,3], seriestype = :scatter, label = "9 month")
    plot!(p3, 12*ones(size(counts_12_month,1)), counts_12_month[:,3], seriestype = :scatter, label = "12 month")
end


function plot_π_posterior(chain,in_priors)
    # plot posterior for each π_k
    # assumes that prior was Dirichlet
    α = in_priors["π_vals"].alpha
    p = [plot() for _ in 1:length(α)]
    for k in 1:length(α)
        # extract posterior samples for π_k
        samples = Float64.(vec(chain[ Symbol("π_vals[$k]")]))
        # histogram of posterior
        histogram!(p[k], samples;
            normalize = :pdf,
            label="Posterior",
            title="π[$k]",
            legend=:topright)
        # overlay prior marginal Beta(α_k, ∑α₋ₖ)
        # because Dirichlet marginal for π_k ~ Beta(α_k, ∑_{j≠k}α_j)
        a, b = α[k], sum(α) - α[k]
        xs = range(0, 1, length=200)
        plot!(p[k], xs, pdf.(Beta(a,b), xs), label="Prior")
    end
    return p
end


function plot_empirical_stats(mean_data, cov_data, mean_quantiles_model,cov_quantiles_model;   
        ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance",logscale=true)
    
    @assert size(mean_quantiles_model,3) ==3
    @assert size(cov_quantiles_model,4) ==3


    plt_mean = scatter(vec(vcat(mean_data...)), vec(mean_quantiles_model[:,:,2]), 
        yerr = (vec(mean_quantiles_model[:,:,2]) .- vec(mean_quantiles_model[:,:,1]), 
                    vec(mean_quantiles_model[:,:,3]) .- vec(mean_quantiles_model[:,:,2])),
        xlabel="Empirical mean", ylabel=ylabel_mean, xaxis=:log10, yaxis=:log10,label=false)                                  

    plot!(plt_mean,[1e-6, 1e6], [1e-6, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_mean),
        xlims=xlims(plt_mean), grid=false)

    
    if logscale
        plt_cov = scatter(vec(vcat(cov_data...)), vec(cov_quantiles_model[:,:,:,2]), 
        yerr = (vec(cov_quantiles_model[:,:,:,2]) .- vec(cov_quantiles_model[:,:,:,1]), 
                    vec(cov_quantiles_model[:,:,:,3]) .- vec(cov_quantiles_model[:,:,:,2])),
        xlabel="Empirical covariance", ylabel=ylabel_cov, label=false,xaxis=:log10, yaxis=:log10)    
        plot!(plt_cov,[1e-6, 1e6], [1e-6, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_cov),xlims=xlims(plt_cov), grid=false)
    else
        plt_cov = scatter(vec(vcat(cov_data...)), vec(cov_quantiles_model[:,:,:,2]), 
        yerr = (vec(cov_quantiles_model[:,:,:,2]) .- vec(cov_quantiles_model[:,:,:,1]), 
                    vec(cov_quantiles_model[:,:,:,3]) .- vec(cov_quantiles_model[:,:,:,2])),
        xlabel="Empirical covariance", ylabel=ylabel_cov, label=false)    
        plot!(plt_cov,[-1e6, 1e6], [-1e6, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_cov),xlims=xlims(plt_cov), grid=false)
    end

    return plt_mean, plt_cov
end