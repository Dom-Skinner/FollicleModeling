using Plots
using StatsPlots


# Returns an array of n_obs initialized plots, each with nested credible ribbon bands.
# quantiles is (n_q_levels × n_t × n_obs); t_vals is the matching time grid.
function credible_ribbon_plots(quantiles, t_vals)
    n_obs = size(quantiles, 3)
    p_arr = [plot(grid=false) for _ in 1:n_obs]
    nbands = (size(quantiles, 1)-1) >> 1
    for i in 1:n_obs, j in 1:nbands
        plot!(p_arr[i], t_vals, quantiles[nbands+1,:,i],
              ribbon=(quantiles[nbands+1,:,i] .- quantiles[nbands+1-j,:,i],
                      quantiles[nbands+j+1,:,i] .- quantiles[nbands+1,:,i]),
              fillalpha=0.2, fc=:blue, lc=:black)
    end
    return p_arr
end


# Returns one histogram+prior-overlay plot per parameter.
# param_keys: chain variable names e.g. ["ic[1]", "rate_params[1]", ...]
# priors:     matching distribution objects
# x_ranges:   matching plot ranges
# labels:     x-axis labels; pass ylabel="Density" for presentation plots
function plot_param_posteriors(chain, param_keys, priors, x_ranges, labels; ylabel="")
    plots = []
    for (key, prior, xr, lab) in zip(param_keys, priors, x_ranges, labels)
        p = histogram(vec(chain[key]); normalize=:pdf, xlabel=lab, label="Posterior", grid=false)
        isempty(ylabel) || plot!(p; ylabel=ylabel)
        plot!(p, collect(xr), pdf.(prior, collect(xr)); label="Prior")
        push!(plots, p)
    end
    return plots
end

# Runs chain_stats_sample and returns (plt_mean, plt_cov) calibration scatter plots.
# mean_data / cov_data should be pre-computed with empirical_stats() and reused across calls.
# Caller is responsible for plot() and savefig().
function calibration_plots(sample_fun, input_data, times_vec, times_unique, mean_data, cov_data;
        N=5000, probs=[0.025, 0.5, 0.975],
        ylabel_mean="Posterior mean", ylabel_cov="Posterior covariance")
    mean_q, cov_q = chain_stats_sample(sample_fun, input_data, times_vec, times_unique; N, probs)
    return plot_empirical_stats(mean_data, cov_data, mean_q, cov_q; ylabel_mean, ylabel_cov)
end


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


function plot_π_posterior(chain,π_priors)
    # plot posterior for each π_k
    # assumes that prior was Dirichlet
    α = π_priors.alpha
    p = [plot() for _ in 1:length(α)]
    for k in 1:length(α)
        # extract posterior samples for π_k
        samples = Float64.(vec(chain[ "π_vals[$k]"]))
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

    # Diagonal entries of the covariance matrix are variances, off-diagonal are
    # covariances; colour and label the two so they can be told apart.
    c_val    = vec(vcat([ones(Int64,3) .+ I(3) for _ in 1:length(cov_data)]...))
    xcov     = vec(vcat(cov_data...))
    ycov     = vec(cov_quantiles_model[:,:,:,2])
    var_mask = c_val .== 2          # diagonal     -> variances
    cov_mask = c_val .== 1          # off-diagonal -> covariances
    yerr_cov = (vec(cov_quantiles_model[:,:,:,2]) .- vec(cov_quantiles_model[:,:,:,1]),
                vec(cov_quantiles_model[:,:,:,3]) .- vec(cov_quantiles_model[:,:,:,2]))
    if logscale
        plt_cov = scatter(xcov, ycov, yerr=yerr_cov, mc=:lightgray, msc=:lightgray,
                    xlabel="Empirical covariance", ylabel=ylabel_cov, label=false,
                    xaxis=:log10, yaxis=:log10, legend=:topleft)
        scatter!(plt_cov, xcov[var_mask], ycov[var_mask], c=2, label="Variance (diagonal)")
        scatter!(plt_cov, xcov[cov_mask], ycov[cov_mask], c=1, label="Covariance (off-diagonal)")
        plot!(plt_cov,[1e-6, 1e6], [1e-6, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_cov),xlims=xlims(plt_cov), grid=false)
    else
        plt_cov = scatter(xcov, ycov, yerr=yerr_cov, mc=:lightgray, msc=:lightgray,
                    xlabel="Empirical covariance", ylabel=ylabel_cov, label=false, legend=:topleft)
        scatter!(plt_cov, xcov[var_mask], ycov[var_mask], c=2, label="Variance (diagonal)")
        scatter!(plt_cov, xcov[cov_mask], ycov[cov_mask], c=1, label="Covariance (off-diagonal)")
        plot!(plt_cov,[-1e6, 1e6], [-1e6, 1e6], label=false, lc=:black, ls=:dash,ylims=ylims(plt_cov),xlims=xlims(plt_cov), grid=false)
    end

    return plt_mean, plt_cov
end