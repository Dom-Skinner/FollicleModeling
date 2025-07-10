# This file contains useful functions that are reused in multiple places
using DataFrames,CSV
using Turing 


function params_logn(m,v)
    # returns the parameters of a lognormal distribution with mean m and variance v
    σ_ln = sqrt(log(1 + v/m^2))
    μ_ln = log(m) - 0.5*log(1 + v/m^2)
    return μ_ln, σ_ln
end

function multinomial_approx_inv(X, π_vals)
    #https://aloneinthefart.blogspot.com/2012/09/normal-approximation-of-multinomial.html
    N = sum(X)
    π_safe = max.(π_vals, 1e-3)
    v = NaNMath.sqrt.(π_safe) 
    v[end] -= 1.0
    v = v/norm(v)
    Q = I(length(π_safe)) - 2*v *v'

    Z = Q[1:end-1,:] * (X ./NaNMath.sqrt.(N*π_safe) .- NaNMath.sqrt.(N*π_safe))
    
    if any(isnan.(Z)) || any(isinf.(Z))
        return 1000*ones(size(Z)) # i.e. very unlikely
    else
        return Z
    end
end


function multinomial_approx(N, π_vals,z_vec)
    #https://aloneinthefart.blogspot.com/2012/09/normal-approximation-of-multinomial.html
    π_safe = max.(π_vals, 1e-3)
    v = NaNMath.sqrt.(π_safe) 
    v[end] -= 1.0
    v = v/norm(v)
    Q = I(length(π_safe)) - 2*v *v'
    z = N*π_vals .+ NaNMath.sqrt.(N*π_safe).*(Q[:,1:end-1]*z_vec)
    if any(isnan.(z))
        return N*π_vals
    else
        return z
    end
end

function rand_draw(chain::Chains)
    df = DataFrame(chain)
    row = df[rand(1:nrow(df)), Not([:chain,:iteration])]
    return NamedTuple(row)                    # e.g. (r=…, p=…, π1=…, …)
end


function rand_draw(df::DataFrame)
    row = df[rand(1:nrow(df)), :]
    return NamedTuple(row)                    # e.g. (r=…, p=…, π1=…, …)
end



function transition_matrix_unpaused(θ)
    return [
        -(θ[1]+θ[2])    0.0      0.0       0.0  
        θ[1]     -(θ[3]+θ[4])      0.0       0.0
        0.0         θ[4]       -θ[5]      0.0
        θ[2]          θ[3]       θ[5]       0.0
    ]
end



function transition_matrix_unpaused_v2(θ)
    return [
    -(θ[1]+θ[2])     0.0         0.0             0.0       0.0    0.0
        θ[1]     -(θ[3]+θ[4])    0.0             0.0       0.0    0.0
        0.0          θ[3]     -(θ[5] + θ[6])     0.0       0.0    0.0
        0.0          0.0         θ[5]       -(θ[7]+θ[8])   0.0    0.0
        0.0          0.0         0.0             θ[7]     -θ[9]   0.0
        θ[2]         θ[4]        θ[6]            θ[8]      θ[9]   0.0
    ]
end

function extract_data()
    file = "data/WT C57B6 mouse oocyte counts for Dominic.xlsx - Aging WM Tracker.csv"
    df = CSV.read(file, DataFrame)

    idx = findall(!ismissing,df.Condition)

    counts_2_month = Matrix(df[idx[1]:idx[2]-1,[4,6,7]])
    counts_4_month = Matrix(df[idx[2]:idx[3]-1,[4,6,7]])
    counts_6_month = Matrix(df[idx[3]:idx[4]-1,[4,6,7]])
    counts_9_month = Matrix(df[idx[4]:idx[5]-1,[4,6,7]])
    counts_12_month = Matrix(df[idx[5]:end,[4,6,7]])
    return counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month
end



function confidence_intervals(f,t;N_samples=1000,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975])
    # find confidence intervals for any function f of the samples at some time t
    Sample_arr =  [f(t) for i in 1:N_samples]
    return quantile(Sample_arr, q_levels)
end


function empirical_stats(input_data,times_vec)
    times_unique = sort(unique(times_vec))
    mean_stats = [mean(input_data[times_vec .== t,:], dims=1) for t in times_unique]
    cov_stats = [cov(input_data[times_vec .== t,:]) for t in times_unique]
    return mean_stats, cov_stats
end


"""
    chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                             N::Integer=1000, probs::AbstractVector=[0.025, 0.5, 0.975])

Draws N replicated data sets from `sample_model_faddy(chain, t)` at each t ∈ times_vec,
computes the mean and covariance at each unique time, then returns

  • mean_quantiles :: Array{Float64}(length(times_unique), size(input_data,2), length(probs))
  • cov_quantiles  :: Array{Float64}(length(times_unique), size(input_data,2), size(input_data,2), length(probs))

where
  mean_quantiles[i,j,k] = the `probs[k]` quantile of the jᵗʰ mean‐stat at time times_unique[i],  
  cov_quantiles [i,a,b,k] = the `probs[k]` quantile of the (a,b) covariance‐stat at times_unique[i].
"""
function chain_stats_sample(sample_fun, input_data, times_vec, times_unique; 
                                  N::Integer=1000, probs::AbstractVector=[0.025, 0.5, 0.975])

    T = length(times_unique)          # number of distinct time‐points
    D = size(input_data, 2)           # data‐dimensionality
    P = length(probs)                 # how many quantiles

    # storage for all N replicates
    mean_samples = Array{Float64}(undef, T, D, N)
    cov_samples  = Array{Float64}(undef, T, D, D, N)

    # 1) draw N replicates, compute stats
    for n in 1:N
        synth_data = hcat([ sample_fun(times_unique[t])
                            for t in times_vec ]...) |> permutedims
        mean_synth, cov_synth = empirical_stats(synth_data, times_vec)
        for i in 1:T
            mean_samples[i, :, n]   = vec(mean_synth[i])   # flatten 1×D → D
            cov_samples[i, :, :, n] = cov_synth[i]
        end
    end

    # 2) now compute quantiles across the 3ᵈᵈ dimension
    mean_quantiles = Array{Float64}(undef, T, D, P)
    cov_quantiles  = Array{Float64}(undef, T, D, D, P)

    for i in 1:T, j in 1:D, k in 1:P
        mean_quantiles[i, j, k] = quantile(mean_samples[i, j, :], probs[k])
    end

    for i in 1:T, a in 1:D, b in 1:D, k in 1:P
        cov_quantiles[i, a, b, k] = quantile(cov_samples[i, a, b, :], probs[k])
    end

    return mean_quantiles, cov_quantiles
end