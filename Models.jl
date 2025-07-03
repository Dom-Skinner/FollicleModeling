using Turing
using ExponentialUtilities
using NaNMath
using LinearAlgebra

# ================================ Faddy model ================================

function sample_model_faddy(chain,t)
    samp = rand_draw(chain)
    N = rand(NegativeBinomial(samp.r,samp.p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    W = transition_matrix_faddy([θ12/w1, (1-θ12)/w1, 1/w2, 1/w3])
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]"]
    return rand(Multinomial(N,Λ))
end

function sample_model_exact_faddy(chain,t)
    samp = rand_draw(chain)
    r = samp.var"r"
    p = samp.var"p"
    N = rand(Gamma(r*(1-p),1/p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    W = transition_matrix_faddy([θ12/w1, (1-θ12)/w1, 1/w2, 1/w3])
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]"]
    return multinomial_approx(N, Λ,[randn(),randn(),randn()])
end


function transition_matrix_faddy(θ)
    return [
        -(θ[1]+θ[2])  0.0      0.0       0.0  
        θ[1]      -(θ[3])      0.0       0.0
        0.0         θ[3]      -θ[4]      0.0
        θ[2]          0        θ[4]      0.0
    ]
end


@model function faddy_model(initial_sizes,initial_values,observations,times,times_unique,in_priors)

    
    # First set up initial conditions at 2 months
    r ~ in_priors["r"]#LogNormal(2, 0.5) 
    p ~ in_priors["p"]#Beta(2, 500)

    for i in 1:size(initial_sizes, 1)
        initial_sizes[i] ~ NegativeBinomial(r,p)  
    end


    π_vals ~ in_priors["π_vals"]

    for i in 1:size(initial_values, 1)    
        initial_values[i,:] ~ Multinomial(initial_sizes[i], π_vals)
    end
    
    #=
    dists = [
        Gamma(2.0, 0.2),
        Gamma(2.0, 0.2),
        Gamma(7.5, 0.2),
        Gamma(7.5, 0.2),
        Gamma(16.0, 0.2),
    ]
    θ ~ arraydist(dists)
    =#
    
    #θ ~ in_priors["θ"] 
    w1 ~ in_priors["w1"]
    w2 ~ in_priors["w2"]
    w3 ~ in_priors["w3"] 
    θ12 ~ in_priors["θ12"]

    if length(times_unique) == 0
        return
    end
    # Reuse these parameters, but fit separately to remainder of data
    N ~ filldist(Gamma(r*(1-p),1/p), size(observations, 1))

    W = transition_matrix_faddy([θ12/w1, (1-θ12)/w1, 1/w2, 1/w3])

   
    Λ_all = Vector{typeof(π_vals)}(undef, length(times_unique))
    for i in 1:length(times_unique)
        
        if any(isnan, W) || any(isinf, W) || maximum(abs.(W)) > 1e10
            transition_matrix = W * (times_unique[i] -2.0) # don't throw error for bad theta choice
        else
            #@info "W matrix" W minimum(W) maximum(W) isnan.(W) isinf.(W)
            try
                transition_matrix = exponential!(W * (times_unique[i] -2.0)) 
            catch 
                Turing.@addlogprob!(-Inf) # try to avoid NaN params
                return
            end
        end
        
        Λ_all[i] = transition_matrix[:,1:end-1]* π_vals
        Λ_all[i][Λ_all[i] .<= 0.0] .= 0.0 # Set negative values to zero
        # Λ = exp(W * t)*π 
    end
    #z1 = Vector{Float64}(undef, length(times))


    #obs_aug1 = Vector{Vector{Int}}(undef, length(times))
    for i in 1:length(times)
        obs_probs = Λ_all[times[i]][1:3]
        fourth_prob = 1.0 - sum(obs_probs)
        full_probs = vcat(obs_probs, fourth_prob)

        obs_counts = observations[i,:]
        unobs_count = N[i] - sum(obs_counts)
        z_vals  = multinomial_approx_inv(vcat(obs_counts,unobs_count),full_probs)
        for z in z_vals
            @assert isfinite(z)
            Turing.@addlogprob! logpdf(Normal(zero(z[1]), one(z[1])), z)
        end

    end

end