using Turing
using ExponentialUtilities
using NaNMath
using LinearAlgebra

# This file contains the main model, which is agnostsic to the particular observation/model topology strucure.

function finite_transition_matrix(W,t)
    
    if any(isnan, W) || any(isinf, W) || maximum(abs.(W)) > 1e10
        return W *t # don't throw error for bad theta choice
    end
    #@info "W matrix" W minimum(W) maximum(W) isnan.(W) isinf.(W)
    transition_matrix = exponential!(W * (t - 2.0)) # t=2.0 is the initial time! 
    #=
        try
            transition_matrix = exponential!(W * (t - 2.0)) # t=2.0 is the initial time! 
            return transition_matrix
        catch 
            @warning "Error in finite_transition_matrix: W matrix is not finite or too large"
            Turing.@addlogprob!(-Inf) # try to avoid NaN params
            return
        end
    end
    =#
    return transition_matrix
end

function probability_flow(π_vals,W,times_unique)
    Λ_all = Vector{typeof(π_vals)}(undef, length(times_unique))
    for i in 1:length(times_unique)
        transition_matrix = finite_transition_matrix(W, times_unique[i])
        Λ_all[i] = transition_matrix[:,1:end-1]* π_vals # by convention final column is unobserved state
        Λ_all[i][Λ_all[i] .<= 0.0] .= 0.0 # Set negative values to zero
        Λ_all[i][isnan.(Λ_all[i])] .= 0.0 # Set NaN values to zero

        # Λ = exp(W * t)*π 
    end
    return Λ_all
end
# ================================ Faddy model ================================



@model function total_model(initial_values,observations,times,times_unique,innit_priors,π_priors, rate_priors,
        transition_matrix_fcn, coarse_grain_arr)

    
    # Combined model
    inpriors ~ arraydist(innit_priors)
    μ  = inpriors[1]
    p  = inpriors[2]
    r  = p*μ / (1-p)
    b = p/(1-p)

    π_vals ~ π_priors
    rate_params ~ arraydist(rate_priors)

    A = sum(π_vals)
    π_k = clamp.(coarse_grain_arr*π_vals ./ (A + b),1e-10,1-1e-9)

    for i in 1:size(initial_values, 1)    
        initial_values[i,:] ~ AugmentedGPLikelihoods.SpecialDistributions.NegativeMultinomial(r, π_k)
    end
    
    if length(times_unique) == 0
        return
    end



    W = transition_matrix_fcn(rate_params)

    Λ_all = probability_flow(π_vals,W,times_unique)


    for i in 1:length(times)
        a_k = Λ_all[times[i]][1:end-1] # final state considered unobserved
        A = sum(a_k)
        π_k = clamp.(coarse_grain_arr*a_k ./ (A + b),1e-10,1-1e-9)
        observations[i,:] ~ AugmentedGPLikelihoods.SpecialDistributions.NegativeMultinomial(r, π_k)
    end

end

function extract_array(nt, prefix)
    pairs_for_prefix = filter(p -> startswith(String(p.first), prefix * "["), pairs(nt))

    idx(x) = parse(Int, match(r"\[(\d+)\]$", String(x)).captures[1])

    vals = [p.second for p in sort(collect(pairs_for_prefix), by = p -> idx(p.first))]
    return vals
end


function sample_model(chain,t, transition_matrix_fcn)
    samp = rand_draw(chain)
    inpriors = extract_array(samp, "inpriors")
    μ  = inpriors[1]
    p  = inpriors[2]
    r  = p*μ / (1-p)
    b = p/(1-p)

    π_vals = extract_array(samp, "π_vals")
    rate_params = extract_array(samp, "rate_params")
    W = transition_matrix_fcn(rate_params)

    Λ_all = probability_flow(π_vals,W,[t])

    a_k = Λ_all[1][1:end-1] # final state considered unobserved
    A = sum(a_k)
    π_k = clamp.(a_k ./ (A + b),1e-10,1-1e-9)
    return rand(AugmentedGPLikelihoods.SpecialDistributions.NegativeMultinomial(r, π_k))
    
end


# ================================ Model with paused states ================================


function transition_matrix_paused(θ)
    return [
        -(θ[1]+θ[2])    0.0        0.0     0.0   0.0   0.0
        θ[1]      -(θ[3]+θ[4])     θ[6]    0.0   0.0   0.0
        0.0             0.0       -θ[6]    0.0   0.0   0.0
        0.0             θ[4]       0.0    -θ[5]  θ[7]  0.0
        0.0             0.0        0.0     0.0  -θ[7]  0.0
        θ[2]            θ[3]       0.0     θ[5]  0.0   0.0
    ]
end



@model function pausing_model(initial_sizes,initial_values,observations,times,times_unique,in_priors)

    
    # First set up initial conditions at 2 months
    μ ~ in_priors["μ"]
    p ~ in_priors["p"]
    r  = p*μ / (1-p)

    for i in 1:size(initial_sizes, 1)
        initial_sizes[i] ~ NegativeBinomial(r,p)  
    end


    π_vals ~ in_priors["π_vals"]

    for i in 1:size(initial_values, 1)    
        initial_values[i,:] ~ Multinomial(initial_sizes[i], [π_vals[1],π_vals[2]+π_vals[3],π_vals[4] + π_vals[5]])
    end
    
    w1 ~ in_priors["w1"]
    w2 ~ in_priors["w2"]
    w3 ~ in_priors["w3"]
    θ12 ~ in_priors["θ12"]
    θ34 ~ in_priors["θ34"]
    θ6 ~ in_priors["θ6"]
    θ7 ~ in_priors["θ7"]
    


    # Reuse these parameters, but fit separately to remainder of data
    N ~ filldist(Gamma(r*(1-p),1/p), size(observations, 1))

    W = transition_matrix_paused([θ12/w1, (1-θ12)/w1, θ34/w2, (1-θ34)/w2, 1/w3, θ6, θ7])

   Λ_all = probability_flow(π_vals,W,times_unique)


    for i in 1:length(times)
        obs_probs = Λ_all[times[i]][1:length(π_vals)]
        fourth_prob = 1.0 - sum(obs_probs)
        full_probs = [obs_probs[1],obs_probs[2]+obs_probs[3],obs_probs[4]+obs_probs[5], fourth_prob]

        obs_counts = observations[i,:]
        unobs_count = N[i] - sum(obs_counts)
        z_vals  = multinomial_approx_inv(vcat(obs_counts,unobs_count),full_probs)
        for z in z_vals
            @assert isfinite(z)
            Turing.@addlogprob! logpdf(Normal(zero(z[1]), one(z[1])), z)
        end

    end
    

end


function sample_model_paused(chain,t)
    samp = rand_draw(chain)
    μ = samp.var"μ"
    p = samp.var"p"
    r = p*μ / (1-p)
    N = rand(NegativeBinomial(r,p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    θ34 = samp.var"θ34"
    θ6 = samp.var"θ6"
    θ7 = samp.var"θ7"
    W = transition_matrix_paused([θ12/w1, (1-θ12)/w1, θ34/w2, (1-θ34)/w2, 1/w3, θ6, θ7])
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]", samp.var"π_vals[4]", samp.var"π_vals[5]"]
    return rand(Multinomial(N,Λ))
end

function sample_model_exact_paused(chain,t)
    samp = rand_draw(chain)
    μ = samp.var"μ"
    p = samp.var"p"
    r = p*μ / (1-p)
    N = rand(Gamma(r*(1-p),1/p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    θ34 = samp.var"θ34"
    θ6 = samp.var"θ6"
    θ7 = samp.var"θ7"
    W = transition_matrix_paused([θ12/w1, (1-θ12)/w1, θ34/w2, (1-θ34)/w2, 1/w3, θ6, θ7])
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]", samp.var"π_vals[4]", samp.var"π_vals[5]"]
    return multinomial_approx(N, Λ,randn(5))
end



# ================================ Model with queuing topology ================================


function transition_matrix_queuing(θ)
    return [
    -(θ[1]+θ[2])        0.0      0.0      0.0       0.0   0.0
        θ[1]      -(θ[3]+θ[4])   0.0      0.0       0.0   0.0
        0.0             θ[3]   -θ[5]      0.0       0.0   0.0
        0.0             0.0     θ[5]  -(θ[6]+θ[7])  0.0   0.0
        0.0             0.0      0.0      θ[6]     -θ[8]  0.0
        θ[2]            θ[4]     0.0      θ[7]      θ[8]  0.0
    ]
end

function params_to_rates_queuing(w1,w2,w3,θ12,θ34,θ345,θ678,θ67)
    θ1 = θ12/w1
    θ2 = (1-θ12)/w1
    q = θ345*w2
    θ5 = 1/ ( 1- θ345) / w2
    θ3 = θ34/q
    θ4 = (1-θ34)/q
    q2 = θ678*w3
    θ8 = 1/ (1-θ678)/w3
    θ6 = θ67/q2
    θ7 = (1-θ67)/q2
    return [θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8]
end

@model function queuing_model(initial_sizes,initial_values,observations,times,times_unique,in_priors)

    
   # First set up initial conditions at 2 months
   r ~ in_priors["r"]#LogNormal(2, 0.5) 
   p ~ in_priors["p"]#Beta(2, 500)

    for i in 1:size(initial_sizes, 1)
        initial_sizes[i] ~ NegativeBinomial(r,p)  
    end


    π_vals ~ in_priors["π_vals"]

    for i in 1:size(initial_values, 1)    
        initial_values[i,:] ~ Multinomial(initial_sizes[i], [π_vals[1],π_vals[2]+π_vals[3],π_vals[4] + π_vals[5]])
    end
    
    w1 ~ in_priors["w1"]
    w2 ~ in_priors["w2"]
    w3 ~ in_priors["w3"]
    θ12 ~ in_priors["θ12"]
    θ34 ~ in_priors["θ34"]
    θ345 ~ in_priors["θ345"]
    θ678 ~ in_priors["θ678"]
    θ67 ~ in_priors["θ67"]




    # Reuse these parameters, but fit separately to remainder of data
    N ~ filldist(Gamma(r*(1-p),1/p), size(observations, 1))

    W = transition_matrix_queuing(params_to_rates_queuing(w1,w2,w3,θ12,θ34,θ345,θ678,θ67))

    Λ_all = probability_flow(π_vals,W,times_unique)

    for i in 1:length(times)
        obs_probs = Λ_all[times[i]][1:length(π_vals)]
        fourth_prob = 1.0 - sum(obs_probs)
        full_probs = [obs_probs[1],obs_probs[2]+obs_probs[3],obs_probs[4]+obs_probs[5], fourth_prob]

        obs_counts = observations[i,:]
        unobs_count = N[i] - sum(obs_counts)
        z_vals  = multinomial_approx_inv(vcat(obs_counts,unobs_count),full_probs)
        for z in z_vals
            @assert isfinite(z)
            Turing.@addlogprob! logpdf(Normal(zero(z[1]), one(z[1])), z)
        end

    end
    

end


function sample_model_queuing(chain,t)
    samp = rand_draw(chain)
    N = rand(NegativeBinomial(samp.r,samp.p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    θ34 = samp.var"θ34"
    θ345 = samp.var"θ345"
    θ678 = samp.var"θ678"
    θ67 = samp.var"θ67"
    W = transition_matrix_queuing(params_to_rates_queuing(w1,w2,w3,θ12,θ34,θ345,θ678,θ67))
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]", samp.var"π_vals[4]", samp.var"π_vals[5]"]
    Λ = Λ ./ sum(Λ) # Ensure probabilities sum to 1
    return rand(Multinomial(N,Λ))
end

function sample_model_exact_queuing(chain,t)
    samp = rand_draw(chain)
    r = samp.var"r"
    p = samp.var"p"
    N = rand(Gamma(r*(1-p),1/p))
    w1 = samp.var"w1"
    w2 = samp.var"w2"
    w3 = samp.var"w3"
    θ12 = samp.var"θ12"
    θ34 = samp.var"θ34"
    θ345 = samp.var"θ345"
    θ678 = samp.var"θ678"
    θ67 = samp.var"θ67"
    W = transition_matrix_queuing(params_to_rates_queuing(w1,w2,w3,θ12,θ34,θ345,θ678,θ67))
    transition_matrix = exponential!(W * (t -2.0)) 
    Λ = transition_matrix[:,1:end-1]* [samp.var"π_vals[1]", samp.var"π_vals[2]", samp.var"π_vals[3]", samp.var"π_vals[4]", samp.var"π_vals[5]"]
    @assert abs(sum(Λ) - 1.0) < 1e-6
    Λ = Λ ./ sum(Λ) 
    return multinomial_approx(N, Λ,randn(5))
end

