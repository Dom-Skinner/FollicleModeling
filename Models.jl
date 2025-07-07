using Turing
using ExponentialUtilities
using NaNMath
using LinearAlgebra

# This file contains all of the model definitions
# Each model consists of initial conditions that are propagated through time by a rate matrix.
# We use the fact that if the initial conditions are multinomial, then the counts at later times are multinomial.

# We replace the multinomial with a normal approximation during inference for efficiency and to get a differentialble model

# Having fitted the model we can sample from either the approximate model or the multinomial model.
# Confusingly, we're call the samples from approximate model "exact" since this is the exact model that Turing is fitting

# π_vals is the probability distribution for the initial follicle compositions

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
    θ6 ~ in_priors["θ6"]
    θ7 ~ in_priors["θ7"]
    


    # Reuse these parameters, but fit separately to remainder of data
    N ~ filldist(Gamma(r*(1-p),1/p), size(observations, 1))

    W = transition_matrix_paused([θ12/w1, (1-θ12)/w1, θ34/w2, (1-θ34)/w2, 1/w3, θ6, θ7])

   
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
    N = rand(NegativeBinomial(samp.r,samp.p))
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
    r = samp.var"r"
    p = samp.var"p"
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

