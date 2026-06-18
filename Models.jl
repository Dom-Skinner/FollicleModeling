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



@model function total_model(initial_values,observations,times,times_unique,init_priors,π_priors, rate_priors,
        transition_matrix_fcn, coarse_grain_arr)


    # Combined model
    ic ~ arraydist(init_priors)
    μ  = ic[1]
    p  = ic[2]
    r  = p*μ / (1-p)
    b = p/(1-p)

    π_vals ~ π_priors
    rate_params ~ arraydist(rate_priors)

    A = sum(π_vals)
    π_k = max.(coarse_grain_arr*π_vals ./ (A + b), 1e-10)
    if sum(π_k) > 1 - 1e-10
        π_k = π_k ./ (sum(π_k) + 1e-9)
    end

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
        π_k = max.(coarse_grain_arr*a_k ./ (A + b), 1e-10)
        if sum(π_k) > 1 - 1e-10
            π_k = π_k ./ (sum(π_k) + 1e-9)
        end
        #π_k = π_k ./ (sum(π_k) + 1e-9)
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
    ic = extract_array(samp, "ic")
    μ  = ic[1]
    p  = ic[2]
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





# ================================ Model with queuing topology ================================

# General Erlang/queuing model built from an integer shape vector `k`.
#
# Observed compartment c (1..C) is expanded into `k[c]` exponential substates in
# series. Death acts uniformly: every substate of compartment c progresses at
# rate α_c and dies at rate δ_c (see docs/notes.tex). The marginal time to
# *progress out* of the compartment is Erlang(k[c], α_c).
#
# Semantic rate_params = [w_1,…,w_C, θ_1,…,θ_{C-1}]:
#   w_c : mean residence time in compartment c (total time spent there, whether
#         the follicle ends up progressing or dying)
#   θ_c : survival probability through compartment c (progress vs die),
#         for c < C only.
# The per-substate rates are chosen so that, with per-substate survival
# s_c = θ_c^(1/k[c]) and total rate r_c = α_c + δ_c,
#       (s_c)^{k[c]} = θ_c   and   (1/r_c)·(1 - s_c^{k[c]})/(1 - s_c) = w_c,
# i.e. r_c = (1 - θ_c)/((1 - s_c)·w_c), α_c = s_c·r_c, δ_c = (1 - s_c)·r_c.
# This reduces to the exponential case α_c = θ_c/w_c, δ_c = (1-θ_c)/w_c at k=1.
# The last compartment has no branching parameter: progression and death both
# flow to the same unobserved bin, so δ_C = 0 and α_C = k[C]/w_C.
#
# Returns (; transition_fcn, coarse_grain, n_hidden) for use with `total_model`:
#   transition_fcn(rate_params) -> (n_hidden+1)×(n_hidden+1) generator W
#       (column = source, row = destination; final column is the unobserved state)
#   coarse_grain :: C × n_hidden  sums substates back to observed compartments
#   n_hidden = sum(k)             length of the Dirichlet over initial states
function build_queuing_model(k::AbstractVector{<:Integer})
    C = length(k)
    n_hidden = sum(k)
    n = n_hidden + 1                 # + absorbing/unobserved "growing/dead" state
    dead = n
    offset = [sum(@view k[1:c-1]) for c in 1:C]   # substates before compartment c
    idx(c, m) = offset[c] + m

    function transition_fcn(rate_params)
        w = rate_params[1:C]
        θ = rate_params[C+1:2C-1]
        T = eltype(rate_params)
        W = zeros(T, n, n)
        for c in 1:C
            if c < C
                s = θ[c]^(1/k[c])              # per-substate survival probability
                r = (1 - θ[c]) / ((1 - s) * w[c])   # per-substate total rate α + δ
                α = s * r
                δ = (1 - s) * r
            else                              # last compartment: no death channel
                α = k[c] / w[c]
                δ = zero(α)
            end
            for m in 1:k[c]
                i = idx(c, m)
                W[i, i] = -(α + δ)
                W[dead, i] += δ
                if m < k[c]
                    W[idx(c, m+1), i] = α      # progress within the Erlang chain
                elseif c < C
                    W[idx(c+1, 1), i] = α      # progress to the next compartment
                else
                    W[dead, i] += α            # last compartment exits to unobserved bin
                end
            end
        end
        return W
    end

    coarse_grain = zeros(C, n_hidden)
    for c in 1:C, m in 1:k[c]
        coarse_grain[c, idx(c, m)] = 1.0
    end

    return (; transition_fcn, coarse_grain, n_hidden)
end