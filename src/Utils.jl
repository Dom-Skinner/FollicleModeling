# This file contains useful functions that are reused in multiple places
using DataFrames,CSV
using Turing
using LinearAlgebra


function params_logn(m,v)
    # returns the parameters of a lognormal distribution with mean m and variance v
    σ_ln = sqrt(log(1 + v/m^2))
    μ_ln = log(m) - 0.5*log(1 + v/m^2)
    return μ_ln, σ_ln
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


# Columns are [Primordial, Primary, Secondary] (CSV cols 4,6,7). The inference
# code only ever uses these three. Pass include_tertiary=true to also append
# Tertiary+ (CSV col 8) as a 4th column — used only for the raw-data pie chart.
function extract_data(; include_tertiary=false)
    file = "data/WT C57B6 mouse oocyte counts for Dominic.xlsx - Aging WM Tracker.csv"
    df = CSV.read(file, DataFrame)

    idx = findall(!ismissing,df.Condition)
    cols = include_tertiary ? [4,6,7,8] : [4,6,7]

    counts_2_month = Matrix(df[idx[1]:idx[2]-1,cols])
    counts_4_month = Matrix(df[idx[2]:idx[3]-1,cols])
    counts_6_month = Matrix(df[idx[3]:idx[4]-1,cols])
    counts_9_month = Matrix(df[idx[4]:idx[5]-1,cols])
    counts_12_month = Matrix(df[idx[5]:end,cols])
    return counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month
end

function load_training_data()
    counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month = extract_data()
    input_data = Float64.(vcat(counts_4_month, counts_6_month, counts_9_month, counts_12_month))
    input_times = vcat(4*ones(size(counts_4_month,1)), 6*ones(size(counts_6_month,1)),
                       9*ones(size(counts_9_month,1)), 12*ones(size(counts_12_month,1)))
    times_unique = unique(input_times)
    times_vec = [findfirst(isequal(t), times_unique) for t in input_times]
    return (; counts_2_month, counts_4_month, counts_6_month, counts_9_month, counts_12_month,
              input_data, times_unique, times_vec)
end



function confidence_intervals(f,t;N_samples=1000,q_levels = [0.025,0.1, 0.25,0.5,0.75, 0.9, 0.975])
    # find confidence intervals for any function f of the samples at some time t
    Sample_arr =  [f(t) for i in 1:N_samples]
    return quantile(Sample_arr, q_levels)
end

# Returns t -> coarse_grain * sample_model(chain, t, transition_matrix_fcn).
# Pass coarse_grain=coarse_grain_paused (3×5) for models with hidden sub-states;
# default I leaves the output unchanged for fully-observed models.
function make_sample_fun(chain, transition_matrix_fcn; coarse_grain=I)
    return t -> coarse_grain * sample_model(chain, t, transition_matrix_fcn)
end

# Computes a (n_q_levels × n_t × n_obs) quantile array suitable for credible_ribbon_plots.
# n_obs is inferred by calling sample_fun once.
function compute_quantiles(sample_fun, t_vals; N_samples=1000,
                           q_levels=[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    n_obs = length(sample_fun(first(t_vals)))
    return stack([stack([confidence_intervals(t -> sample_fun(t)[k], t;
                                              q_levels=q_levels, N_samples=N_samples)
                         for t in t_vals])
                  for k in 1:n_obs])
end


# ---- Conditional sojourn-time analysis ---------------------------------------

# Simulate one CTMC trajectory (Gillespie) through observed compartment `c`,
# starting in the first hidden substate of `c`, using generator `W`
# (column = source, row = destination) and the `coarse_grain` map that assigns
# each hidden substate to an observed compartment. Returns (t, exit_comp) where
# `t` is the time spent in compartment `c` and `exit_comp` is the observed
# compartment the follicle moved to on leaving (0 = unobserved/dead bin).
#
# `count_states` selects which hidden states' holding time is accumulated into
# `t` (default: all substates of compartment `c`). Pass a subset to time only
# part of a compartment — e.g. the *active* states of the Pausing model, so that
# time spent paused is excluded while the trajectory still passes through it.
function simulate_compartment_time(W, coarse_grain, c; count_states=nothing)
    n        = size(W, 1)
    n_hidden = size(coarse_grain, 2)
    comp = zeros(Int, n)                       # hidden state -> observed compartment
    for j in 1:n_hidden
        comp[j] = findfirst(>(0), coarse_grain[:, j])
    end                                        # comp[n] stays 0 (dead/unobserved)
    counted = isnothing(count_states) ? Set(findall(==(c), comp)) : Set(count_states)

    state = findfirst(==(c), comp)             # first substate of compartment c
    t = 0.0
    while comp[state] == c
        rate = -W[state, state]
        τ = -log(rand()) / rate                # holding time ~ Exp(rate)
        state in counted && (t += τ)           # only accumulate time in counted states
        u = rand() * rate                      # pick destination ∝ W[:,state]
        cum = 0.0
        for j in 1:n
            j == state && continue
            cum += W[j, state]
            if u <= cum
                state = j
                break
            end
        end
    end
    return t, comp[state]
end

# Posterior-predictive distribution of the time spent in observed compartment `c`,
# conditional on the follicle successfully progressing out of it (to the next
# compartment, or — for the final observed compartment — graduating to the
# unobserved growing bin). Draws posterior parameter samples and simulates one
# trajectory per draw, integrating over posterior uncertainty. `count_states` is
# forwarded to `simulate_compartment_time` (e.g. active-only states for Pausing).
function posterior_sojourn_times(chain, transition_fcn, coarse_grain, c; N=10_000, count_states=nothing)
    C = size(coarse_grain, 1)
    times = Float64[]
    while length(times) < N
        rate_params = extract_array(rand_draw(chain), "rate_params")
        W = transition_fcn(rate_params)
        t, exit_comp = simulate_compartment_time(W, coarse_grain, c; count_states=count_states)
        success = (exit_comp == c + 1) || (c == C)   # graduating from last compartment = success
        success && push!(times, t)
    end
    return times
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