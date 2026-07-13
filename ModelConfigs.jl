# Single source of truth for the candidate models: their topology and their priors.
#
# The analysis scripts (FaddyModel.jl, QueuingModel.jl, PausingModel.jl) and the
# cross-validation code all pull their model definition from here, so the priors
# are defined in exactly one place and cannot drift out of sync. This file has no
# side effects (no sample() calls) and is safe to include from anywhere.
#
# model_registry() returns one NamedTuple per model:
#     (; name, k, paused, transition_fcn, coarse_grain, n_hidden,
#        init_priors, π_priors, rate_priors)
# Use model_config(name) to fetch one by name.

isdefined(@__MODULE__, :total_model)        || include("Models.jl")
isdefined(@__MODULE__, :load_training_data) || include("Utils.jl")


function model_registry()
    # Ballpark timescales from the Faddy fit (converted to 1/month), used to set
    # priors. Shared across all three models.
    θ_fixed = [0.0043, 0.0017, 0.043, 0.057] * 30.4
    μ1_fixed = 1 / (θ_fixed[1] + θ_fixed[2])
    μ2_fixed = 1 / θ_fixed[3]
    μ3_fixed = 1 / θ_fixed[4]

    # Shared initial-condition prior: [μ_N (total follicle scale), p (overdispersion)].
    init_priors = [LogNormal(params_logn(1750, 35_000)...),
                   Truncated(Beta(3, 750), 1e-8, Inf)]

    # Shared residence-time priors for the three compartment means [μ1, μ2, μ3].
    μ_priors = [LogNormal(params_logn(μ1_fixed, 3.0)...),
                LogNormal(params_logn(μ2_fixed, 0.008)...),
                LogNormal(params_logn(μ3_fixed, 0.008)...)]

    # --- Faddy / pure-exponential ---
    # Exactly build_queuing_model([1,1,1]) with Primary survival pinned to 1
    # (Faddy has no death from Primary/Secondary); the last compartment's survival
    # is irrelevant, so we pass 1 for it too. rate_params = [μ1, μ2, μ3, θ12].
    fq = build_queuing_model([1, 1, 1])
    faddy_transition = rp -> fq.transition_fcn(vcat(rp, one(eltype(rp)), one(eltype(rp))))
    faddy = (; name = "Faddy", k = [1, 1, 1], paused = falses(3),
               transition_fcn = faddy_transition,
               coarse_grain   = fq.coarse_grain, n_hidden = fq.n_hidden,
               init_priors    = init_priors,
               π_priors       = Dirichlet(ones(fq.n_hidden)),
               rate_priors    = [μ_priors..., Beta(4, 4)])

    # --- Erlang / queuing (rate_params = [μ1, μ2, μ3, θ1, θ2, θ3]) ---
    k = [1, 8, 8]
    q = build_queuing_model(k)
    queuing = (; name = "Queuing", k = k, paused = falses(3),
                 transition_fcn = q.transition_fcn,
                 coarse_grain   = q.coarse_grain, n_hidden = q.n_hidden,
                 init_priors    = init_priors,
                 π_priors       = Dirichlet(ones(q.n_hidden)),
                 rate_priors    = [μ_priors..., Beta(4, 4), Beta(4, 4), Beta(4, 4)])

    # --- Paused + Erlang (rate_params = [μ1, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary]) ---
    paused_flags = [false, true, true]
    p = build_queuing_model(k; paused = paused_flags)
    paused = (; name = "Paused", k = k, paused = paused_flags,
                transition_fcn = p.transition_fcn,
                coarse_grain   = p.coarse_grain, n_hidden = p.n_hidden,
                init_priors    = init_priors,
                π_priors       = Dirichlet(ones(p.n_hidden)),
                rate_priors    = [μ_priors..., Beta(4, 4), Beta(4, 4), Beta(4, 4),
                                  Exponential(5.0), Exponential(5.0)])

    return [faddy, queuing, paused]
end


# Fetch a single model configuration by name ("Faddy", "Queuing", "Paused").
model_config(name) = only(filter(c -> c.name == name, model_registry()))
