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
    # build_queuing_model([1,1,1]): primordial and primary each carry a survival
    # parameter (θ12, θ23), i.e. both can undergo atresia. The last compartment
    # (secondary) sends all exits to the unobserved bin, so for k=1 its survival
    # has no effect on the generator and we pin it to 1 (leaving it free would just
    # sample the prior). rate_params = [μ1, μ2, μ3, θ12, θ23].
    fq = build_queuing_model([1, 1, 1])
    faddy_transition = rp -> fq.transition_fcn(vcat(rp, one(eltype(rp))))
    faddy = (; name = "Faddy", k = [1, 1, 1], paused = falses(3),
               transition_fcn = faddy_transition,
               coarse_grain   = fq.coarse_grain, n_hidden = fq.n_hidden,
               init_priors    = init_priors,
               π_priors       = Dirichlet(ones(fq.n_hidden)),
               rate_priors    = [μ_priors..., Beta(4, 4), Beta(4, 4)])

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

    # --- Time-dependent variants: primordial exit μ1 -> μ(t) (sigmoid, τ fixed at 2
    # months). Each wraps the corresponding base model with build_timedep, which splits
    # the first rate param into (μ_early, μ_late); both reuse the μ1 residence prior. ---

    # FaddyTimeDep: rate_params = [μ_early, μ_late, μ2, μ3, θ12, θ23]
    td = build_timedep_faddy(; t0 = 2.0, τ = 2.0)
    faddy_td = (; name = "FaddyTimeDep", k = [1, 1, 1], paused = falses(3),
                  transition_fcn = td.transition_fcn,
                  coarse_grain   = td.coarse_grain, n_hidden = td.n_hidden,
                  init_priors    = init_priors,
                  π_priors       = Dirichlet(ones(td.n_hidden)),
                  rate_priors    = [μ_priors[1], μ_priors[1], μ_priors[2], μ_priors[3],
                                    Beta(4, 4), Beta(4, 4)])

    # QueuingTimeDep: rate_params = [μ_early, μ_late, μ2, μ3, θ1, θ2, θ3]
    qtd = build_timedep(q; t0 = 2.0, τ = 2.0)
    queuing_td = (; name = "QueuingTimeDep", k = k, paused = falses(3),
                    transition_fcn = qtd.transition_fcn,
                    coarse_grain   = qtd.coarse_grain, n_hidden = qtd.n_hidden,
                    init_priors    = init_priors,
                    π_priors       = Dirichlet(ones(qtd.n_hidden)),
                    rate_priors    = [μ_priors[1], μ_priors[1], μ_priors[2], μ_priors[3],
                                      Beta(4, 4), Beta(4, 4), Beta(4, 4)])

    # PausedTimeDep: rate_params = [μ_early, μ_late, μ2, μ3, θ1, θ2, θ3, μ_pause_primary, μ_pause_secondary]
    ptd = build_timedep(p; t0 = 2.0, τ = 2.0)
    paused_td = (; name = "PausedTimeDep", k = k, paused = paused_flags,
                   transition_fcn = ptd.transition_fcn,
                   coarse_grain   = ptd.coarse_grain, n_hidden = ptd.n_hidden,
                   init_priors    = init_priors,
                   π_priors       = Dirichlet(ones(ptd.n_hidden)),
                   rate_priors    = [μ_priors[1], μ_priors[1], μ_priors[2], μ_priors[3],
                                     Beta(4, 4), Beta(4, 4), Beta(4, 4),
                                     Exponential(5.0), Exponential(5.0)])

    return [faddy, faddy_td, queuing, queuing_td, paused, paused_td]
end


# Fetch a single model configuration by name ("Faddy", "FaddyTimeDep", "Queuing",
# "QueuingTimeDep", "Paused", "PausedTimeDep").
model_config(name) = only(filter(c -> c.name == name, model_registry()))
