# Negative Multinomial distribution, vendored from AugmentedGPLikelihoods'
# SpecialDistributions submodule (theogf/AugmentedGPLikelihoods, MIT licence) so the
# models don't depend on the whole AGP / MeasureTheory tree just for this one
# distribution. That dependency was pinning us to old, Julia-1.11-incompatible
# versions of the SciML/OrdinaryDiffEq stack; vendoring frees the environment.
#
#   p(x | x₀, p) = Γ(x₀ + Σᵢxᵢ)/Γ(x₀) · p₀^{x₀} · Πᵢ pᵢ^{xᵢ}/xᵢ!,   p₀ = 1 - Σᵢ pᵢ.
#
# Only the pieces total_model uses are kept — construction, logpdf and rand — which
# is all Turing needs to treat it as an observation likelihood. Verified to match
# the AGP implementation bit-for-bit (see the parity check in the commit that added
# this file). loggamma is ForwardDiff-differentiable in x₀, exactly as before.
using Distributions
using Random: AbstractRNG
using SpecialFunctions: loggamma

struct NegativeMultinomial{Tx₀ <: Real, Tp <: AbstractVector} <:
       Distributions.DiscreteMultivariateDistribution
    x₀::Tx₀
    p::Tp
    function NegativeMultinomial(x₀::Real, p::AbstractVector)
        x₀ > 0 || throw(ArgumentError("x₀ has to be positive"))
        (all(>=(0), p) && sum(p) < 1) || throw(ArgumentError(
            "All p should be positive and their sum strictly smaller than 1, got $(p)."))
        return new{typeof(x₀), typeof(p)}(x₀, p)
    end
end

_p₀(d::NegativeMultinomial) = 1 - sum(d.p)

Distributions.params(d::NegativeMultinomial) = (d.x₀, d.p)
Base.eltype(::NegativeMultinomial) = Int
Base.length(d::NegativeMultinomial) = length(d.p)

function Distributions._rand!(rng::AbstractRNG, d::NegativeMultinomial, x::AbstractVector{<:Real})
    p₀ = _p₀(d)
    θ = rand(rng, Gamma(d.x₀, inv(p₀) - 1))
    λ = d.p * θ / (1 - p₀)                 # convert to the scaled-Poisson parameters
    for i in eachindex(x)
        x[i] = rand(rng, Poisson(λ[i]))
    end
    return x
end

function Distributions._logpdf(d::NegativeMultinomial, x::AbstractVector)
    return loggamma(d.x₀ + sum(x)) + d.x₀ * log(_p₀(d)) - loggamma(d.x₀) +
           mapreduce(+, d.p, x) do pᵢ, xᵢ
               xᵢ * log(pᵢ) - loggamma(xᵢ + 1)   # log(xᵢ!) = loggamma(xᵢ+1)
           end
end
