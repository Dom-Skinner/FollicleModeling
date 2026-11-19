using Random
using JLD2

include("src/Models.jl")
include("src/Utils.jl")
include("src/ModelConfigs.jl")

# Stochastically simulate every follicle of one ovary from 2 to 12 months (see
# simulate_ovary in src/Utils.jl) for n_sims independent posterior draws of a model.
# The posterior is loaded from models/<model>_chain.jld2, which the model's analysis
# script saves after its long fit (e.g. PausedTimeDepModel.jl -> PausedTimeDep_chain).
# Each simulation is saved as plain arrays in simulations/<model>_sim_<k>.jld2, which
# VisualizeOvary.jl animates.
#
# Usage:  julia --project=. SimulateOvary.jl <model name from ModelConfigs.jl> [n_sims]
model_name = length(ARGS) >= 1 ? ARGS[1] : "Faddy"
n_sims     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
seed = 1

(; transition_fcn, coarse_grain) = model_config(model_name)
chain = load("models/$(model_name)_chain.jld2", "chain")

Random.seed!(seed)
mkpath("simulations")
for k in 1:n_sims
    samp = rand_draw(chain)
    sim = simulate_ovary(samp, transition_fcn, coarse_grain; t0=2.0, t1=12.0)
    jldsave("simulations/$(model_name)_sim_$(k).jld2"; model=model_name, draw=k, seed, sim...,
            rate_params=extract_array(samp, "rate_params"), ic=extract_array(samp, "ic"),
            π_vals=extract_array(samp, "π_vals"))
    println("$(model_name) draw $k: $(length(sim.initial)) follicles, $(length(sim.event_time)) events")
end
