# EXPERIMENTAL: in-silico lineage tracing ("GFP labelling") of follicles.
#
# At t0 a random subset of follicles is labelled (green). The label is purely a
# marker: labelled follicles follow exactly the same stochastic dynamics as every
# other follicle (simulate_ovary in src/Utils.jl), we just track where they end up.
#
# Who gets labelled is set by a named scheme (see label_schemes): a rule(category, hidden_state) returning the
# probability that a follicle in that state at t0 is labelled. Categories are
# 1 = Primordial, 2 = Primary, 3 = Secondary; hidden states are the model's substates
# (for the TimeDep models: 1 = primordial, 2:9 = primary Erlang stages, 10:17 = secondary,
# and for PausedTimeDep 18 / 19 = paused primary / paused secondary).
#
# Outputs (prefix plots/<model>_labeled_<scheme>_*):
#   _ovary.mp4   animation of one simulated ovary with labelled follicles in green
#   _summary.pdf three panels (median and 90% band over posterior draws): GFP+ fraction
#                of each compartment, cumulative GFP+ that left Secondary, raw GFP+ counts
#
# Usage:  julia --project=. LabelingExperiment.jl [model name] [labelling scheme] [n_sims]

using Random
using JLD2
using Plots
using Measures
using Statistics

include("src/Models.jl")
include("src/Utils.jl")
include("src/ModelConfigs.jl")

# Reuse the drawing/physics helpers of VisualizeOvary.jl. It is loaded inside its own
# module because it defines a two-argument helper called `mean`, which would otherwise
# shadow Statistics.mean here.
module Viz
    include(joinpath(@__DIR__, "VisualizeOvary.jl"))
end

model_name = length(ARGS) >= 1 ? ARGS[1] : "QueuingTimeDep"
scheme     = length(ARGS) >= 2 ? ARGS[2] : "primordial3pct"
n_sims     = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
seed = 1

# ---------------- labelling schemes ----------------
# Each scheme is (description, rule), where rule(category, hidden_state) is the
# probability of labelling a follicle in that state at t0. Add new schemes here.
config = model_config(model_name)
paused_states = sum(config.k)+1:config.n_hidden   # reservoir states come after the active substates (empty if none)
label_schemes = Dict(
    "primordial3pct" => ("3% of primordials",
                         (c, s) -> c == 1 ? 0.03 : 0.0),
    "paused"         => ("all initially paused follicles",
                         (c, s) -> s in paused_states ? 1.0 : 0.0),
    "paused_plus37pct" => ("all initially paused + 37% of all other follicles",
                         (c, s) -> s in paused_states ? 1.0 : 0.37),
)
label_description, label_rule = label_schemes[scheme]
tag = "$(model_name)_labeled_$(scheme)"           # output file prefix

label_follicles(sim, rule) = [rand() < rule(c, s) for (c, s) in zip(sim.initial, sim.initial_state)]


# ---------------- quantification ----------------
const CATEGORY_LABELS = ["Primordial", "Primary", "Secondary", "Left secondary (cum.)", "Died (cum.)"]

# Category of every follicle at each time in `times` (N × T matrix), replaying the event list.
function category_trajectories(sim, times)
    cat = copy(sim.initial)
    traj = zeros(Int, length(cat), length(times))
    k = 1
    for (f, t) in enumerate(times)
        while k <= length(sim.event_time) && sim.event_time[k] <= t
            cat[sim.event_follicle[k]] = sim.event_to[k]; k += 1
        end
        traj[:, f] = cat
    end
    return traj
end

# 5 × T matrix: number of follicles in `subset` in each category at each time.
counts_in(traj, subset) = [count(==(c), view(traj, subset, f)) for c in 1:5, f in 1:size(traj, 2)]


# ---------------- animation ----------------
# Same toy physics as Viz.animate_ovary; unlabelled follicles are drawn faded in their
# compartment colour, labelled ones in green on top. The lower panel shows the labelled
# follicles only.
const GREEN = RGB(0.1, 0.75, 0.2)

function animate_labeled_ovary(sim, labeled; dt_frame=0.025, substeps=10, seed=1, title_prefix="")
    Random.seed!(seed)
    ev_time, ev_fol, ev_to = sim.event_time, sim.event_follicle, sim.event_to
    cat  = copy(sim.initial)
    N    = length(cat)
    θ    = 2π .* rand(N)
    ρ    = [clamp(Viz.TARGET_ρ[c] + 0.06 * randn(), 0.05, 0.97) for c in cat]
    x, y = ρ .* cos.(θ), ρ .* sin.(θ)
    rad  = [Viz.RADIUS[c] for c in cat]

    exit_slots = Viz.pile_slots(count(==(4), ev_to), Viz.EXIT_X, Viz.PILE_Y)
    dead_slots = Viz.pile_slots(count(==(5), ev_to), Viz.DEAD_X, Viz.PILE_Y)
    pile_r     = minimum(Viz.RADIUS) * 0.9
    slot = fill((NaN, NaN), N)
    n_exit, n_dead = 0, 0

    for _ in 1:400; Viz.physics_step!(x, y, rad, cat); end

    frame_times = sim.t0:dt_frame:sim.t1
    counts = counts_in(category_trajectories(sim, frame_times), labeled)
    circle = Viz.discs([0.0], [0.0], [1.0]; n=200)
    k = 1
    anim = @animate for (f, t) in enumerate(frame_times)
        while k <= length(ev_time) && ev_time[k] <= t
            i, to = ev_fol[k], ev_to[k]
            cat[i] = to
            to == 4 && (n_exit += 1; slot[i] = exit_slots[n_exit])
            to == 5 && (n_dead += 1; slot[i] = dead_slots[n_dead])
            k += 1
        end
        for _ in 1:substeps; Viz.physics_step!(x, y, rad, cat); end
        for i in 1:N
            if cat[i] <= 3
                rad[i] += 0.25 * (Viz.RADIUS[cat[i]] - rad[i])
            else
                x[i] += 0.2 * (slot[i][1] - x[i]); y[i] += 0.2 * (slot[i][2] - y[i])
                rad[i] += 0.2 * (pile_r - rad[i])
            end
        end

        # --- ovary panel ---
        p = plot(circle..., lc=:black, lw=1.5, aspect_ratio=:equal, legend=false,
                 framestyle=:none, xlims=(-2.75, 2.75), ylims=(-1.25, 1.3),
                 title="$(title_prefix)t = $(round(t, digits=2)) months")
        piled = findall(i -> cat[i] >= 4 && !labeled[i], 1:N)          # unlabelled piles: grey
        isempty(piled) || plot!(p, Viz.discs(x[piled], y[piled], rad[piled]; n=8)...,
                                seriestype=:shape, c=:gray70, lw=0, alpha=0.5)
        for c in 1:3                                                    # unlabelled, faded
            idx = findall(i -> cat[i] == c && !labeled[i], 1:N)
            isempty(idx) || plot!(p, Viz.discs(x[idx], y[idx], rad[idx])...,
                                  seriestype=:shape, c=Viz.COLORS[c], lc=:gray40, lw=0.2, alpha=0.3)
        end
        idx = findall(labeled)                                          # labelled, on top
        plot!(p, Viz.discs(x[idx], y[idx], rad[idx])...,
              seriestype=:shape, c=GREEN, lc=:black, lw=0.4, alpha=0.95)
        annotate!(p, -1.95, Viz.PILE_Y[2] + 0.18, text("Died ($(counts[5, f]) GFP+)", 11))
        annotate!(p,  1.95, Viz.PILE_Y[2] + 0.18, text("Left secondary ($(counts[4, f]) GFP+)", 11))

        # --- labelled counts panel ---
        q = plot(grid=false, yaxis=:log10, ylims=(1, 1.3 * maximum(counts)),
                 xlims=(sim.t0, sim.t1), xlabel="Age (months)", ylabel="GFP+ count",
                 legend=:outerright)
        styles = [(Viz.COLORS[1], :solid), (Viz.COLORS[2], :solid), (Viz.COLORS[3], :solid),
                  (Viz.COLORS[3], :dash), (:gray40, :dash)]
        for c in 1:5
            yv = Float64.(counts[c, :]); yv[yv .== 0] .= NaN
            plot!(q, frame_times, yv, label=CATEGORY_LABELS[c], lc=styles[c][1], ls=styles[c][2], lw=2)
        end
        vline!(q, [t], lc=:black, label=false)

        plot(p, q, layout=grid(2, 1, heights=[0.68, 0.32]), size=(1000, 660), margin=3mm)
    end
    return anim
end


# ---------------- run ----------------
(; transition_fcn, coarse_grain) = config
chain = load("models/$(model_name)_chain.jld2", "chain")
Random.seed!(seed)

times = 2.0:0.1:12.0
# green_counts[c, f, k]: number of labelled follicles in category c at times[f] in simulation k
# (c = 1..3 compartments, 4 = cumulative left secondary, 5 = cumulative died).
# green_frac[c, f, k]: labelled fraction of compartment c.
green_counts = fill(NaN, 5, length(times), n_sims)
green_frac   = fill(NaN, 3, length(times), n_sims)

mkpath("simulations")
for k in 1:n_sims
    samp = rand_draw(chain)
    sim = simulate_ovary(samp, transition_fcn, coarse_grain; t0=times[1], t1=times[end])
    labeled = label_follicles(sim, label_rule)
    traj = category_trajectories(sim, times)

    all_counts = counts_in(traj, 1:length(labeled))
    green_counts[:, :, k] = counts_in(traj, findall(labeled))
    green_frac[:, :, k] = green_counts[1:3, :, k] ./ all_counts[1:3, :]

    if k == 1          # keep and animate the first simulation
        jldsave("simulations/$(tag)_sim_1.jld2"; model=model_name, seed, sim..., labeled,
                scheme, rate_params=extract_array(samp, "rate_params"))
        println("Animating draw 1: $(length(labeled)) follicles, $(count(labeled)) labelled")
        anim = animate_labeled_ovary(sim, labeled; title_prefix="$(model_name), GFP labelled: ")
        mp4(anim, "plots/$(tag)_ovary.mp4", fps=20)
    end
end

# Median and 90% band over simulations (ignoring NaNs from empty compartments).
function band(A)
    q(p) = [let v = filter(!isnan, A[c, f, :]); isempty(v) ? NaN : quantile(v, p) end
            for c in axes(A, 1), f in axes(A, 2)]
    return q(0.5), q(0.05), q(0.95)
end
styles = [(Viz.COLORS[1], :solid), (Viz.COLORS[2], :solid), (Viz.COLORS[3], :solid),
          (Viz.COLORS[3], :dash), (:gray40, :dash)]

# Plot the median line with a 90% ribbon for each row c of A in `rows`.
function plot_band!(p, A, rows; floor=-Inf)
    med, lo, hi = band(A)
    lo = max.(lo, floor)                       # keep ribbons on a log axis
    for c in rows
        plot!(p, times, med[c, :], ribbon=(med[c, :] .- lo[c, :], hi[c, :] .- med[c, :]),
              fillalpha=0.2, label=CATEGORY_LABELS[c], c=styles[c][1], ls=styles[c][2], lw=2)
    end
    return p
end

# Panel 1: labelled fraction of each compartment
p_frac = plot(grid=false, ylims=(0, Inf), xlabel="Age (months)", ylabel="GFP+ fraction of compartment",
              title="GFP+ fraction", legend=:topright)
plot_band!(p_frac, green_frac, 1:3)

# Panel 2: cumulative labelled follicles that have left Secondary
p_exit = plot(grid=false, xlabel="Age (months)", ylabel="GFP+ follicles (cumulative)",
              title="GFP+ left secondary", legend=false, ylims=(0, Inf))
plot_band!(p_exit, green_counts, [4])

# Panel 3: raw counts of labelled follicles per compartment (+ cumulative deaths)
p_counts = plot(grid=false, yaxis=:log10, ylims=(1, 1.3 * maximum(filter(!isnan, green_counts))),
                xlabel="Age (months)", ylabel="GFP+ count", title="GFP+ counts", legend=:outerright)
plot_band!(p_counts, green_counts, [1, 2, 3, 5]; floor=1)

plot(p_frac, p_exit, p_counts, layout=grid(1, 3, widths=[0.3, 0.27, 0.43]), size=(1500, 430),
     margin=6mm, plot_title="$(model_name): labelled $(label_description) at $(times[1]) months " *
                            "(median, 90% band over $n_sims draws)", plot_titlefontsize=11)
savefig("plots/$(tag)_summary.pdf")

# A few numbers at the end of the window (medians over simulations)
med_counts, med_frac = band(green_counts)[1], band(green_frac)[1]
println("\nAt $(times[end]) months (median over $n_sims posterior draws):")
println("  GFP+ counts: ", join(["$(CATEGORY_LABELS[c]) $(med_counts[c, end])" for c in 1:5], ", "))
println("  GFP+ fraction: ", join(["$(CATEGORY_LABELS[c]) $(round(100med_frac[c, end], digits=1))%" for c in 1:3], ", "))
