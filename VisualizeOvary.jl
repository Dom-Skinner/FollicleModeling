# Standalone animation of one simulated ovary. Does no modelling: it reads the event
# list saved by SimulateOvary.jl and draws it. The "physics" is a toy, purely for
# display: follicles are discs in a unit-circle ovary with overdamped dynamics under
# soft pairwise repulsion, a wall at the ovary edge, and a radial spring towards a
# compartment-dependent preferred radius (primordial near the edge, growing follicles
# move inwards). A follicle that dies flies out to a pile on the left (coloured by the
# compartment it died from); one that leaves Secondary flies out to a pile on the right.
#
# Usage:  julia --project=. VisualizeOvary.jl [simulations/Faddy_sim_1.jld2] [plots/Faddy_ovary_1.mp4]
# (a .gif output name also works)
#
# Simulation file format (plain arrays; categories 1..3 = Primordial/Primary/Secondary,
# 4 = exited Secondary, 5 = died):
#   t0, t1                         simulated time window (months)
#   initial[i]                     category of follicle i at t0
#   event_time, event_follicle,    time-sorted category changes: follicle event_follicle[k]
#   event_to                       moves to category event_to[k] at time event_time[k]

using JLD2
using Plots
using Measures
using Random

# ---------------- display settings (cosmetic only) ----------------
const RADIUS   = [0.011, 0.024, 0.042]          # disc radius per compartment
const TARGET_ρ = [0.87, 0.55, 0.25]             # preferred distance from centre
const K_RADIAL = [0.5, 4.0, 4.0]                # spring strength towards TARGET_ρ
const K_REPEL  = 150.0                          # soft repulsion between overlapping discs
const K_WALL   = 150.0                          # confinement to the ovary
const DT       = 0.003                          # physics step (arbitrary units)
const COLORS   = [:steelblue, :darkorange, :crimson]
const EXIT_X   = (1.2, 2.7)                     # x-range of the "left Secondary" pile
const DEAD_X   = (-2.7, -1.2)                   # x-range of the "died" pile
const PILE_Y   = (-1.0, 1.0)

# Grid of n slot positions filling a box, bottom row first.
function pile_slots(n, xr, yr)
    n == 0 && return Tuple{Float64,Float64}[]
    w, h = xr[2] - xr[1], yr[2] - yr[1]
    s = sqrt(w * h / n)                          # spacing so that n slots fit
    ncol = max(1, floor(Int, w / s))
    s = w / ncol
    return [(xr[1] + s * (mod(k, ncol) + 0.5), yr[1] + s * (div(k, ncol) + 0.5)) for k in 0:n-1]
end

# One overdamped step for the follicles still in the ovary (category ≤ 3).
function physics_step!(x, y, rad, cat)
    alive = findall(<=(3), cat)
    fx, fy = zeros(length(x)), zeros(length(y))
    for i in alive
        ρ = hypot(x[i], y[i]) + 1e-9
        f = -K_RADIAL[cat[i]] * (ρ - TARGET_ρ[cat[i]])   # radial spring
        f -= K_WALL * max(ρ + rad[i] - 1, 0)            # ovary wall
        fx[i] += f * x[i] / ρ
        fy[i] += f * y[i] / ρ
    end
    for a in eachindex(alive), b in a+1:length(alive)
        i, j = alive[a], alive[b]
        dx, dy, s = x[j] - x[i], y[j] - y[i], rad[i] + rad[j]
        (abs(dx) > s || abs(dy) > s) && continue
        d = max(hypot(dx, dy), 1e-6)
        d >= s && continue
        f = K_REPEL * (s - d) / d
        fx[i] -= f * dx; fy[i] -= f * dy
        fx[j] += f * dx; fy[j] += f * dy
    end
    x[alive] .+= DT .* fx[alive]
    y[alive] .+= DT .* fy[alive]
end

# Polygon outlines of many discs as one NaN-separated series (drawn in data units).
function discs(x, y, r; n=14)
    xs, ys = Float64[], Float64[]
    for k in eachindex(x)
        for θ in range(0, 2π, length=n)
            push!(xs, x[k] + r[k] * cos(θ)); push!(ys, y[k] + r[k] * sin(θ))
        end
        push!(xs, NaN); push!(ys, NaN)
    end
    return xs, ys
end

# Category counts at each frame time: rows Primordial, Primary, Secondary, cum. exited, cum. died.
function count_series(sim, frame_times)
    cat = copy(sim["initial"])
    counts = zeros(Int, 5, length(frame_times))
    k = 1
    for (f, t) in enumerate(frame_times)
        while k <= length(sim["event_time"]) && sim["event_time"][k] <= t
            cat[sim["event_follicle"][k]] = sim["event_to"][k]; k += 1
        end
        for c in 1:5; counts[c, f] = count(==(c), cat); end
    end
    return counts
end

function animate_ovary(sim; dt_frame=0.025, substeps=10, seed=1, title_prefix="")
    Random.seed!(seed)
    ev_time, ev_fol, ev_to = sim["event_time"], sim["event_follicle"], sim["event_to"]
    cat  = copy(sim["initial"])
    N    = length(cat)
    from = copy(cat)                                    # compartment it was in before leaving
    θ    = 2π .* rand(N)
    ρ    = [clamp(TARGET_ρ[c] + 0.06 * randn(), 0.05, 0.97) for c in cat]
    x, y = ρ .* cos.(θ), ρ .* sin.(θ)
    rad  = [RADIUS[c] for c in cat]

    exit_slots = pile_slots(count(==(4), ev_to), EXIT_X, PILE_Y)
    dead_slots = pile_slots(count(==(5), ev_to), DEAD_X, PILE_Y)
    pile_r     = minimum(RADIUS) * 0.9                  # piled follicles drawn as small dots
    slot = fill((NaN, NaN), N)
    n_exit, n_dead = 0, 0

    for _ in 1:400; physics_step!(x, y, rad, cat); end  # relax the initial packing

    frame_times = sim["t0"]:dt_frame:sim["t1"]
    counts = count_series(sim, frame_times)
    circle = discs([0.0], [0.0], [1.0]; n=200)
    k = 1
    anim = @animate for (f, t) in enumerate(frame_times)
        while k <= length(ev_time) && ev_time[k] <= t        # apply this frame's events
            i, to = ev_fol[k], ev_to[k]
            if to <= 3
                cat[i] = to
            else
                from[i] = cat[i]; cat[i] = to
                to == 4 ? (n_exit += 1; slot[i] = exit_slots[n_exit]) :
                          (n_dead += 1; slot[i] = dead_slots[n_dead])
            end
            k += 1
        end
        for _ in 1:substeps; physics_step!(x, y, rad, cat); end
        for i in 1:N
            if cat[i] <= 3
                rad[i] += 0.25 * (RADIUS[cat[i]] - rad[i])        # grow smoothly
            else                                                 # fly to the pile
                x[i] += 0.2 * (slot[i][1] - x[i]); y[i] += 0.2 * (slot[i][2] - y[i])
                rad[i] += 0.2 * (pile_r - rad[i])
            end
        end

        # --- ovary panel ---
        p = plot(circle..., lc=:black, lw=1.5, aspect_ratio=:equal, legend=false,
                 framestyle=:none, xlims=(-2.75, 2.75), ylims=(-1.25, 1.3),
                 title="$(title_prefix)t = $(round(t, digits=2)) months")
        for c in 1:3                                              # piled follicles, by origin
            for (pile, a) in ((5, 0.55), (4, 0.9))
                idx = findall(i -> cat[i] == pile && from[i] == c, 1:N)
                isempty(idx) || plot!(p, discs(x[idx], y[idx], rad[idx]; n=8)...,
                                      seriestype=:shape, c=COLORS[c], lw=0, alpha=a)
            end
        end
        for c in 1:3                                              # follicles in the ovary
            idx = findall(==(c), cat)
            isempty(idx) || plot!(p, discs(x[idx], y[idx], rad[idx])...,
                                  seriestype=:shape, c=COLORS[c], lc=:black, lw=0.3, alpha=0.9)
        end
        annotate!(p, mean(DEAD_X), PILE_Y[2] + 0.18, text("Died ($(counts[5, f]))", 11))
        annotate!(p, mean(EXIT_X), PILE_Y[2] + 0.18, text("Left secondary ($(counts[4, f]))", 11))

        # --- counts panel ---
        q = plot(grid=false, yaxis=:log10, ylims=(1, 1.3 * maximum(counts)),
                 xlims=(sim["t0"], sim["t1"]), xlabel="Age (months)", ylabel="Count",
                 legend=:outerright)
        labels = ["Primordial", "Primary", "Secondary", "Left secondary (cum.)", "Died (cum.)"]
        styles = [(COLORS[1], :solid), (COLORS[2], :solid), (COLORS[3], :solid),
                  (COLORS[3], :dash), (:gray40, :dash)]
        for c in 1:5
            yv = Float64.(counts[c, :]); yv[yv .== 0] .= NaN       # log axis: hide zeros
            plot!(q, frame_times, yv, label=labels[c], lc=styles[c][1], ls=styles[c][2], lw=2)
        end
        vline!(q, [t], lc=:black, label=false)

        plot(p, q, layout=grid(2, 1, heights=[0.68, 0.32]), size=(1000, 660), margin=3mm)
    end
    return anim
end

mean(r) = (r[1] + r[2]) / 2

# ---------------- run ----------------
if abspath(PROGRAM_FILE) == @__FILE__
    sim_file = length(ARGS) >= 1 ? ARGS[1] : "simulations/Faddy_sim_1.jld2"
    out_file = length(ARGS) >= 2 ? ARGS[2] :
               "plots/" * replace(basename(sim_file), "_sim" => "_ovary", ".jld2" => ".mp4")
    sim = load(sim_file)
    label = get(sim, "model", "") * (haskey(sim, "draw") ? " (draw $(sim["draw"]))" : "")
    anim = animate_ovary(sim; title_prefix = isempty(label) ? "" : "$label: ")
    endswith(out_file, ".gif") ? gif(anim, out_file, fps=20) : mp4(anim, out_file, fps=20)
end
