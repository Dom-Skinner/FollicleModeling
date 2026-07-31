# Plot step for the Snakemake LOO pipeline. This is RunLOO.jl's PART 2, reading the
# combined results (named input `results`) and writing the ΔELPD figure (named
# output `figure`). Cheap and re-runnable: it touches only the summary tables, none
# of the expensive fits.
#
# Snakemake `script:` target.

const ROOT = snakemake.params["root"]
using JLD2
using DataFrames
using Statistics
using Plots, StatsPlots, Measures

include(joinpath(ROOT, "ModelConfigs.jl"))
include(joinpath(ROOT, "CrossValidation.jl"))          # elpd_delta

resultsfile = snakemake.input["results"]
figfile     = snakemake.output["figure"]

results_ovary, results_age = jldopen(resultsfile, "r") do f
    f["results_ovary"], f["results_age"]
end

# Panel A from leave-one-ovary-out; Panel B from leave-one-age-out. Both use the
# same reference model (the overall best from Panel A) so the zero line matches.
ref, dA = elpd_delta(results_ovary; by = [:model])
_,   dB = elpd_delta(results_age;   by = [:model, :age], ref = ref)

println("\n=== Overall ΔELPD (leave-one-ovary-out, ages 4-12) ===")
println("Reference = $ref. Per held-out ovary d_i = elpd_i(model) - elpd_i(ref);")
println("Δelpd = Σ d_i, se = sqrt(n·Var d_i). Reference sits at 0, worse < 0.\n")
show(sort(dA, :Δelpd, rev = true), allrows = true, allcols = true)

println("\n\n=== ΔELPD by age (leave-one-age-out; reference = $ref) ===\n")
show(sort(dB, [:age, :model]), allrows = true, allcols = true)
println()

# We plot the paired ΔELPD rather than absolute per-ovary ELPD: the latter is
# dominated by ovary-to-ovary count size (shared across models), which swamps the
# model differences. The reference model sits on the dashed zero line; below = worse.
# Simple -> complex order, restricted to the models actually present in the results
# (a partial run — e.g. only the non-time-dependent models — must still plot; an
# absent model would otherwise feed an empty series to GR and crash savefig).
model_order = ["Faddy", "FaddyTimeDep", "Queuing", "QueuingTimeDep", "Paused", "PausedTimeDep"]
present     = filter(m -> m in results_ovary.model, model_order)
dodge       = range(-0.15, 0.15, length = max(length(present), 1))   # separate models at each x

# Panel A: overall, leave-one-ovary-out.
pA = plot(title = "Panel A: overall (leave-one-ovary-out)", ylabel = "ΔELPD vs $ref",
          xlabel = "", xticks = ([1], ["all ages"]), xlims = (0.5, 1.5),
          grid = false, legend = :bottomright)
hline!(pA, [0]; lc = :gray, ls = :dash, label = "")
for (j, m) in enumerate(present)
    row = dA[dA.model .== m, :]
    isempty(row) && continue
    scatter!(pA, [1 + dodge[j]], row.Δelpd; yerror = row.se, label = m, ms = 7, msw = 2)
end

# Panel B: by age, leave-one-age-out, dodged and lined by model.
ages_sorted = sort(unique(results_age.age))
pB = plot(title = "Panel B: by age (leave-one-age-out)", ylabel = "ΔELPD vs $ref",
          xlabel = "Age (months)", xticks = (1:length(ages_sorted), string.(Int.(ages_sorted))),
          grid = false, legend = false)
hline!(pB, [0]; lc = :gray, ls = :dash, label = "")
for (j, m) in enumerate(present)
    sub = sort(dB[dB.model .== m, :], :age)
    isempty(sub) && continue
    xs = [findfirst(==(a), ages_sorted) for a in sub.age] .+ dodge[j]
    scatter!(pB, xs, sub.Δelpd; yerror = sub.se, marker = :circle, ms = 5, msw = 2, lw = 1.5, label = m)
end

plot(pA, pB, layout = (1, 2), size = (1100, 450), margin = 5mm)
mkpath(dirname(figfile))
savefig(figfile)
println("wrote $figfile")
