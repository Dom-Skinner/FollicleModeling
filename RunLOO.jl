# Driver: out-of-sample cross-validation across all candidate models, two ways.
#
# For each model (Faddy, Queuing, Paused) we refit total_model on a reduced data
# set and score the held-out ovaries' count vectors under the marginal
# NegativeMultinomial likelihood. Two complementary schemes (see
# docs/LOO_instructions.md):
#   * leave-one-ovary-out  — drop each 4-12 month ovary in turn (the overall
#                            comparison, Panel A). Training still contains other
#                            ovaries at the held-out ovary's age.
#   * leave-one-age-out    — drop each whole age group in turn, then score every
#                            withheld ovary at that age (Panel B). A stricter test
#                            of the dynamics: no same-age siblings in training.
#
# The script has two parts:
#   PART 1 (expensive) runs both sets of refits and saves the pointwise ELPD
#           tables to models/loo_results.jld2 (keys results_ovary, results_age).
#   PART 2 (cheap)     loads them back and makes the summary tables and plots. It
#           depends only on the header below (usings + includes), NOT on Part 1,
#           so once the jld2 exists you can re-run just PART 2 to tweak the
#           figures without redoing any MCMC.
# While iterating on PART 1, drop N_SAMPLES or restrict the folds to keep runs fast.

using JLD2
using DataFrames
using Statistics
using Plots, StatsPlots, Measures

include("ModelConfigs.jl")
include("CrossValidation.jl")

const RESULTS_FILE = "models/loo_results.jld2"


# ============================================================
# PART 1 — refit + score (expensive; writes RESULTS_FILE)
# ============================================================
# ---- Tunables ----
const N_SAMPLES = 75     # post-warmup draws per chain per fold
const N_CHAINS  = 2
const SEED      = 1
# ------------------

data   = load_training_data()
models = model_registry()

# Leave-one-ovary-out (Panel A): ~N_ovaries refits per model.
results_ovary = reduce(vcat,
    [elpd_cv(m, data; n_samples = N_SAMPLES, n_chains = N_CHAINS, seed = SEED) for m in models])

# Leave-one-age-out (Panel B): one refit per age per model.
folds_age = age_folds(data)
results_age = reduce(vcat,
    [elpd_cv(m, data; folds = folds_age, n_samples = N_SAMPLES, n_chains = N_CHAINS, seed = SEED)
     for m in models])

jldsave(RESULTS_FILE; results_ovary, results_age)


# ============================================================
# PART 2 — summarize + plot (cheap; reads RESULTS_FILE)
# ============================================================
# Self-contained: run the header above once, then this block can be re-run on its
# own against a previously saved RESULTS_FILE.
results_ovary, results_age = jldopen(RESULTS_FILE, "r") do f
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
model_order = ["Faddy", "Queuing", "Paused"]                     # simple -> complex
dodge = range(-0.15, 0.15, length = length(model_order))         # separate models at each x

# Panel A: overall, leave-one-ovary-out.
pA = plot(title = "Panel A: overall (leave-one-ovary-out)", ylabel = "ΔELPD vs $ref",
          xlabel = "", xticks = ([1], ["all ages"]), xlims = (0.5, 1.5),
          grid = false, legend = :bottomright)
hline!(pA, [0]; lc = :gray, ls = :dash, label = "")
for (j, m) in enumerate(model_order)
    row = dA[dA.model .== m, :]
    scatter!(pA, [1 + dodge[j]], row.Δelpd; yerror = row.se, label = m, ms = 7, msw = 2)
end

# Panel B: by age, leave-one-age-out, dodged and lined by model.
ages_sorted = sort(unique(results_age.age))
pB = plot(title = "Panel B: by age (leave-one-age-out)", ylabel = "ΔELPD vs $ref",
          xlabel = "Age (months)", xticks = (1:length(ages_sorted), string.(Int.(ages_sorted))),
          grid = false, legend = false)
hline!(pB, [0]; lc = :gray, ls = :dash, label = "")
for (j, m) in enumerate(model_order)
    sub = sort(dB[dB.model .== m, :], :age)
    xs = [findfirst(==(a), ages_sorted) for a in sub.age] .+ dodge[j]
    scatter!(pB, xs, sub.Δelpd; yerror = sub.se, marker = :circle, ms = 5, msw = 2, lw = 1.5, label = m)
end

plot(pA, pB, layout = (1, 2), size = (1100, 450), margin = 5mm)
savefig("plots/loo_elpd_delta.pdf")
