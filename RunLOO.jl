# Driver: exact leave-one-ovary-out cross-validation across all candidate models.
#
# For each model (Faddy, Queuing, Paused) this refits total_model once per
# held-out 4-12 month ovary (keeping all 2-month data + all other ovaries) and
# scores the held-out ovary's count vector under the marginal NegativeMultinomial
# likelihood. Saves the pointwise ELPD table to models/loo_results.jld2 and prints
# per-model and per-age summaries. See docs/LOO_instructions.md for the design.
#
# This is expensive: ~N_ovaries refits per model. It is currently serial;
# parallelization is deferred. While iterating, drop N_SAMPLES or restrict the
# folds (see the elpd_cv `folds` keyword) to keep runs fast.

using JLD2
using DataFrames
using Plots, StatsPlots, Measures

include("ModelConfigs.jl")
include("CrossValidation.jl")

# ---- Tunables -----------------------------------------------------------------
const N_SAMPLES = 15     # post-warmup draws per chain per fold
const N_CHAINS  = 2
const SEED      = 1
# -------------------------------------------------------------------------------

data   = load_training_data()
models = model_registry()

results = reduce(vcat,
    [elpd_cv(m, data; n_samples = N_SAMPLES, n_chains = N_CHAINS, seed = SEED) for m in models])

jldsave("models/loo_results.jld2"; results)

println("\n=== Out-of-sample ELPD (leave-one-ovary-out, ages 4-12) ===")
println("Δelpd is relative to the best model (0 = best); se is the SE of the")
println("pointwise ELPD difference vs the best model.\n")
show(elpd_compare(results), allrows = true, allcols = true)

println("\n\n=== ELPD by age (for the eventual Panel B) ===\n")
show(elpd_by_age(results), allrows = true, allcols = true)
println()


# ---- Quick look at the held-out pointwise ELPD -------------------------------
# Panel A: overall distribution of held-out pointwise ELPD, one box per model.
# Panel B: the same split by age (boxes dodged by model). Higher (less negative)
# is better. NB the spread within a box is dominated by ovary-to-ovary count size
# (shared across models), so the paired ΔELPD table above is the sharper model
# comparison — these boxes are an exploratory overview.
pA = groupedboxplot(fill("All", nrow(results)), results.elpd; group = results.model,
    ylabel = "Held-out pointwise ELPD", title = "Panel A: overall", xlabel = "",
    grid = false, legend = :bottomright, fillalpha = 0.7)

ages_sorted = sort(unique(results.age))
agepos = [findfirst(==(a), ages_sorted) for a in results.age]
pB = groupedboxplot(agepos, results.elpd; group = results.model,
    xticks = (1:length(ages_sorted), string.(Int.(ages_sorted))),
    xlabel = "Age (months)", ylabel = "Held-out pointwise ELPD",
    title = "Panel B: by age", grid = false, legend = false, fillalpha = 0.7)

plot(pA, pB, layout = (1, 2), size = (1100, 450), margin = 5mm)
savefig("plots/loo_elpd_boxplots.pdf")
