# Gather step for the Snakemake LOO pipeline: concatenate every per-fold CSV into
# the two DataFrames RunLOO.jl PART 2 expects, and save them to one JLD2.
#
# Snakemake `script:` target. The exact per-fold file lists arrive as the named
# inputs `ovary` and `age` (see the Snakefile's expand()), so there is no directory
# walking — and Snakemake guarantees every fit finished before this runs.
#
# Writes results_ovary and results_age, each a long DataFrame(model, ovary, age,
# elpd) sorted by (model, ovary), to the named output `jld2` — deliberately a
# different path from RunLOO's models/loo_results.jld2, so the two never clobber.

const ROOT = snakemake.params["root"]
include(joinpath(ROOT, "Utils.jl"))                    # load_training_data (for the guards)
using CSV, DataFrames, JLD2

read_rows(files) =
    sort!(reduce(vcat, [DataFrame(CSV.read(f, DataFrame)) for f in files]), [:model, :ovary])

results_ovary = read_rows(collect(snakemake.input["ovary"]))
results_age   = read_rows(collect(snakemake.input["age"]))

# ---- Guards: catch a data change or a missing/duplicated fit before plotting ----
data        = load_training_data()
ages        = data.times_unique[data.times_vec]
N           = length(ages)
group_sizes = Dict(a => count(==(a), ages) for a in unique(ages))

function check_scheme(df, name)
    models = sort(unique(df.model))
    for m in models
        sub = df[df.model .== m, :]
        @assert sort(sub.ovary) == collect(1:N) "$name: model $m does not cover ovaries 1:$N exactly (got $(sort(sub.ovary)))"
        for a in unique(ages)
            @assert count(==(a), sub.age) == group_sizes[a] "$name: model $m has $(count(==(a), sub.age)) ovaries at age $a, expected $(group_sizes[a])"
        end
    end
    @assert nrow(df) == N * length(models) "$name: expected $(N*length(models)) rows, got $(nrow(df))"
    return models
end

mo = check_scheme(results_ovary, "results_ovary")
ma = check_scheme(results_age,   "results_age")
@assert mo == ma "ovary and age tables cover different model sets: $mo vs $ma"

outfile = snakemake.output["jld2"]
mkpath(dirname(outfile))
jldsave(outfile; results_ovary, results_age)
println("combined $(length(mo)) models x $N ovaries -> $outfile")
println("  results_ovary: $(nrow(results_ovary)) rows;  results_age: $(nrow(results_age)) rows")
