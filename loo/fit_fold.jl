# Per-fit worker for the Snakemake LOO pipeline: one model, one fold -> one CSV.
#
# This is a Snakemake `script:` target — it is run by the Snakefile, not directly.
# Snakemake injects a `snakemake` object carrying this job's inputs/outputs/params/
# wildcards, so there is no command-line parsing here. The fit+score itself is
# elpd_cv_fold in CrossValidation.jl — the SAME function RunLOO.jl uses serially,
# so the parallel and serial paths cannot drift.
#
# The 2 MCMC chains run via MCMCThreads(); the Snakefile sets JULIA_NUM_THREADS to
# n_chains so those chains actually get separate threads. JULIA_PROJECT (also set by
# the Snakefile) pins this project's Manifest. `root` is passed as a param because
# under `script:` @__DIR__ points at Snakemake's temp wrapper, not the repo.

const ROOT = snakemake.params["root"]
include(joinpath(ROOT, "ModelConfigs.jl"))
include(joinpath(ROOT, "CrossValidation.jl"))
using CSV, DataFrames

scheme    = snakemake.params["scheme"]                 # "ovary" or "age"
model     = snakemake.wildcards["model"]
foldstr   = string(snakemake.params["fold"])           # ovary row index, or age value
n_samples = Int(snakemake.params["n_samples"])
n_chains  = Int(snakemake.params["n_chains"])
seed      = Int(snakemake.params["seed"])
outfile   = snakemake.output["csv"]

data   = load_training_data()
config = model_config(model)
ages   = data.times_unique[data.times_vec]             # Float64 age per ovary row

# Held-out global row indices for this fold.
fold = scheme == "ovary" ? [parse(Int, foldstr)] :
       scheme == "age"   ? findall(==(parse(Float64, foldstr)), ages) :
       error("unknown scheme '$scheme' (expected 'ovary' or 'age')")
isempty(fold) && error("fold $foldstr (scheme=$scheme) selected no ovaries")

rows = elpd_cv_fold(config, data, fold; n_samples = n_samples, n_chains = n_chains, seed = seed)

mkpath(dirname(outfile))
CSV.write(outfile, rows)
println("wrote $(nrow(rows)) row(s) -> $outfile")
