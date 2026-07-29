# Snakemake workflow for the leave-one-out cross-validation of the follicle models.
#
# Parallel replacement for RunLOO.jl PART 1: the 6 models x (27 leave-one-ovary-out
# + 4 leave-one-age-out) = 186 refits are independent, so each becomes its own job
# (2 cores, since the 2 MCMC chains run via MCMCThreads). A gather step concatenates
# the per-fold CSVs into one JLD2, and a plot step draws the ΔELPD figure.
#
#   snakemake -n                                  # dry run
#   snakemake --cores N all                       # run locally (each fit takes 2 cores)
#   snakemake --cores 2 --config n_samples=5 ...  # quick smoke test
#
# The Julia steps are `script:` targets: Snakemake injects a `snakemake` object
# (inputs/outputs/params/wildcards), so the scripts read their I/O and settings from
# it rather than the command line. The two env vars below stand in for the julia
# flags we would otherwise pass: JULIA_PROJECT pins this project's Manifest (the
# pinned AugmentedGPLikelihoods, source of NegativeMultinomial), and
# JULIA_NUM_THREADS gives MCMCThreads its chains' worth of threads. `root` is passed
# as a param because under `script:` the file runs from a Snakemake temp wrapper.
#
# RunLOO.jl is left untouched; this pipeline uses its own results/loo/ tree and
# writes results/loo/loo_results.jld2 (NOT RunLOO's models/loo_results.jld2). Only
# the final figure, plots/loo_elpd_delta.pdf, is shared (it is the same figure).

import os

ROOT = workflow.basedir

MODELS  = ["Faddy", "FaddyTimeDep", "Queuing", "QueuingTimeDep", "Paused", "PausedTimeDep"]
N_OVARY = 27                                     # from load_training_data(); combine.jl guards this
AGES    = [4, 6, 9, 12]

# ---- Tunables (override on the CLI, e.g. --config n_samples=5) ----
N_SAMPLES = int(config.get("n_samples", 300))    # post-warmup draws per chain per fold
N_CHAINS  = int(config.get("n_chains", 2))
SEED      = int(config.get("seed", 1))
# -------------------------------------------------------------------

# Stand-ins for `julia --project=ROOT --threads=N_CHAINS`, inherited by every job.
os.environ["JULIA_PROJECT"]     = ROOT
os.environ["JULIA_NUM_THREADS"] = str(N_CHAINS)

wildcard_constraints:
    model = "|".join(MODELS),
    i     = r"\d+",
    a     = r"\d+"

rule all:
    input:
        "plots/loo_elpd_delta.pdf"

rule fit_ovary:
    output:
        csv = "results/loo/ovary/{model}/{i}.csv"
    log:
        "results/loo/logs/ovary/{model}/{i}.log"
    params:
        root      = ROOT,
        scheme    = "ovary",
        fold      = lambda wc: wc.i,
        n_samples = N_SAMPLES,
        n_chains  = N_CHAINS,
        seed      = SEED
    threads: N_CHAINS
    script:
        "loo/fit_fold.jl"

rule fit_age:
    output:
        csv = "results/loo/age/{model}/{a}.csv"
    log:
        "results/loo/logs/age/{model}/{a}.log"
    params:
        root      = ROOT,
        scheme    = "age",
        fold      = lambda wc: wc.a,
        n_samples = N_SAMPLES,
        n_chains  = N_CHAINS,
        seed      = SEED
    threads: N_CHAINS
    script:
        "loo/fit_fold.jl"

rule combine:
    input:
        ovary = expand("results/loo/ovary/{model}/{i}.csv", model=MODELS, i=range(1, N_OVARY + 1)),
        age   = expand("results/loo/age/{model}/{a}.csv",   model=MODELS, a=AGES)
    output:
        jld2 = "results/loo/loo_results.jld2"
    log:
        "results/loo/logs/combine.log"
    params:
        root = ROOT
    script:
        "loo/combine.jl"

rule plot:
    input:
        results = "results/loo/loo_results.jld2"
    output:
        figure = "plots/loo_elpd_delta.pdf"
    log:
        "results/loo/logs/plot.log"
    params:
        root = ROOT
    script:
        "loo/plot_loo.jl"
