using Plots
using StatsPlots
using Measures
using Random
using Statistics
using Distributions

include("Models.jl")
include("Utils.jl")

# Purpose: compare two maximum-likelihood models for the 2-month counts and show,
# per follicle class, how well each reproduces the data's distribution (CDF):
#   (i)  independent Poisson for each class
#   (ii) Negative-Binomial for the total count + Multinomial sorting into classes
#        (the observation model used in total_model elsewhere)
# Overdispersion in the data should make the Poisson far too tight while the
# NB+multinomial brackets it.

labels = ["Primordial", "Primary", "Secondary"]


# ============================================================
# Raw data as stacked bars: one bar per ovary, grouped by age
# ============================================================
# Bar height = total follicle count, segments = the
# same composition (Tertiary+ included). Ovaries are sorted by total (tallest first) within each
# age group; groups are separated by a gap and labelled at their centre.

# One absolute stacked bar per ovary. `counts_by_age` is a vector of n×stage
# count matrices (one per age); returns the assembled plot.
function plot_follicle_stacks(counts_by_age, ages, stage_labels, colors;
                              age_gap=1.5, bar_width=0.85, size=(1100,450),
                              ylabel="Follicles per ovary", title="")
    xs      = Float64[]           # x position per ovary
    heights = Vector{Float64}[]   # stage counts per ovary (in plotting order)
    centers = Float64[]           # group centre for the age tick
    next_x  = 1.0
    for counts in counts_by_age
        totals = vec(sum(counts, dims=2))
        order  = sortperm(totals, rev=true)          # tallest bars first within a group
        pos    = next_x .+ (0:length(order)-1)
        append!(xs, pos)
        for i in order
            push!(heights, Float64.(counts[i, :]))
        end
        push!(centers, sum(pos)/length(pos))
        next_x = last(pos) + 1 + age_gap
    end
    Y = permutedims(hcat(heights...))                # rows = ovaries, cols = stages
    # groupedbar stacks the first column on top, so reverse the column order to
    # place the first stage (Primordial) at the bottom and Tertiary+ on top.
    Y = Y[:, end:-1:1]
    return groupedbar(xs, Y; bar_position=:stack, bar_width=bar_width,
        label=permutedims(reverse(stage_labels)), color=permutedims(reverse(colors)),
        linecolor=:white, lw=0.3, xticks=(centers, ["$a mo" for a in ages]),
        ylabel=ylabel, title=title, size=size, legend=:topright, grid=false,
        foreground_color_legend=nothing, background_color_legend=nothing,
        left_margin=5mm, bottom_margin=5mm)
end

pstack = plot_follicle_stacks(collect(extract_data(include_tertiary=true)),
    [2, 4, 6, 9, 12], [labels..., "Tertiary+"], palette(:default)[1:4];
    title="Follicle composition per ovary (absolute counts)")
savefig(pstack, "plots/initial_condition_stacks.pdf")


# ============================================================
# Maximum-likelihood fits  (3 classes only — Tertiary+ excluded)
# ============================================================
(; counts_2_month) = load_training_data()
n2 = size(counts_2_month, 1)

# Maximum-likelihood NegativeBinomial(r, p) fit. For fixed shape r the MLE of p
# is r/(r+mean(x)), so we profile out p and minimise the 1-D negative
# log-likelihood over r by golden-section search (no closed form for the shape).
function fit_nb_mle(x; r_bounds=(1e-3, 1e6))
    m = mean(x)
    negll(r) = -sum(logpdf.(NegativeBinomial(r, r/(r + m)), x))
    φ = (sqrt(5) - 1)/2
    a, b = log.(r_bounds)
    c = b - φ*(b - a); d = a + φ*(b - a)
    fc, fd = negll(exp(c)), negll(exp(d))
    for _ in 1:200
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ*(b - a); fc = negll(exp(c))
        else
            a, c, fc = c, d, fd
            d = a + φ*(b - a); fd = negll(exp(d))
        end
        b - a < 1e-10 && break
    end
    r = exp((a + b)/2)
    return NegativeBinomial(r, r/(r + m))
end

# (i) Independent Poisson: MLE rate is the per-class sample mean.
λ        = vec(mean(counts_2_month, dims=1))
pois_bin = [Poisson(λ[j]) for j in 1:3]

# (ii) NB for totals + multinomial split into classes.
totals   = vec(sum(counts_2_month, dims=2))
nb_total = fit_nb_mle(totals)
r, q     = nb_total.r, nb_total.p
θ        = (1 - q)/q                                  # Gamma scale: Λ ~ Gamma(r, θ)
pmix     = vec(sum(counts_2_month, dims=1)) ./ sum(counts_2_month)   # multinomial MLE
# Marginal of class j: y_j | Λ ~ Poisson(Λp_j), Λ ~ Gamma(r, θ)  ⇒  NB(r, 1/(1+θp_j)).
nb_bin   = [NegativeBinomial(r, 1/(1 + θ*pmix[j])) for j in 1:3]


# ============================================================
# CDF comparison figure (one panel per follicle class)
# ============================================================
# Overlay empirical CDFs of synthetic data sets (same size as the data) to show
# the sampling spread; colors match the corresponding theoretical curve.
# Poisson draws each class independently; NB+multinomial draws a total then sorts.
n_overlay = 10
pois_synth = [hcat([rand(pois_bin[j], n2) for j in 1:3]...) for _ in 1:n_overlay]
nb_synth   = [reduce(vcat, [rand(Multinomial(N, pmix))' for N in rand(nb_total, n2)])
              for _ in 1:n_overlay]

panels = Vector{Any}(undef, 3)
for j in 1:3
    data = counts_2_month[:, j]
    xmax = ceil(Int, max(maximum(data), quantile(nb_bin[j], 0.999)))
    xs   = 0:xmax

    p = plot(title=labels[j],
        xlabel=labels[j]*" follicles", ylabel="Cumulative probability",
        grid=false)

    for s in pois_synth; ecdfplot!(p, s[:, j], lc=1, lw=0.5, alpha=0.75, label=false); end
    for s in nb_synth;   ecdfplot!(p, s[:, j], lc=3, lw=0.5, alpha=0.75, label=false); end

    plot!(p, xs, cdf.(pois_bin[j], xs), lc=1, lw=2, label="Poisson")
    plot!(p, xs, cdf.(nb_bin[j],   xs), lc=3, lw=2, label="Negative Binomial")
    ecdfplot!(p, data, lc=:black, lw=2, label="Data", title=labels[j],
        xlabel=labels[j]*" follicles", ylabel="Cumulative probability",
         legend=(j == 1 ? :bottomright : false))
    panels[j] = p
end

plot(panels..., layout=(1,3), size=(1300,400), margin=5mm)
savefig("plots/initial_condition_data.pdf")


# ============================================================
# Mean–variance / covariance structure across all timepoints
# ============================================================
# Diagnostic for the Gamma–Poisson (Negative-Binomial) + multinomial observation
# model. Under that model each class marginally has
#     Var(y_j) = mean_j + mean_j^2 / r      and      Cov(y_j, y_k) = mean_j·mean_k / r,
# so on log–log axes the mean–variance points sit above the Poisson line and bend
# toward slope 2, and the covariance-vs-mean·mean points lie on a slope-1 line with
# offset log(1/r). One point per (class, timepoint) [=15] on the left; one per
# (class pair, timepoint) [=15] on the right; colored by age. The dashed line is
# the model prediction using r from the 2-month NB fit above (`r`), extended to
# all ages to check whether one overdispersion parameter is consistent throughout.
mv_counts   = collect(extract_data())            # 5 matrices, n×3 [Primordial, Primary, Secondary]
mv_ages     = [2, 4, 6, 9, 12]
class_pairs = [(1, 2), (1, 3), (2, 3)]
agecols     = palette(:default)

pmv = plot(xscale=:log10, yscale=:log10, grid=false, legend=:topleft,
           xlabel="mean(y)", ylabel="var(y)", title="Mean–variance")
pcov = plot(xscale=:log10, yscale=:log10, grid=false, legend=false,
            xlabel="mean(y₁)·mean(y₂)", ylabel="cov(y₁, y₂)", title="Covariance structure")
for (ti, c) in enumerate(mv_counts)
    m = vec(mean(c, dims=1))
    scatter!(pmv, m, vec(var(c, dims=1)); mc=agecols[ti], msw=0, ms=6, label="$(mv_ages[ti]) mo")
    for (j, k) in class_pairs
        scatter!(pcov, [m[j]*m[k]], [cov(c[:, j], c[:, k])]; mc=agecols[ti], msw=0, ms=6, label="")
    end
end

# Reference lines. Poisson (no overdispersion): var = mean. NB (2-month fit, shape
# r): var = mean + mean^2/r and cov = mean·mean/r. Points above the Poisson line
# are overdispersed; the gap to the NB line shows whether one r fits all ages.
mlo, mhi = extrema(vcat([vec(mean(c, dims=1)) for c in mv_counts]...))
m_ref = 10 .^ range(log10(mlo), log10(mhi); length=100)
plot!(pmv, m_ref, m_ref;                    lc=:black, ls=:dot,  label="Poisson (var = mean)")
plot!(pmv, m_ref, m_ref .+ m_ref.^2 ./ r;   lc=:gray,  ls=:dash, label="NB (2-mo fit)")
mm_ref = 10 .^ range(log10(mlo^2), log10(mhi^2); length=100)
plot!(pcov, mm_ref, mm_ref ./ r; lc=:gray, ls=:dash, label="")

plot(pmv, pcov, layout=(1,2), size=(1100,450), margin=5mm)
savefig("plots/initial_condition_mean_variance.pdf")
