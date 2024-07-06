using Plots
using Plots: mm
using StatsBase: Weights
using Distributions: Normal, Uniform
using Optim: optimize, minimizer

function g(x, mu1, mu2, sigma1, sigma2, w1, w2) 
    w1, w2 = w1/(w1 + w2), w2/(w1 + w2)
    n1 = w1 / sqrt(2 * pi) / sigma1 * exp(-1/2/sigma1^2 * (x - mu1)^2)
    n2 = w2 / sqrt(2 * pi)/sigma2 * exp(-1/2/sigma2^2 * (x - mu2)^2)
    return n1 + n2
end


mu1, mu2, sigma1, sigma2, w1, w2 = -5, 4, 2, 3, 2, 4
params = mu1, mu2, sigma1, sigma2, w1, w2

y = -3
gy = g(y, params...)
x1 = minimizer(optimize(x -> (g(x, params...) - gy)^2, -10, -5))
x2 = minimizer(optimize(x -> (g(x, params...) - gy)^2, 0, 2))
x3 = minimizer(optimize(x -> (g(x, params...) - gy)^2, 5, 10))

#################

xi, xf = -15, 15
xs = LinRange(xi, xf, 10000)
p = plot(legend=:topleft, size=(800, 400), left_margin=6Plots.mm, bottom_margin=6Plots.mm);
plot!(xs, g.(xs, params...), label=nothing, c=:black, w=3);
for x in [y, x1, x2, x3]
    plot!([x1, x1], [0, g(x, params...)], label=nothing);
end
hline!([gy], w=1, s=:dash, label=nothing);

xs1 = LinRange(xi, x1, 100)
plot!(xs1, zeros(size(xs1)), fillrange=g.(xs1, params...), fillalpha=0.35, lw=0, c=:green, label="Presence probability");
xs2 = LinRange(y, x2, 100)
plot!(xs2, zeros(size(xs2)), fillrange=g.(xs2, params...), fillalpha=0.35, lw=0, c=:green, label=nothing);
xs3 = LinRange(x3, xf, 100)
plot!(xs3, zeros(size(xs3)), fillrange=g.(xs3, params...), fillalpha=0.35, lw=0, c=:green, label=nothing);
scatter!([x1, y, x2, x3], [gy, gy, gy, gy], msw=0, c=7, ms=6, label="Isocontour");
xlabel!("Environmental space X");
ylabel!(L"g(x\vert\pi_D)");
xlims!(xi, xf);

nb_samples = 10000
Xs = zeros(nb_samples)
Ys = zeros(nb_samples)
for i in 1:nb_samples
    dist = sample([Normal(mu1, sigma1), Normal(mu2, sigma2)], Weights([w1, w2]))
    Xs[i] = rand(dist)
    Ys[i] = rand(Uniform(0, g(Xs[i], params...)))
end

scatter!(Xs[Xs .< x1], Ys[Xs .< x1], msw=0, c=9, label=nothing);
scatter!(Xs[x1 .< Xs .< y], Ys[x1 .< Xs .< y], msw=0, c=9, ma=0.2, label=nothing);
scatter!(Xs[y .< Xs .< x2], Ys[y .< Xs .< x2], msw=0, c=9, label=nothing);
scatter!(Xs[x2 .< Xs .< x3], Ys[x2 .< Xs .< x3], msw=0, c=9, ma=0.2, label=nothing);
scatter!(Xs[Xs .> x3], Ys[Xs .> x3], msw=0, c=9, 
         label=L"g(\hat X\vert\pi_D) < g(y\vert\pi_D)");

annotate!(y+0.5, gy+0.005, "y", :black);
title!("Presence probability at point y");
display("image/png", p)

#################

nb_samples = 10000
Xs = zeros(nb_samples)
Ys = zeros(nb_samples)
for i in 1:nb_samples
    dist = sample([Normal(mu1, sigma1), Normal(mu2, sigma2)], Weights([w1, w2]))
    Xs[i] = rand(dist)
    Ys[i] = rand(Uniform(0, g(Xs[i], params...)))
end

ss = zeros(length(xs))
for i in 1:length(xs)
    ss[i] = sum(g.(Xs, params...) .< g(xs[i], params...))/length(Xs)
end

logodd(p) = log(p) - log(1 - p)

p = plot(left_margin=6mm, bottom_margin=6mm,
right_margin=12mm, top_margin=6mm);
# plot!(xs, ss, label=nothing, lw=3);
plot!(xs, logodd.(ss), label=nothing, lw=3);
plot!(twinx(), xs, g.(xs, params...), label=nothing, lw=3, color=:black)

#################

elements = [x for cluster in chain.clusters for x in cluster];
elements_summaries = tail_probability_summary(elements, chain, 1000)

scatter(Tuple.(elements), marker_z=elements_summaries.median, msw=0, ms=3, color=:viridis, label="Presence probability")
scatter(Tuple.(elements), marker_z=elements_summaries.iqr, msw=0, ms=3, color=:viridis, label="Presence probability")

logodd(p) = log(p) - log(1 - p)
elements_presprobs = elements_summaries.median
elements_preslogodds = logodd.(elements_presprobs)
scatter(Tuple.(elements[isfinite.(elements_preslogodds)]), marker_z=elements_preslogodds[isfinite.(elements_preslogodds)], msw=0, ms=3, color=:viridis, label="Presence log odd", legend=:topleft)

corr_parcorr = response_correlation(elements, chain)
