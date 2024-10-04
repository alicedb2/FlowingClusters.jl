using CairoMakie
using Random, StatsBase, Distributions, LinearAlgebra, Optim




# ws = rand(Uniform(0, 10), 3)
ws = rand(Dirichlet([1, 1, 1, 1]))
# ws = [1, 1, 1]
# ws = ws ./ sum(ws)

mu1 = rand(Uniform(-6, 6), 2)
mu2 = rand(Uniform(-6, 6), 2)
mu3 = rand(Uniform(-6, 6), 2)
mu4 = rand(Uniform(-6, 6), 2)

dist = MixtureModel([MvNormal(mu1, diagm(rand(Uniform(0.5, 9), 2))), MvNormal(mu2, diagm(rand(Uniform(0.5, 9), 2))), MvNormal(mu3, diagm(rand(Uniform(0.5, 9), 2))), MvNormal(mu3, diagm(rand(Uniform(0.5, 9), 2)))], Categorical(ws))
y = rand(Uniform(-5, 5), 2)
isog = pdf(dist, y)
logisog = logpdf(dist, y)
# isog = pdf(dist, [-6, -4.5])
# isog = 1

xs = LinRange(-10, 10, 500)
ys = LinRange(-10, 10, 500)
zs = [pdf(dist, [x, y]) for x in xs, y in ys]
logzs = [logpdf(dist, [x, y]) for x in xs, y in ys]
zsbelow = [pdf(dist, [x, y]) >= isog ? NaN : pdf(dist, [x, y]) for x in xs, y in ys]
# zsabove = [pdf(dist, [x, y]) < isog ? NaN : pdf(dist, [x, y]) for x in xs, y in ys]
mask = logisog - 0.01 .<= logzs .<= logisog + 0.01
isocontour = [(xs[i], ys[j], isog) for i in 1:length(xs), j in 1:length(ys) if mask[i, j]]

# _y = rand(isocontour)

fig = Figure();
ax = Axis3(fig[1, 1], aspect=(2, 2, 1))
hidespines!(ax)
hidedecorations!(ax)
surface!(ax, xs, ys, zsbelow, colormap=Reverse(:blues), shading=false, colorrange=(-isog/2, isog))
scatter!(isocontour, markersize=3, color=Makie.wong_colors()[6], label="Isocontour")
scatter!(ax, [_y], markersize=20, color=:black, label="Test point")
zlims!(0, maximum(zs))
fig