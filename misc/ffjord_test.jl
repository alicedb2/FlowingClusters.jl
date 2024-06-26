using BenchmarkTools
using CairoMakie
using MultivariateNormalCRP
using SpeciesDistributionToolkit
# using GeoMakie
using Distributions
using Random
using CSV, DataFrames
using DiffEqFlux
using DifferentialEquations
using ComponentArrays: ComponentArray

ebird_df = DataFrame(CSV.File("data/ebird_data/ebird_bioclim_landcover.csv", delim="\t"))
_left, _right, _bottom, _top = (1.1 * minimum(ebird_df.lon), 
                            0.9 * maximum(ebird_df.lon),
                            0.9 * minimum(ebird_df.lat), 
                            1.1 * maximum(ebird_df.lat))
lon_0, lat_0 = (_left + _right)/2, (_top + _bottom)/2
_left, _right, _bottom, _top = max(_left, -180.0), min(_right, 180.0), max(_bottom, -90.0), min(_top, 90.0)

eb_subsample = reduce(push!, sample(eachrow(ebird_df), 3000, replace=false), init=DataFrame())
# eb_subsample = ebird_df
eb_subsample_presences = subset(eb_subsample, :sp2 => x -> x .== 1.0)
eb_subsample_absences = subset(eb_subsample, :sp2 => x -> x .== 0.0)

# bioclim_layernames = "BIO" .* string.(1:19)

# Temperature, Annual precipitations
_layernames = "BIO" .* string.([1, 12])
# Temperature, Isothermality, Annual Precipitation
# _layernames = "BIO" .* string.([1, 3, 12])
# Temperature, Mean Diurnal Range, Isothermality, Temperature Seasonality, Annual Precipitation, Precipitation Seasonality
# _layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])

_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), 
                                     layer=layer, resolution=10.0,
                                     left=_left, right=_right, bottom=_bottom, top=_top)
                 for layer in _layernames]

dataset = MNCRPDataset(eb_subsample_presences, _layers, longlatcols=["lon", "lat"], layernames=_layernames)
abs_dataset = MNCRPDataset(eb_subsample_absences, _layers, longlatcols=["lon", "lat"], layernames=_layernames)

train_presences, validation_presences, test_presences = split(dataset, 3)
train_absences, validation_absences, test_absences = split(abs_dataset, 3)

standardize!(train_presences)
standardize!(validation_presences, with=train_presences)
standardize!(test_presences, with=train_presences)
standardize!(train_absences, with=train_presences)
standardize!(validation_absences, with=train_presences)
standardize!(test_absences, with=train_presences)

presence_chain = MNCRPChain(train_presences, chain_samples=100)

advance_chain!(presence_chain, 512; nb_splitmerge=100, nb_hyperparams=3, sampler=:amwg, attempt_map=false)
plot(presence_chain)
advance_chain!(presence_chain, 512; nb_splitmerge=250, nb_hyperparams=3, sampler=:amwg, attempt_map=true)


# nn = Chain(Dense(2, 10, tanh_fast), Dense(10, 10, tanh_fast), Dense(10, 2, tanh_fast))
nn = Chain(
        Dense(2, 4, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(4, 4, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(4, 2, identity, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
    )

ffjord_mdl = FFJORD(nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())
ps, st = Lux.setup(Xoshiro(), ffjord_mdl)
ps = ComponentArray(ps)

ffjord_mdl = FFJORD(presence_chain.hyperparams.nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())
ps, st = presence_chain.map_hyperparams.nn_params, presence_chain.map_hyperparams.nn_state
ps, st = presence_chain.hyperparams.nn_params, presence_chain.hyperparams.nn_state

ret, _ = ffjord_mdl(rand(Uniform(-2, 2), 2, 1000), ps, st)
bar = DiffEqFlux.__backward_ffjord(ffjord_mdl, rand(Uniform(-2, 2), 2, 1000), ps, st)
scatter(ret.z[1, :], ret.z[2, :])
scatter(bar[1, :], bar[2, :])


presence_chain = MNCRPChain(train_presences, chain_samples=100, ffjord_nn=nn)
# plot(presence_chain.map_clusters)

advance_chain!(presence_chain, 256; nb_splitmerge=50, 
               ffjord_every=nothing,
               sample_every=nothing, attempt_map=false)
advance_chain!(presence_chain, 256; nb_splitmerge=50, 
               ffjord_every=nothing,
               sample_every=nothing, attempt_map=true)
advance_chain!(presence_chain, 512; nb_splitmerge=50, 
               ffjord_every=:map, ffjord_nbiter=30,
               sample_every=nothing, attempt_map=true)

advance_ffjord!(presence_chain.clusters, presence_chain.hyperparams, presence_chain.data2original, nbiter=100, training_fraction=0.9)

advance_chain!(presence_chain, 200; nb_splitmerge=50, 
               ffjord_every=:map, ffjord_nbiter=77, ffjord_training_fraction=0.5,
               sample_every=20, attempt_map=true)


advance_chain!(presence_chain, 256; nb_splitmerge=50, 
               sample_every=nothing, attempt_map=false)
advance_chain!(presence_chain, 256; nb_splitmerge=50, 
               sample_every=20, attempt_map=true)
plot(presence_chain.map_clusters)
# plot(presence_chain)

advance_ffjord!(presence_chain.map_clusters, presence_chain.map_hyperparams)


dist = predictive_distribution(presence_chain.map_clusters, presence_chain.map_hyperparams)

# nn = Chain(Dense(2, 10, tanh_fast), Dense(10, 10, tanh_fast), Dense(10, 2, tanh_fast))
nn = Chain(
    Dense(2, 4, tanh_fast, init_bias=zeros32, init_weight=identity_init), 
    Dense(4, 4, tanh_fast, init_bias=zeros32, init_weight=identity_init), 
    Dense(4, 2, tanh_fast, init_bias=zeros32, init_weight=identity_init)
    )
ffjord_mdl = FFJORD(nn, (0.0, 10.0), (2,), Tsit5(), ad=AutoForwardDiff())
                    # basedist=dist)
ps, st = Lux.setup(Xoshiro(), ffjord_mdl)

unittransform = DiffEqFlux.__backward_ffjord(ffjord_mdl, rand(Uniform(), 2, 10000), ps, st)
scatter(unittransform[1, :], unittransform[2, :])

foo = DiffEqFlux.__forward_ffjord(ffjord_mdl, rand(2, 1000), ps, st)


ps, st = Lux.setup(Xoshiro(), ffjord_mdl)
DiffEqFlux.__forward_ffjord(ffjord_mdl, rand(2, 10), ps, st)
# ps = ComponentArray(ps)
ffdist = FFJORDDistribution(ffjord_mdl, ps, st)

foo = rand(2, 100)
@btime DiffEqFlux.__forward_ffjord(ffjord_mdl, foo, ps, st)

# ffjord_mdl = FFJORD(presence_chain.map_hyperparams.nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())

foo = rand(ffdist, 100)
scatter(foo[1, :], foo[2, :])

bar = rand(dist, 100)
scatter(bar[1, :], bar[2, :])



nn = Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
tspan = (0.0f0, 10.0f0)

ffjord_mdl = FFJORD(nn, tspan, (1,), Tsit5(); ad = AutoZygote())
ps, st = Lux.setup(Xoshiro(0), ffjord_mdl)
ps = ComponentArray(ps)
# model = StatefulLuxLayer{true}(ffjord_mdl, ps, st)
model = StatefulLuxLayer{true}(ffjord_mdl, ps, st)

# Training
data_dist = Normal(6.0f0, 0.7f0)
train_data = Float32.(rand(data_dist, 1, 100))

function loss(θ)
    logpx, λ₁, λ₂ = model(train_data, θ)
    return -mean(logpx)
end

function cb(p, l)
    @info "FFJORD Training" loss=l
    return false
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)

res1 = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.01); maxiters = 20, callback = cb)