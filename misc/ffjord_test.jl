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

function deco!(axis)
    hidespines!(axis, :t, :r)
    hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
    return axis
end

ebird_df = DataFrame(CSV.File("data/ebird_data/ebird_bioclim_landcover.csv", delim="\t"))
_left, _right, _bottom, _top = (1.1 * minimum(ebird_df.lon), 
                            0.9 * maximum(ebird_df.lon),
                            0.9 * minimum(ebird_df.lat), 
                            1.1 * maximum(ebird_df.lat))
lon_0, lat_0 = (_left + _right)/2, (_top + _bottom)/2
_left, _right, _bottom, _top = max(_left, -180.0), min(_right, 180.0), max(_bottom, -90.0), min(_top, 90.0)

eb_subsample = reduce(push!, sample(eachrow(ebird_df), 3000, replace=false), init=DataFrame())
# eb_subsample = ebird_df
eb_subsample_presences = subset(eb_subsample, :sp1 => x -> x .== 1.0)
eb_subsample_absences = subset(eb_subsample, :sp1 => x -> x .== 0.0)

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

presence_chain2 = MNCRPChain(train_presences, chain_samples=100)
advance_chain!(presence_chain2, 1000; nb_splitmerge=50, nb_hyperparams=1, attempt_map=true)

# nn = Chain(Dense(2, 10, tanh_fast), Dense(10, 10, tanh_fast), Dense(10, 2, tanh_fast))
nn = Chain(
        Dense(2, 4, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(4, 4, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(4, 2, identity, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
    )

presence_chain = MNCRPChain(train_presences, chain_samples=100, ffjord_nn=nn)
advance_chain!(presence_chain, 1000; nb_splitmerge=50, nb_hyperparams=1,
               ffjord_sampler=:am,
               sample_every=nothing, attempt_map=true)

ffjord_mdl = FFJORD(presence_chain.hyperparams.nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())
# ps, st = presence_chain.hyperparams.nn_params, presence_chain.hyperparams.nn_state
ps, st = presence_chain.map_hyperparams.nn_params, presence_chain.map_hyperparams.nn_state

probgridys = zeros(2, 0)
probgridys = reduce(hcat, [[x, y] for x in LinRange(-4, 4, 10), y in LinRange(-4, 4, 100)])
probgridxs = zeros(2, 0)
probgridxs = reduce(hcat, [[x, y] for y in LinRange(-4, 4, 10), x in LinRange(-4, 4, 100)])
probgrid = hcat(probgridys, probgridxs)
ret, _ = ffjord_mdl(probgrid, ps, st)
basespace = ret.z
realspace = DiffEqFlux.__backward_ffjord(ffjord_mdl, probgrid, ps, st)

fig = Figure(size=(900, 500));
ax1 = Axis(fig[1, 1], xlabel="Temperature -> Base axis 1", ylabel="Annual Precipitation -> Base axis 2")
scatter!(ax1, basespace[1, :], basespace[2, :], markersize=3, label=nothing); scatter!(ax1, probgrid[1, :], probgrid[2, :], color=:grey, alpha=0.5, markersize=3, label=nothing);
scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real/environmental space"); 
scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Base space"); 
ylims!(ax1, nothing, 7); axislegend(ax1, framecolor=:white); 
ax2 = Axis(fig[1, 2], xlabel="Base axis 1 -> Temperature", ylabel="Base axis 2 -> Annual Precipitation")
scatter!(ax2, realspace[1, :], realspace[2, :], markersize=3, label=nothing); scatter!(ax2, probgrid[1, :], probgrid[2, :], color=:grey, alpha=0.5, markersize=3, label=nothing);
scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Real/environmental space");
ylims!(ax2, nothing, 7); axislegend(ax2, framecolor=:white);
deco!(ax1); deco!(ax2);
fig

nn_D = size(ps, 1)
nn_chain = reduce(hcat, [h.nn_params for h in presence_chain.hyperparams_chain])
fig = Figure();
ax = Axis(fig[1, 1])
for r in eachrow(nn_chain)
    lines!(ax, r)
end
fig
