using BenchmarkTools
using CairoMakie
using FlowingClusters
using SpeciesDistributionToolkit
# using GeoMakie
using Distributions
using LinearAlgebra
using Random
using CSV, DataFrames
using DiffEqFlux
using DiffEqFlux: __backward_ffjord
using DifferentialEquations
using ComponentArrays: ComponentArray

function deco!(axis)
    hidespines!(axis, :t, :r)
    hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
    return axis
end

include("misc/ebird.jl")

presence_chain_ = MNCRPChain(train_presences, chain_samples=200)
advance_chain!(presence_chain_, 2000; nb_splitmerge=100, nb_hyperparams=2, attempt_map=true, sample_every=:autocov)

nn2d = Chain(
        Dense(2, 16, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(16, 2, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
    )
presence_chain = MNCRPChain(train_presences, ffjord_nn=nn2d, chain_samples=200)
advance_chain!(presence_chain, 5000; nb_splitmerge=100, nb_hyperparams=1, ffjord_sampler=:am, attempt_map=true, sample_every=:autocov)


nn3d = Chain(
        Dense(3, 5, sqrttanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(5, 5, sqrttanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(6, 3, x->x*abs(sqrttanh(x)), init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        # Dense(5, 3, x->x + sqrttanh(x), init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        # Dense(5, d, x->6f0*sqrttanh(x/6f0), init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
    )
    
presence_chain3 = MNCRPChain(train_presences, chain_samples=100, ffjord_nn=nn3d)
advance_chain!(presence_chain3, 1000; nb_splitmerge=100, nb_hyperparams=1,
                ffjord_sampler=:am,
                sample_every=nothing, attempt_map=true)

    


ffjord_mdl = FFJORD(presence_chain.hyperparams.nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())
ps, st = presence_chain.map_hyperparams.nn_params, presence_chain.map_hyperparams.nn_state
# ps, st = presence_chain.hyperparams.nn_params, presence_chain.hyperparams.nn_state
# dims = [1, 2, 3]
# for z in LinRange(-3, 3, 14)
    probgridys = zeros(2, 0)
    probgridys = reduce(hcat, [[x, y] for x in LinRange(-4, 4, 10), y in LinRange(-4, 4, 100)])
    probgridxs = zeros(2, 0)
    probgridxs = reduce(hcat, [[x, y] for y in LinRange(-4, 4, 10), x in LinRange(-4, 4, 100)])
    probgrid = hcat(probgridys, probgridxs)
    # probgrid = vcat(probgrid, fill(z, 1, size(probgrid, 2)))
    # probgrid = probgrid[dims, :]

    ret, _ = ffjord_mdl(probgrid, ps, st); basespace = ret.z
    realspace = __backward_ffjord(ffjord_mdl, probgrid, ps, st)

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
    display(fig)
# end