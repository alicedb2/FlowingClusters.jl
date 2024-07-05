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

eb = FCDataset("data/ebird_data/ebird_bioclim_landcover.csv", 
                splits=[2, 1, 2], subsample=0.2)

# BIO1 temperature
# BIO2 meandiurnalrange
# BIO3 isothermality
# BIO4 temperatureseasonality
# BIO12 precipitation
# BIO15 precipitationseasonality

presence_chain_ = MNCRPChain(eb.training.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12), nb_samples=200)

MNCRPChain(eb.training.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12))

advance_chain!(presence_chain_, Inf; nb_splitmerge=100, nb_hyperparams=2, attempt_map=true, sample_every=:autocov)

els = elements(Matrix, presence_chain_)
coords = hcat(rand(Uniform(0.9*minimum(els[1, :]), 1.1*maximum(els[1, :])), 2000), rand(Uniform(0.9*minimum(els[2, :]), 1.1*maximum(els[2, :])), 2000))'

nn2d = Chain(
        Dense(2, 16, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
        Dense(16, 2, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
    )
presence_chain = MNCRPChain(train_presences, ffjord_nn=nn2d, nb_samples=200)
advance_chain!(presence_chain, Inf; nb_splitmerge=100, nb_hyperparams=2, attempt_map=true, sample_every=:autocov)


maptpfun = tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams)
summfun = tail_probability_summary(presence_chain.clusters_samples, presence_chain.hyperparams_samples)
maptp, summ = maptpfun(coords), summfun(coords)
mapidx = sortperm(maptp)
fdidx = sortperm(summ.modefd)
scatter(coords[1, fdidx], coords[2, fdidx], color=summ.modefd[fdidx], markersize=16)
plot(presence_chain.map_clusters)

bioclim = bioclim_predictor(reduce(hcat, train_presences.data))
bioclim_scores = bioclim(coords)
bioclimidx = sortperm(bioclim_scores)

scatter(coords[1, mapidx_], coords[2, mapidx_], color=maptp_[mapidx_], markersize=16)
scatter(coords[1, mapidx], coords[2, mapidx], color=maptp[mapidx], markersize=16)
scatter(coords[1, bioclimidx], coords[2, bioclimidx], color=bioclim_scores[bioclimidx], markersize=16)
plot(presence_chain_.map_clusters)

gaussian_tree_parameters = EvoTreeGaussian(; loss=:gaussian, metric=:gaussian, nrounds=100, nbins=100, λ=0.0, γ=0.0, η=0.1, max_depth=7, min_weight=1.0, rowsample=0.5, colsample=1.0)


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
