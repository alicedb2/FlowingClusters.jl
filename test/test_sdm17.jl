using Pkg
Pkg.activate(".")
# using Revise
using SpeciesDistributionToolkit
using CSV
# using ProgressMeter
# using MCMCDiagnosticTools
using DataFrames
using Random
# using StatsBase
# using StatsFuns
# using LinearAlgebra
# using ColorSchemes
# using SpecialFunctions
# using Distributions
# using StatsPlots
# using Plots
# using EvoTrees
using MultivariateNormalCRP
# import CairoMakie
# import GeoMakie
# include(joinpath(pwd(), "src/helpers.jl"))
# include("src/CLR.jl")

bioclim_layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])
landcover_layernames = collect(keys(layerdescriptions(RasterData(EarthEnv, LandCover))))
deleteat!(landcover_layernames, 5)

layernames = vcat(bioclim_layernames, landcover_layernames)

ebird_df = DataFrame(CSV.File(joinpath(pwd(), "data/ebird_data/ebird_bioclim_landcover.csv"), delim="\t"))

# species = "sp1"
species = ARGS[1]

ebird_pres_dataset = MNCRPDataset(subset(ebird_df, species => x -> x .== 1.0), layernames)
ebird_abs_dataset = MNCRPDataset(subset(ebird_df, species => x -> x .== 0.0), layernames)

println(performance_scores)
println(species)

exit()

Random.seed!(7651)
train_pres, valid_pres, test_pres = split(ebird_pres_dataset, 3)
train_abs, valid_abs, test_abs = split(ebird_abs_dataset, 3)

standardize!(train_pres)
standardize!(valid_pres, with=train_pres)
standardize!(test_pres, with=train_pres)
standardize!(train_abs, with=train_pres)
standardize!(valid_abs, with=train_pres)
standardize!(test_abs, with=train_pres)


presence_chain = MNCRPChain(train_pres, chain_samples=100)
advance_chain!(presence_chain, 2048; nb_splitmerge=200, 
               sample_every=nothing, attempt_map=false)
advance_chain!(presence_chain, 4000; nb_splitmerge=200, 
               sample_every=40, attempt_map=true)

absence_chain = MNCRPChain(train_abs, chain_samples=100)
advance_chain!(absence_chain, 2048; nb_splitmerge=200, 
               sample_every=nothing, attempt_map=false)
advance_chain!(absence_chain, 4000; nb_splitmerge=200, 
               sample_every=40, attempt_map=true)

