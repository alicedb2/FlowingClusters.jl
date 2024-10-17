#!/usr/bin/env -S julia --color=yes --startup-file=no

using ArgParse
using SplitMaskStandardize
using Term
using JSON, JLD2
using Pkg

Pkg.activate(".")
using FlowingClusters

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input"
            help="Data file"
            arg_type=String
            required=true
        "--species"
            help="Species to analyze"
            arg_type=String
            required=true
        "--predictors"
            help="Predictors to use"
            nargs='+'
            arg_type=String
            required=true
        "--seed"
            help="Seed for MCMC"
            arg_type=Int
        "--dataset-seed"
            help="Seed for random number generator when splitting the dataset"
            arg_type=Int
        "--splits"
            help="Split proportions for training, validation, and testing"
            nargs=3
            arg_type=Int
            default=[2, 1, 2]
        "--subsample"
            help="Subsample size"
            arg_type=Int
        "--output-prefix"
            help="Prefix for output files"
            arg_type=String
            default="output"
        "--output-dir"
            help="Output directory"
            arg_type=String
            default="."
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    # println("Parsed args:")
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val")
    # end
    
    dataset_file = abspath(parsed_args["input"])
    species = Symbol(parsed_args["species"])
    predictors = Tuple(Symbol.(parsed_args["predictors"]))
    dataset = SMSDataset(dataset_file, seed=parsed_args["seed"], subsample=parsed_args["subsample"], splits=parsed_args["splits"])
    output_prefix = parsed_args["output-prefix"]
    output_dir = abspath(parsed_args["output-dir"])

    seed = parsed_args["seed"]
    dataset_seed = parsed_args["dataset-seed"]

    if isnothing(seed)
        seed = rand(1:10^6)
    end
    if isnothing(dataset_seed)
        dataset_seed = rand(1:10^6)
    end

    println(@green @bold "Flowing Cluster analysis")
    println(@blue("        Input file:  "), dataset_file)
    println(@blue("      Dataset seed:  "), dataset_seed)
    println(@blue("         Algo seed:  "), seed)
    println(@blue("            Splits:  "), parsed_args["splits"])
    println(@blue("     Output prefix:  "), output_prefix)
    println(@blue("  Output directory:  "), output_dir)
    println()    
    println(@blue("           Species:  "), species)
    println(@blue("        Predictors:  "), predictors)

    training_dataset = dataset.training.presence(species).standardize(predictors...)(predictors...)

    brt_eval = evaluate_brt(dataset, species, predictors)
    bioclim_eval = evaluate_bioclim(dataset, species, predictors)

    mkpath(output_dir)
    
    result_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_result.json")
    chain_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_chain.jld2")

    open(result_file, "w") do io
        write(io, json((
            dataset_seed=dataset_seed,
            dataset=dataset_file,
            trainingsize=(presence=size(dataset.training.presence(species).__df, 1), absence=size(dataset.training.absence(species).__df, 1)),
            seed=seed,
            species=species,
            predictors=predictors,
            brt_eval=brt_eval,
            bioclim_eval=bioclim_eval,
            chain_file=chain_file
        ), 4))
    end

end

include("$(pwd())/misc/bioclim_predictions.jl")
include("$(pwd())/misc/brt_predictions.jl")
include("$(pwd())/misc/flowingclusters_predictions.jl")

main()