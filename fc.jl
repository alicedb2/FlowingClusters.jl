#!/usr/bin/env -S julia --color=yes --startup-file=no

using ArgParse, Term
using SplitMaskStandardize
using Distributions: Uniform
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
        "--nb-mcmc-samples"
            help="Number of MCMC samples"
            arg_type=Int
            default=200
        "--nb-iter"
            help="Number of iterations"
            arg_type=Int
            default=-1
        "--nb-gibbs"
            help="Number of Gibbs sweep per iteration"
            arg_type=Int
            default=1
        "--nb-splitmerge"
            help="Number of split-merge attempts per iteration"
            arg_type=Int
            default=350
        "--nb-amwg"
            help="Number of adaptive Metropolis within Gibbs per iteration"
            arg_type=Int
            default=1
        "--hidden-nodes"
            help="Activate FFJORD, hidden layer sizes (off by default)"
            nargs='*'
            arg_type=Int
        "--nb-ffjord-am"
            help="Number of adaptive Metropolis per iteration"
            arg_type=Int
            default=1
        "--pre-training"
            help="Pre-training neural network"
            arg_type=Int
            default=6000
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
        "--progress-output"
            help="Output progress"
            arg_type=Bool
            default=false
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    
    dataset_file = abspath(parsed_args["input"])
    dataset = SMSDataset(dataset_file, seed=parsed_args["seed"], subsample=parsed_args["subsample"], splits=parsed_args["splits"])
    output_prefix = parsed_args["output-prefix"]
    output_dir = abspath(parsed_args["output-dir"])
    
    seed = parsed_args["seed"]
    dataset_seed = parsed_args["dataset-seed"]
    
    species = Symbol(parsed_args["species"])
    predictors = Tuple(Symbol.(parsed_args["predictors"]))
    nb_mcmc_samples = parsed_args["nb-mcmc-samples"]

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
    println(@blue("    # MCMC samples:  "), nb_mcmc_samples)
    
    nn = nothing
    hn = parsed_args["hidden-nodes"]
    nb_pretraining = parsed_args["pre-training"]
    if !isempty(hn)
        println(@blue("     Pre-training:  "), nb_pretraining)
        println(@blue("     Hidden nodes:  "), hn)

        hn = [length(predictors), hn..., length(predictors)]
        nn = Chain(
            [Dense(hn[i], hn[i+1], 
                   softsign, 
                   init_bias=(i + 1 == length(hn) ? zeros64 : initbias(-4, 4)), 
                   init_weight=kaiming_uniform(gain=0.1)
                ) for i in 1:length(hn)-1]...
        )
    end

    training_dataset = dataset.training.presence(species).standardize(predictors...)(predictors...)

    brt_eval = evaluate_brt(dataset, species, predictors)
    bioclim_eval = evaluate_bioclim(dataset, species, predictors)

    if isnothing(nn)
        chain = FCChain(training_dataset, SetCluster, nb_samples=nb_mcmc_samples, strategy=:sequential, seed=seed)
    else
        chain = FCChain(training_dataset, SetCluster, nb_samples=nb_mcmc_samples, strategy=:1, ffjord_nn=nn, seed=seed)
        # Pre-training neural network
        advance_chain!(chain, nb_pretraining, 
            nb_ffjord_am=1, nb_amwg=0, nb_gibbs=0, nb_splitmerge=0,
            attempt_map=false, sample_every=nothing)
        burn!(chain, nb_pretraining)
    end
    println(chain)

    advance_chain!(chain, parsed_args["nb-iter"],
        nb_ffjord_am=parsed_args["nb-ffjord-am"],
        nb_gibbs=parsed_args["nb-gibbs"],
        nb_amwg=parsed_args["nb-amwg"],
        nb_splitmerge=parsed_args["nb-splitmerge"],
        attempt_map=true, sample_every=:autocov, stop_criterion=:sample_ess)

    fc_eval = evaluate_flowingclusters(chain, dataset, species, predictors)

    mkpath(output_dir)
    
    result_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_result.json")
    chain_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_chain.jld2")

    results = (
        parsed_args=parsed_args,
        dataset=dataset_file,
        brt_eval=brt_eval,
        bioclim_eval=bioclim_eval,
        fc_eval=fc_eval,
        trainingsize=(presence=size(dataset.training.presence(species).__df, 1), absence=size(dataset.training.absence(species).__df, 1)),
        chain_file=chain_file
    )

    open(result_file, "w") do io
        write(io, json(results, 4))
    end

    jldsave(chain_file; chain=chain)
    run(`gzip -f $chain_file`)

    return results
end

include("$(pwd())/misc/bioclim_predictions.jl")
include("$(pwd())/misc/brt_predictions.jl")
include("$(pwd())/misc/flowingclusters_predictions.jl")

function initbias(lower, upper)
    fun(rng, outdims, _) = rand(rng, Uniform(lower, upper), outdims, 1)
    return fun
end
        
main()