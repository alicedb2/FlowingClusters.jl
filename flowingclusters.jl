#!/usr/bin/env -S julia +1.10 --color=yes --startup-file=no

using Pkg
using ArgParse
using Term

Pkg.activate(@__DIR__, io=devnull)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input", "-i"
            help="Data file"
            arg_type=String
            required=true
        "--species", "-s"
            help="Species to analyze"
            arg_type=String
            required=true
        "--predictors", "-p"
            help="Predictors to use"
            nargs='+'
            arg_type=String
            required=true
        "--nb-mcmc-samples", "-m"
            help="Number of MCMC samples"
            arg_type=Int
            default=200
        "--nb-iter", "-n"
            help="Number of iterations"
            arg_type=Int
            default=-1
        "--nb-gibbs", "-g"
            help="Number of Gibbs sweep per iteration"
            arg_type=Int
            default=1
        "--nb-splitmerge", "-c"
            help="Number of split-merge attempts per iteration"
            arg_type=Int
            default=350
        "--nb-amwg", "-w"
            help="Number of adaptive Metropolis within Gibbs per iteration"
            arg_type=Int
            default=1
        "--hidden-nodes", "-H"
            help="Activate FFJORD, hidden layer sizes (off by default)"
            nargs='*'
            arg_type=Int
        "--activation-function", "-f"
            help="Activation function (tanh_fast, softsign, relu, sigmoid_fast, swish, etc.)"
            arg_type=String
            default="tanh_fast"
        "--nb-ffjord-am", "-a"
            help="Number of adaptive Metropolis per iteration"
            arg_type=Int
            default=1
        "--pre-training"
            help="Number of iteration of pre-training over neural network"
            arg_type=Int
            default=0
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
        "--output-dir", "-o"
            help="Output directory"
            arg_type=String
            default="."
        "--progress-output"
            help="Output progress"
            arg_type=String
            default="repl"
        # "--evaluate-fc"
        #     help="Evaluate FlowingClusters model on data"
        #     action=:store_true
        "--evaluate-bioclim"
            help="Evaluate BIOCLIM model on data"
            action=:store_true
        "--evaluate-brt"
            help="Evaluate BRT model on data"
            action=:store_true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

print("Initializing FlowingClusters.jl...")
using JSON, JLD2
using SplitMaskStandardize
using FlowingClusters
println("done!")

function main(parsed_args)

    dataset_file = abspath(parsed_args["input"])
    output_prefix = parsed_args["output-prefix"]
    output_dir = abspath(parsed_args["output-dir"])

    seed = parsed_args["seed"]
    dataset_seed = parsed_args["dataset-seed"]

    species = Symbol(parsed_args["species"])
    predictors = Tuple(Symbol.(parsed_args["predictors"]))

    nb_mcmc_samples = parsed_args["nb-mcmc-samples"]

    if isnothing(seed)
        seed = rand(1:10^8)
    end
    if isnothing(dataset_seed)
        dataset_seed = rand(1:10^8)
    end

    dataset = SMSDataset(dataset_file, seed=dataset_seed, subsample=parsed_args["subsample"], splits=parsed_args["splits"])

    if parsed_args["progress-output"] == "repl"
        progressoutput = :repl
    elseif parsed_args["progress-output"] == "file"
        progressoutput = :file
    else
        progressoutput = joinpath(output_dir, parsed_args["progress-output"])
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
    _act = parsed_args["activation-function"]
    nb_pretraining = parsed_args["pre-training"]
    if !isempty(hn)
        println(@blue("     Pre-training:  "), nb_pretraining)
        println(@blue("     Hidden nodes:  "), hn)
        println(@blue("   Activation fun:  "), _act)

        if hasproperty(FlowingClusters, Symbol(_act))
            act = getproperty(FlowingClusters, Symbol(_act))
        elseif hasproperty(FlowingClusters.Lux, Symbol(_act))
            act = getproperty(FlowingClusters.Lux, Symbol(_act))
        else
            error("Activation function $_act not found. Available functions can be found here: https://lux.csail.mit.edu/stable/api/NN_Primitives/ActivationFunctions")
        end

        hn = [length(predictors), hn..., length(predictors)]
        nn = dense_nn(hn...; act=act)
    end

    training_dataset = dataset.training.presence(species).standardize(predictors...)(predictors...)

    bioclim_eval = parsed_args["evaluate-bioclim"] ? evaluate_bioclim(dataset, species, predictors) : nothing
    brt_eval = parsed_args["evaluate-brt"] ? evaluate_brt(dataset, species, predictors, halftraining=false) : nothing
    brt_eval_halftraining = parsed_args["evaluate-brt"] ? evaluate_brt(dataset, species, predictors, halftraining=true) : nothing

    mkpath(output_dir)

    if isnothing(nn)
        chain = FCChain(training_dataset, SetCluster, nb_samples=nb_mcmc_samples, strategy=:sequential, seed=seed)
    else
        chain = FCChain(training_dataset, SetCluster, nb_samples=nb_mcmc_samples, strategy=:1, ffjord_nn=nn, seed=seed)
        if nb_pretraining > 0
        # Pre-training neural network
            advance_chain!(chain, nb_pretraining,
                nb_ffjord_am=1, nb_amwg=0, nb_gibbs=0, nb_splitmerge=0,
                attempt_map=false, sample_every=nothing,
                progressoutput=progressoutput)
            burn!(chain, nb_pretraining)
            sequential_gibbs!(chain.rng, chain.clusters, chain.hyperparams)
        end
    end
    println(chain)

    fc_eval_init = evaluate_flowingclusters(chain, dataset, species, predictors)

    advance_chain!(chain, parsed_args["nb-iter"],
        nb_ffjord_am=parsed_args["nb-ffjord-am"],
        nb_gibbs=parsed_args["nb-gibbs"],
        nb_amwg=parsed_args["nb-amwg"],
        nb_splitmerge=parsed_args["nb-splitmerge"],
        attempt_map=true, sample_every=:autocov, stop_criterion=:sample_ess,
        progressoutput=progressoutput)

    fc_eval = evaluate_flowingclusters(chain, dataset, species, predictors)

    result_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_result.json")
    chain_file = joinpath(output_dir, "$(output_prefix)_pid$(getpid())_chain.jld2")

    results = (
        parsed_args=parsed_args,
        dataset=dataset_file,
        brt_eval=brt_eval,
        brt_eval_halftraining=brt_eval_halftraining,
        bioclim_eval=bioclim_eval,
        fc_eval_init=fc_eval_init,
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

if parsed_args["evaluate-bioclim"]
    include(joinpath(@__DIR__, "misc", "bioclim_predictions.jl"))
end
if parsed_args["evaluate-brt"]
    include(joinpath(@__DIR__, "misc", "brt_predictions.jl"))
end
include(joinpath(@__DIR__, "misc", "flowingclusters_predictions.jl"))

main(parsed_args)