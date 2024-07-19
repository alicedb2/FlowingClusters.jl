module FlowingClusters

    using Random: randperm, shuffle, shuffle!, seed!, MersenneTwister, AbstractRNG, default_rng
    using StatsBase
    using StatsFuns: logsumexp, logmvgamma, logit, logistic
    using LinearAlgebra
    using StaticArrays: SVector

    using Distributions: MvNormal, MvTDist, InverseWishart, Normal,
                         Cauchy, Uniform, Exponential, Dirichlet,
                         Multinomial, Beta, MixtureModel, Categorical,
                         Distribution, logpdf, InverseGamma
    using PDMats
    using SpecialFunctions: loggamma, polygamma, logbeta

    using Makie: Figure, Axis, axislegend, lines!, vlines!, hlines!,
                 hidespines!, hidedecorations!, Cycled, scatter!, hist!

    using JLD2: jldsave, load
    using ProgressMeter: Progress, ProgressUnknown, next!
    using Optim: optimize, minimizer, LBFGS, NelderMead, Options
    using DataStructures: CircularBuffer

    using DifferentialEquations: Tsit5
    using DiffEqFlux: Lux, Chain, Dense, FFJORD, __forward_ffjord, __backward_ffjord

    using ComponentArrays: ComponentArray

    import MCMCDiagnosticTools: ess_rhat
    import Makie: plot, plot!

    include("types/diagnostics.jl")
    export Diagnostics, DiagnosticsFFJORD
    export clear_diagnostics!

    include("types/hyperparams.jl")
    export FCHyperparams, FCHyperparamsFFJORD
    export dimension, modeldimension, ij, flatk, foldL, foldpsi, flatten, niwparams
    export perturb!

    include("types/cluster.jl")
    export BitCluster, SetCluster
    export project_cluster, project_clusters
    export isvalidpartition, iscompletepartition
    export pop!, push!, find
    export availableelements, allelements

    include("types/chain.jl")
    export FCChain
    export logprob_chain, nbclusters_chain, largestcluster_chain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export nn_chain, nn_alpha_chain, nn_scale_chain
    export ess_rhat, stats


    include("model/conjugateupdates.jl")
    export log_Zniw, updated_niw_hyperparams, updated_mvstudent_params, log_cluster_weight

    include("model/hyperpriors.jl")
    include("model/modelprobabilities.jl")
    export logprobgenerative

    include("model/ffjord.jl")
    export forwardffjord, backwardffjord

    include("sampler.jl")
    export advance_gibbs!, advance_splitmerge_seq!, advance_hyperparams_adaptive!
    export advance_alpha!, advance_mu!, advance_lambda!, advance_psi!, advance_nu!
    # export advance_ffjord!, advance_nn_alpha!, advance_nn_scale!

    include("mcmc.jl")
    export advance_chain!, attempt_map!

    include("plotting.jl")
    export plot, plot!

    include("naivebioclim.jl")
    using .NaiveBIOCLIM
    export bioclim_predictor

    # include("predictions.jl")
    # export predictive_distribution, tail_probability, tail_probability_summary

    include("helpers.jl")
    export generate_data
    export performance_statistics, best_score_threshold
    export drawNIW
    export sqrtsigmoid, sqrttanh, sqrttanhgrow
    export chunk
    export logdetpsd, logdetflatLL
    export freedmandiaconis, doane
    export project_vec, project_mat

end
