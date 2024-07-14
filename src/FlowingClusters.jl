module FlowingClusters

    using Random: randperm, shuffle, shuffle!, seed!, MersenneTwister, AbstractRNG, default_rng
    using StatsBase
    using StatsFuns: logsumexp, logmvgamma, logit, logistic
    using LinearAlgebra

    using Distributions: MvNormal, MvTDist, InverseWishart, Normal,
                         Cauchy, Uniform, Exponential, Dirichlet,
                         Multinomial, Beta, MixtureModel, Categorical,
                         Distribution, logpdf, InverseGamma
    using PDMats
    using SpecialFunctions: loggamma, polygamma, logbeta

    using Makie: Figure, Axis, axislegend, lines!, vlines!, hlines!,
                 hidespines!, hidedecorations!, Cycled, scatter!, hist!

    using JLD2
    using ProgressMeter: Progress, ProgressUnknown, next!
    using Optim: optimize, minimizer, LBFGS, NelderMead, Options
    using DataStructures: CircularBuffer

    using DifferentialEquations
    using DiffEqFlux
    using DiffEqFlux: __forward_ffjord, __backward_ffjord

    using ComponentArrays: ComponentArray

    import MCMCDiagnosticTools: ess_rhat
    import Makie: plot, plot!

    include("types/diagnostics.jl")
    export Diagnostics, DiagnosticsFFJORD
    export clear_diagnostics!, am_sigma

    include("types/hyperparams.jl")
    export AbstractFCHyperparams, FCHyperparams, FCHyperparamsFFJORD
    export dimension, modeldimension, ij, flatk, foldL, foldpsi, flatten, niwparams
    export perturb!

    include("types/cluster.jl")
    export AbstractCluster, BitCluster, SetCluster, EmptyCluster
    export project_cluster, project_clusters
    export isvalidpartition, iscompletepartition
    export pop!, push!, find
    export availableelements, allelements

    include("conjugateupdates.jl")
    export log_Zniw, updated_niw_hyperparams, updated_mvstudent_params, log_cluster_weight

    include("hyperpriors.jl")
    include("modelprobabilities.jl")
    export logprobgenerative

    include("ffjord.jl")
    export forwardffjord, backwardffjord

    include("sampler.jl")
    export advance_gibbs!, advance_splitmerge_seq!, advance_hyperparams_adaptive!
    export advance_alpha!, advance_mu!, advance_lambda!, advance_psi!, advance_nu!
    # export advance_ffjord!, advance_nn_alpha!, advance_nn_scale!

    include("types/chain.jl")
    export FCChain
    export logprob_chain, nbclusters_chain, largestcluster_chain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export nn_chain, nn_alpha_chain, nn_scale_chain
    export ess_rhat, stats

    include("mcmc.jl")
    export advance_chain! #, attempt_map!

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

    # function optimize_hyperparams(
    #     clusters::Vector{Cluster},
    #     hyperparams0::MNCRPHyperparams;
    #     jacobian=false, verbose=false
    #     )

    #     objfun(x) = -logprobgenerative(clusters, x, jacobian=jacobian)

    #     x0 = unpack(hyperparams0)

    #     if verbose
    #         function callback(x)
    #             print(" * Iter $(x.iteration),   objfun $(-round(x.value, digits=2)),   g_norm $(round(x.g_norm, digits=8))\r")
    #             return false
    #         end
    #     else
    #         callback =  nothing
    #     end
    #     opt_options = Options(iterations=50000,
    #                           x_tol=1e-8,
    #                           f_tol=1e-6,
    #                           g_tol=2e-2,
    #                           callback=callback)

    #     optres = optimize(objfun, x0, NelderMead(), opt_options)

    #     if verbose
    #         println()
    #     end

    #     opt_hp = pack(minimizer(optres))

    #     return MNCRPHyperparams(opt_hp..., deepcopy(hyperparams0.diagnostics), hyperparams0.nn, hyperparams0.nn_params, hyperparams0.nn_state)

    # end

    # function optimize_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; jacobian=false, verbose=false)

    #     opt_res = optimize_hyperparams(clusters, hyperparams, jacobian=jacobian, verbose=verbose)

    #     hyperparams.alpha = opt_res.alpha
    #     hyperparams.mu = opt_res.mu
    #     hyperparams.lambda = opt_res.lambda
    #     hyperparams.flatL = opt_res.flatL
    #     hyperparams.L = opt_res.L
    #     hyperparams.psi = opt_res.psi
    #     hyperparams.nu = opt_res.nu

    #     return hyperparams

    # end

    # function updated_mvstudent_params(
    #     ::Nothing,
    #     mu::AbstractVector{Float64},
    #     lambda::Float64,
    #     psi::AbstractMatrix{Float64},
    #     nu::Float64
    #     )::Tuple{Float64, Vector{Float64}, Matrix{Float64}}

    #     d = length(mu)

    #     return (nu - d + 1, mu, (lambda + 1)/lambda/(nu - d + 1) * psi)

    # end

    # function updated_mvstudent_params(
    #     cluster::Cluster,
    #     mu::AbstractVector{Float64},
    #     lambda::Float64,
    #     psi::AbstractMatrix{Float64},
    #     nu::Float64
    #     )::Tuple{Float64, Vector{Float64}, Matrix{Float64}}

    #     d = length(mu)
    #     mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

    #     return (nu_c - d + 1, mu_c, (lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c)

    # end

    # function updated_mvstudent_params(
    #     clusters::Vector{Cluster},
    #     mu::AbstractVector{Float64},
    #     lambda::Float64,
    #     psi::AbstractMatrix{Float64},
    #     nu::Float64;
    #     add_empty=true
    #     )::Vector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}}

    #     updated_mvstudent_degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]
    #     if add_empty
    #         push!(updated_mvstudent_degs_mus_sigs, updated_mvstudent_params(nothing, mu, lambda, psi, nu))
    #     end

    #     return updated_mvstudent_degs_mus_sigs
    # end

    # function updated_mvstudent_params(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; add_empty=true)
    #     return updated_mvstudent_params(clusters, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu, add_empty=add_empty)
    # end

    # function predictive_distribution(
    #     clusters::Vector{Cluster},
    #     hyperparams::MNCRPHyperparams;
    #     ignore_weights=false
    #     )

    #     alpha, mu, lambda, psi, nu = collect(hyperparams)

    #     if ignore_weights
    #         weights = ones(length(clusters) + 1)
    #         weights ./= sum(weights)
    #     else
    #         weights = 1.0 .* length.(clusters)
    #         push!(weights, alpha)
    #         weights ./= sum(weights)
    #     end

    #     updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)

    #     return predictive_distribution(weights, updated_mvstudent_degs_mus_sigs)
    # end

    # function predictive_distribution(
    #     component_weights::AbstractVector{Float64},
    #     mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}}
    #     )

    #     @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

    #     d = length(first(mvstudent_degs_mus_sigs)[2])

    #     return MixtureModel(
    #             [MvTDist(deg, d, mu, PDMat((sig + sig')/2)) for (deg, mu, sig) in mvstudent_degs_mus_sigs],
    #             Categorical(component_weights)
    #         )

    # end

    # function tail_probability(
    #     component_weights::AbstractVector{Float64},
    #     mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}};
    #     nb_rejection_samples=10000
    #     )

    #     @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

    #     dist = predictive_distribution(component_weights, mvstudent_degs_mus_sigs)

    #     sample_draw = rand(dist, nb_rejection_samples)
    #     logpdf_samples = logpdf(dist, sample_draw)

    #     function tailprob_func(coordinates::AbstractArray)
    #         isocontours = logpdf(dist, coordinates)
    #         if size(isocontours) === ()
    #             # if a single coordinate is passed return a single tail probability
    #             in_tail = logpdf_samples .<= isocontours
    #             return mean(in_tail)
    #         else
    #             # else if an array of coordinates is passed
    #             # we assume the first axis is the dimension of coordinates
    #             # and we return an array of tail probabilities of
    #             # the same shape as the trailing dimensions
    #             in_tail = logpdf_samples .<= reshape(isocontours, 1, size(isocontours)...)
    #             return dropdims(mean(in_tail, dims=1), dims=1)
    #         end
    #     end

    #     return tailprob_func

    # end

    # function tail_probability(
    #     clusters::Vector{Cluster},
    #     hyperparams::MNCRPHyperparams;
    #     nb_rejection_samples=10000,
    #     ignore_weights=false)

    #     alpha, mu, lambda, psi, nu = collect(hyperparams)

    #     if ignore_weights
    #         weights = ones(length(clusters) + 1)
    #     else
    #         weights = 1.0 * length.(clusters)
    #         push!(weights, alpha)
    #     end

    #     weights ./= sum(weights)

    #     updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)

    #     if hyperparams.nn !== nothing
    #         basetailprob_func = tail_probability(weights, updated_mvstudent_degs_mus_sigs, nb_rejection_samples=nb_rejection_samples)
    #         ffjord_mdl = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), basedist=nothing, ad=AutoForwardDiff())
    #         function tailprob_func(coordinates::AbstractArray)
    #             ret, _ = ffjord_mdl(coordinates isa AbstractVector ? reshape(coordinates, :, 1) : reshape(coordinates, first(size(coordinates)), :), hyperparams.nn_params, hyperparams.nn_state)
    #             base_elements = reshape(ret.z, size(coordinates)...)
    #             return basetailprob_func(base_elements)
    #         end
    #         return tailprob_func
    #     else
    #         return tail_probability(weights, updated_mvstudent_degs_mus_sigs, nb_rejection_samples=nb_rejection_samples)
    #     end

    # end

    # function tail_probability(clusters_samples::AbstractVector{Vector{Cluster}}, hyperparams_samples::AbstractVector{MNCRPHyperparams}; nb_rejection_samples=10000)

    #     tailprob_funcs = tail_probability.(clusters_samples, hyperparams_samples, nb_rejection_samples=nb_rejection_samples)

    #     function tailprob(coordinates::AbstractArray)
    #         tailprob_samples = [tailprob_func(coordinates) for tailprob_func in tailprob_funcs]
    #         if coordinates isa AbstractVector
    #             return tailprob_samples
    #         else
    #             return stack(tailprob_samples, dims=ndims(coordinates))
    #         end
    #     end

    #     return tailprob

    # end

    # function tail_probability_summary(clusters_samples::AbstractVector{Vector{Cluster}}, hyperparams_samples::AbstractVector{MNCRPHyperparams}; nb_rejection_samples=10000)
    #     tailprob_func = tail_probability(clusters_samples, hyperparams_samples, nb_rejection_samples=nb_rejection_samples)

    #     function _CI95(x::AbstractArray)
    #         return quantile(x, 0.975) - quantile(x, 0.025)
    #     end

    #     function _CI90(x::AbstractArray)
    #        return quantile(x, 0.95) - quantile(x, 0.05)
    #     end

    #     # Mode from Freedman-Diaconis rule
    #     function _modefd(x::AbstractArray)
    #         fdw = freedmandiaconis(x)
    #         fdbins = nothing
    #         fdbins = LinRange(minimum(x), maximum(x), ceil(Int64, (maximum(x) - minimum(x))/fdw) + 1)
    #         histfd = fit(Histogram, x, fdbins)
    #         modefdidx = sortperm(histfd.weights)[end]
    #         return mean(histfd.edges[1][modefdidx:modefdidx+1])

    #     end

    #     # Mode from Doane's rule
    #     function _modedoane(x::AbstractArray)
    #         dk = ceil(Int64, doane(x))
    #         dbins = LinRange(minimum(x), maximum(x), dk + 1)
    #         histd = fit(Histogram, x, dbins)
    #         modedidx = sortperm(histd.weights)[end]
    #         return mean(histd.edges[1][modedidx:modedidx+1])
    #     end

    #     function summaries(coordinates::AbstractArray)
    #         tps = tailprob_func(coordinates)

    #         if coordinates isa AbstractVector

    #             return (median=median(tps),
    #                     mean=mean(tps),
    #                     std=std(tps),
    #                     iqr=iqr(tps),
    #                     CI95=_CI95(tps),
    #                     CI90=_CI90(tps),
    #                     quantile=q -> quantile(tps, q),
    #                     modefd=_modefd(tps),
    #                     modedoane=_modedoane(tps)
    #                     )
    #         else
    #             return (median=dropdims(mapslices(median, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     mean=dropdims(mapslices(mean, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     std=dropdims(mapslices(stT, Dps, dims=ndims(tps)), dims=ndims(tps)),
    #                     iqr=dropdims(mapslices(iqr, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     CI95=dropdims(mapslices(_CI95, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     CI90=dropdims(mapslices(_CI90, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     quantile=q -> dropdims(mapslices(sl -> quantile(sl, q), tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     modefd=dropdims(mapslices(_modefT, Dps, dims=ndims(tps)), dims=ndims(tps)),
    #                     modedoane=dropdims(mapslices(_modedoane, tps, dims=ndims(tps)), dims=ndims(tps))
    #                     )
    #         end
    #     end

    # end

end
