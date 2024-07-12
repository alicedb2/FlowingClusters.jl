module FlowingClusters

    using Random: randperm, shuffle, shuffle!, seed!, Xoshiro, AbstractRNG
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
                 hidespines!, hidedecorations!, Cycled, scatter!

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

    export advance_chain!
    export advance_hyperparams_adaptive!
    export advance_ffjord!
    export attempt_map!, burn!

    export logprobgenerative
    export optimize_hyperparams, optimize_hyperparams!

    # Plotting related functions
    export project_clusters, project_cluster, project_hyperparams, project_mu, project_psi
    export plot, plot!

    # Predictions
    export predictive_distribution, tail_probability, tail_probability_summary

    include("types/diagnostics.jl")
    export Diagnostics, DiagnosticsFFJORD
    export clear_diagnostics!, am_sigma

    include("types/hyperparams.jl")
    export FCHyperparams, FCHyperparamsFFJORD
    export dimension, modeldimension, ij, flatk, foldL, foldpsi, flatten
    export forwardffjord, backwardffjord

    include("types/cluster.jl")
    export AbstractCluster, BitCluster, SetCluster
    export project_cluster, project_clusters, elements
    export isvalidpartition, iscompletepartition
    export pop!, push!, find

    include("types/chain.jl")
    export FCChain
    
    include("conjugateupdates.jl")
    export log_Zniw, updated_niw_hyperparams, updated_mvstudent_params

    # include("hyperpriors.jl")
    # include("modelprobabilities.jl")
    # export logprobgenerative

    include("chainsteps.jl")


    export logprob_chain, nbclusters_chain, largestcluster_chain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export nn_chain, nn_alpha_chain, nn_scale_chain
    export ess_rhat, stats

    include("plotting.jl")

    include("naivebioclim.jl")
    using .NaiveBIOCLIM
    export bioclim_predictor

    include("helpers.jl")
    export generate_data
    export performance_statistics, best_score_threshold
    export drawNIW
    export sqrtsigmoid, sqrttanh, sqrttanhgrow
    export chunk
    export logdetpsd, logdetflatLL
    export freedmandiaconis, doane
    export project_vec, project_mat


    # function advance_chain!(chain::MNCRPChain, nb_steps=100;
    #     nb_hyperparams=1, nb_gibbs=1,
    #     nb_splitmerge=30, splitmerge_t=3,
    #     amwg_batch_size=50,
    #     nb_ffjord_am=2,
    #     sample_every=:autocov, stop_chain=nothing,
    #     checkpoint_every=-1, checkpoint_prefix="chain",
    #     attempt_map=true, pretty_progress=:repl)

    #     checkpoint_every == -1 || typeof(checkpoint_prefix) == String || throw("Must specify a checkpoint prefix string")

    #     # Used for printing stats #
    #     hp = chain.hyperparams

    #     last_accepted_split = hp.diagnostics.accepted_split
    #     last_rejected_split = hp.diagnostics.rejected_split
    #     split_total = 0

    #     last_accepted_merge = hp.diagnostics.accepted_merge
    #     last_rejected_merge = hp.diagnostics.rejected_merge
    #     merge_total = 0

    #     # nb_fullseq_moves = 0

    #     nb_map_attemps = 0
    #     nb_map_successes = 0

    #     last_checkpoint = -1
    #     last_map_idx = chain.map_idx

    #     ###########################

    #     delta_minusnn = 0.0
    #     if hp.nn !== nothing
    #         delta_minusnn = nn_prior(hp.nn, similar(hp.nn_params) .= 0.0, hp.nn_alpha)
    #     end

    #     if pretty_progress === :repl
    #         progressio = stderr
    #     elseif pretty_progress === :file
    #         progressio = open("progress_pid$(getpid()).txt", "w")
    #     end

    #     if pretty_progress === :repl || pretty_progress === :file || pretty_progress
    #         if nb_steps === nothing || !isfinite(nb_steps) || nb_steps < 0
    #             progbar = ProgressUnknown(showspeed=true, output=progressio)
    #             nb_steps = nothing
    #             _nb_steps = typemax(Int64)
    #         else
    #             progbar = Progress(nb_steps; showspeed=true, output=progressio)
    #             _nb_steps = nb_steps
    #         end
    #     end

    #     for step in 1:_nb_steps

    #         for i in 1:nb_hyperparams

    #             advance_hyperparams_adaptive!(
    #                 chain.clusters,
    #                 chain.hyperparams,
    #                 chain.base2original,
    #                 nb_ffjord_am=nb_ffjord_am, hyperparams_chain=chain.hyperparams_chain,
    #                 amwg_batch_size=amwg_batch_size, acceptance_target=0.44
    #                 )
    #         end

    #         push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

    #         # Sequential split-merge
    #         if nb_splitmerge isa Int64
    #             for i in 1:nb_splitmerge
    #                 advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t)
    #             end
    #         elseif nb_splitmerge isa Float64 || nb_splitmerge === :matchgibbs
    #             # old_nb_ssuccess = chain.hyperparams.diagnostics.accepted_split
    #             # old_nb_msuccess = chain.hyperparams.diagnostics.accepted_merge
    #             # while (chain.hyperparams.diagnostics.accepted_split == old_nb_ssuccess) && (chain.hyperparams.diagnostics.accepted_merge == old_nb_msuccess)
    #             #     advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
    #             # end
    #         end

    #         # Gibbs sweep
    #         if nb_gibbs isa Int64
    #             for i in 1:nb_gibbs
    #                 advance_gibbs!(chain.clusters, chain.hyperparams)
    #             end
    #         elseif (0.0 < nb_gibbs < 1.0) && (rand() < nb_gibbs)
    #             advance_gibbs!(chain.clusters, chain.hyperparams)
    #         end

    #         push!(chain.nbclusters_chain, length(chain.clusters))
    #         push!(chain.largestcluster_chain, maximum(length.(chain.clusters)))

    #         # Stats #
    #         split_ratio = "$(hp.diagnostics.accepted_split - last_accepted_split)/$(hp.diagnostics.rejected_split - last_rejected_split)"
    #         merge_ratio = "$(hp.diagnostics.accepted_merge - last_accepted_merge)/$(hp.diagnostics.rejected_merge - last_rejected_merge)"
    #         split_total += hp.diagnostics.accepted_split - last_accepted_split
    #         merge_total += hp.diagnostics.accepted_merge - last_accepted_merge
    #         split_per_step = round(split_total/step, digits=2)
    #         merge_per_step = round(merge_total/step, digits=2)
    #         last_accepted_split = hp.diagnostics.accepted_split
    #         last_rejected_split = hp.diagnostics.rejected_split
    #         last_accepted_merge = hp.diagnostics.accepted_merge
    #         last_rejected_merge = hp.diagnostics.rejected_merge


    #         ########################

    #         # logprob
    #         logprob = logprobgenerative(chain.clusters, chain.hyperparams, chain.base2original, ffjord=chain.hyperparams.nn !== nothing)
    #         push!(chain.logprob_chain, logprob)

    #         # MAP
    #         history_length = 500
    #         short_logprob_chain = chain.logprob_chain[max(1, end - history_length):end]

    #         logp_quantile95 = quantile(short_logprob_chain, 0.95)

    #         map_success = false # try block has its own scope

    #         if logprob > logp_quantile95 && attempt_map
    #                 nb_map_attemps += 1
    #                 try
    #                     map_success = attempt_map!(chain, max_nb_pushes=15, verbose=false, optimize_hyperparams=false)
    #                 catch e
    #                     throw(e)
    #                     map_success = false
    #                 end

    #                 nb_map_successes += map_success
    #                 last_map_idx = chain.map_idx
    #         end

    #         sample_eta = -1

    #         start_sampling_at = 5 * param_dimension(chain.hyperparams)

    #         if sample_every !== nothing
    #             if sample_every === :autocov
    #                 if length(chain) >= start_sampling_at
    #                     convergence_burnt = ess_rhat(largestcluster_chain(chain)[div(end, 2):end])
    #                     latest_sample_idx = length(chain.samples_idx) > 0 ? chain.samples_idx[end] : 0
    #                     curr_idx = length(chain)
    #                     sample_eta = floor(Int64, latest_sample_idx + 2 * div(curr_idx, 2) / convergence_burnt.ess + 1 - curr_idx)
    #                     if curr_idx - latest_sample_idx > 2 * div(curr_idx, 2) / convergence_burnt.ess
    #                         push!(chain.clusters_samples, deepcopy(chain.clusters))
    #                         push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
    #                         push!(chain.base2original_samples, deepcopy(chain.base2original))
    #                         push!(chain.samples_idx, curr_idx)
    #                     end

    #                     if length(chain.samples_idx) >= 20
    #                         sample_ess = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples]).ess
    #                         while 2 * sample_ess < length(chain.samples_idx)
    #                             popfirst!(chain.clusters_samples)
    #                             popfirst!(chain.hyperparams_samples)
    #                             popfirst!(chain.base2original_samples)
    #                             popfirst!(chain.samples_idx)
    #                             sample_ess = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples]).ess
    #                         end
    #                     end
    #                 end
    #             elseif sample_every >= 1
    #                 if mod(length(chain.logprob_chain), sample_every) == 0
    #                     push!(chain.clusters_samples, deepcopy(chain.clusters))
    #                     push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
    #                     push!(chain.base2original_samples, deepcopy(chain.base2original))
    #                     chain.oldest_sample = length(chain.logprob_chain)
    #                 end
    #             end
    #         end

    #         if checkpoint_every > 0 &&
    #             (mod(length(chain.logprob_chain), checkpoint_every) == 0
    #             || last_checkpoint == -1)

    #             last_checkpoint = length(chain.logprob_chain)
    #             mkpath(dirname("$(checkpoint_prefix)"))
    #             filename = "$(checkpoint_prefix)_pid$(getpid())_iter$(last_checkpoint).jld2"
    #             jldsave(filename; chain)
    #         end


    #         if length(chain) >= start_sampling_at
    #             largestcluster_convergence = ess_rhat(largestcluster_chain(chain)[div(end, 2):end])
    #         else
    #             largestcluster_convergence = (ess=0, rhat=0)
    #         end

    #         if length(chain.samples_idx) >= 20
    #             samples_convergence = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples])
    #         else
    #             samples_convergence = (ess=0, rhat=0)
    #         end

    #         if chain.hyperparams.nn !== nothing
    #             _delta_minusnn_map = delta_minusnn + log_jeffreys_t(chain.map_hyperparams.nn_alpha, chain.map_hyperparams.nn_scale)
    #         end

    #         if pretty_progress === :repl || pretty_progress === :file || pretty_progress
    #             next!(progbar;
    #             showvalues=[
    #             (:"step (hyperparams per, gibbs per, splitmerge per)", "$(step)/$(nb_steps === nothing ? Inf : nb_steps) ($nb_hyperparams, $nb_gibbs, $(round(nb_splitmerge, digits=2)))"),
    #             (:"chain length", "$(length(chain))"),
    #             (:"conv largestcluster chain (burn 50%)", "ess=$(round(largestcluster_convergence.ess, digits=1)), rhat=$(round(largestcluster_convergence.rhat, digits=3))$(length(chain) < start_sampling_at ? " (wait $start_sampling_at)" : "")"),
    #             (:"#chain samples (oldest, latest, eta) convergence", "$(pretty_progress === :repl ? "\033[37m" : "")$(length(chain.samples_idx))/$(length(chain.samples_idx.buffer)) ($(length(chain.samples_idx) > 0 ? chain.samples_idx[begin] : -1), $(length(chain.samples_idx) > 0 ? chain.samples_idx[end] : -1), $(max(0, sample_eta))) ess=$(samples_convergence.ess > 0 ? round(samples_convergence.ess, digits=1) : "wait 20") rhat=$(samples_convergence.rhat > 0 ? round(samples_convergence.rhat, digits=3) : "wait 20") (trim if ess<$(samples_convergence.ess > 0 ? round(length(chain.samples_idx)/2, digits=1) : "wait"))$(pretty_progress === :repl ? "\033[0m" : "")"),
    #             (:"logprob (max, q95)", "$(round(chain.logprob_chain[end], digits=1)) ($(round(maximum(chain.logprob_chain), digits=1)), $(round(logp_quantile95, digits=1)))"),
    #             (:"nb clusters, nb>1, smallest(>1), median, mean, largest", "$(length(chain.clusters)), $(length(filter(c -> length(c) > 1, chain.clusters))), $(minimum(length.(filter(c -> length(c) > 1, chain.clusters)))), $(round(median([length(c) for c in chain.clusters]), digits=0)), $(round(mean([length(c) for c in chain.clusters]), digits=0)), $(maximum([length(c) for c in chain.clusters]))"),
    #             (:"split #succ/#tot, merge #succ/#tot", split_ratio * ", " * merge_ratio),
    #             (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
    #             (:"MAP #attempts/#successes", "$(nb_map_attemps)/$(nb_map_successes)" * (attempt_map ? "" : " (off)")),
    #             (:"nb clusters, nb>1, smallest(>1), median, mean, largest", "$(length(chain.map_clusters)), $(length(filter(c -> length(c) > 1, chain.map_clusters))), $(minimum(length.(filter(c -> length(c) > 1, chain.map_clusters)))), $(round(median([length(c) for c in chain.map_clusters]), digits=0)), $(round(mean([length(c) for c in chain.map_clusters]), digits=0)), $(maximum([length(c) for c in chain.map_clusters]))"),
    #             (:"last MAP logprob (minus nn)", "$(round(chain.map_logprob, digits=1)) ($(round(chain.map_logprob - _delta_minusnn_map, digits=1)))"),
    #             (:"last MAP at", last_map_idx),
    #             (:"last checkpoint at", last_checkpoint)
    #             ])
    #         else
    #             print("\r$(step)/$(nb_steps === nothing ? Inf : nb_steps)")
    #             print(" (t:$(length(chain.logprob_chain)))")
    #             print("   lp: $(round(chain.logprob_chain[end], digits=1))")
    #             print("   sr:" * split_ratio * " mr:" * merge_ratio)
    #             print("   sps:$(split_per_step), mps:$(merge_per_step)")
    #             print("   #cl:$(length(chain.clusters)), #cl>1:$(length(filter(x -> length(x) > 1, chain.clusters)))")
    #             print("   mapattsuc:$(nb_map_attemps)/$(nb_map_successes)")
    #             print("   lastmap@$(last_map_idx)")
    #             print("   lchpt@$(last_checkpoint)")
    #             print("      ")
    #             if map_success
    #                 print("!      #clmap:$(length(chain.map_clusters))")
    #                 println("   $(round(chain.map_logprob, digits=1))")
    #             end
    #             flush(stdout)
    #         end

    #         if stop_chain === :sample_ess
    #             if samples_convergence.ess >= length(chain.samples_idx.buffer)
    #                 break
    #             end
    #         end

    #         isfile("stop") && break

    #     end

    #     if pretty_progress === :file
    #         close(progressio)
    #     end

    # end




    # function attempt_map!(chain::MNCRPChain; max_nb_pushes=15, optimize_hyperparams=true, verbose=true, force=false)

    #     map_clusters_attempt = deepcopy(chain.clusters)
    #     map_hyperparams = deepcopy(chain.hyperparams)

    #     map_mll = logprobgenerative(map_clusters_attempt, chain.map_hyperparams, ffjord=false)

    #     # Greedy Gibbs!
    #     for p in 1:max_nb_pushes
    #         test_attempt = copy(map_clusters_attempt)
    #         advance_gibbs!(test_attempt, map_hyperparams; temperature=0.0)
    #         test_mll = logprobgenerative(test_attempt, map_hyperparams, ffjord=false)
    #         if test_mll <= map_mll
    #             # We've regressed, so stop and leave
    #             # the previous state before this test
    #             # attempt as the approx. map state
    #             break
    #         else
    #             # the test state led to a better state
    #             # than the previous one, keep going
    #             map_clusters_attempt = test_attempt
    #             map_mll = test_mll
    #         end
    #     end

    #     if optimize_hyperparams
    #         optimize_hyperparams!(map_clusters_attempt, map_hyperparams, verbose=verbose)
    #     end

    #     attempt_logprob = logprobgenerative(map_clusters_attempt, map_hyperparams, chain.base2original, hyperpriors=true, ffjord=true)
    #     if attempt_logprob > chain.map_logprob || force
    #         chain.map_logprob = attempt_logprob
    #         chain.map_clusters = map_clusters_attempt
    #         chain.map_hyperparams = map_hyperparams
    #         chain.map_base2original = deepcopy(chain.base2original)
    #         chain.map_idx = lastindex(chain.logprob_chain)
    #         return true
    #     else
    #         return false
    #     end
    # end

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
