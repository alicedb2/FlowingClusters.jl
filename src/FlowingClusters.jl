module FlowingClusters

    using Random: randperm, shuffle, shuffle!, seed!, Xoshiro
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
    import DiffEqFlux: FFJORD
    const backward_ffjord = DiffEqFlux.__backward_ffjord

    using ComponentArrays: ComponentArray, valkeys

    import Base: pop!, push!, length, isempty, union, delete!, empty!
    import Base: iterate, deepcopy, copy, sort, in, first
    import MCMCDiagnosticTools: ess_rhat
    import Makie: plot, plot!
    import Base.Iterators: flatten

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
    export Diagnostics, clear_diagnostics!, am_sigma

    include("types/hyperparams.jl")
    export FCHyperparams, dimension, modeldimension, ij, flatk, foldL, foldpsi, flatten

    include("types/cluster.jl")
    export Cluster, elements
    export realspace_cluster, realspace_clusters
    
    include("types/chain.jl")
    export FCChain
    
    include("conjugateupdates.jl")
    export log_Zniw, updated_niw_hyperparams, updated_mvstudent_params

    include("hyperpriors.jl")
    include("modelprobabilities.jl")
    export logprobgenerative

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
    export performance_statistics, best_score_threshold
    export drawNIW
    export sqrtsigmoid, sqrttanh, sqrttanhgrow
    export chunk
    export logdetpsd, logdetflatLL
    export freedmandiaconis, doane


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

    # # Sequential splitmerge from Dahl & Newcomb
    # function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; t=3, temperature=1.0)

    #     @assert t >= 0

    #     alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
    #     d = length(mu)

    #     cluster_indices = Tuple{Int64, Vector{Float64}}[(ce, e) for (ce, cluster) in enumerate(clusters) for e in cluster]

    #     (ci, ei), (cj, ej) = sample(cluster_indices, 2, replace=false)

    #     if ci == cj

    #         scheduled_elements = [e for e in clusters[ci] if !(e === ei) && !(e === ej)]
    #         initial_state = Cluster[clusters[ci]]
    #         deleteat!(clusters, ci)

    #     elseif ci != cj

    #         scheduled_elements = [e for e in flatten((clusters[ci], clusters[cj])) if !(e === ei) && !(e === ej)]
    #         initial_state = Cluster[clusters[ci], clusters[cj]]
    #         deleteat!(clusters, sort([ci, cj]))

    #     end

    #     shuffle!(scheduled_elements)

    #     proposed_state = Cluster[Cluster([ei]), Cluster([ej])]
    #     launch_state = Cluster[]

    #     log_q = 0.0

    #     for step in flatten((0:t, [:create_proposed_state]))


    #         # Keep copy of launch state
    #         #
    #         if step == :create_proposed_state
    #             if ci == cj
    #                 # Do a last past to the proposed
    #                 # split state to accumulate
    #                 # q(proposed|launch)
    #                 # remember that
    #                 # q(launch|proposed)
    #                 # = q(merged|some split launch state) = 1
    #                 launch_state = copy(proposed_state)
    #             elseif ci != cj
    #                 # Don't perform last step in a merge,
    #                 # keep log_q as the transition probability
    #                 # to the launch state, i.e.
    #                 # q(launch|proposed) = q(launch|launch-1)
    #                 launch_state = proposed_state
    #                 break
    #             end
    #         end

    #         log_q = 0.0

    #         for e in shuffle!(scheduled_elements)

    #             delete!(proposed_state, e)

    #             # Should be true by construction, just
    #             # making sure the construction is valid
    #             @assert all([!isempty(c) for c in proposed_state])
    #             @assert length(proposed_state) == 2

    #             #############

    #             log_weights = zeros(length(proposed_state))
    #             for (i, cluster) in enumerate(proposed_state)
    #                 log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
    #             end

    #             if temperature > 0.0
    #                 unnorm_logp = log_weights / temperature
    #                 norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
    #                 probs = Weights(exp.(norm_logp))
    #                 new_assignment, log_transition = sample(collect(zip(proposed_state, norm_logp)), probs)
    #                 log_q += log_transition
    #             elseif temperature <= 0.0
    #                 _, max_idx = findmax(log_weights)
    #                 new_assignment = proposed_state[max_idx]
    #                 log_q += 0.0 # symbolic, transition is certain
    #             end

    #             push!(new_assignment, e)

    #         end

    #     end

    #     # At this point if we are doing a split state
    #     # then log_q = q(split*|launch)
    #     # and if we are doing a merge state
    #     # then log_q = q(launch|merge)=  q(launch|launch-1)

    #     if ci != cj
    #         # Create proposed merge state
    #         # The previous loop was only to get
    #         # q(launch|proposed) = q(launch|launch-1)
    #         # and at this point launch_state = proposed_state
    #         proposed_state = Cluster[Cluster(Vector{Float64}[e for cluster in initial_state for e in cluster])]
    #     elseif ci == cj
    #         # do nothing, we already have the proposed state
    #     end

    #     log_acceptance = (logprobgenerative(proposed_state, hyperparams, hyperpriors=false)
    #                     - logprobgenerative(initial_state, hyperparams, hyperpriors=false))

    #     log_acceptance /= temperature

    #     # log_q is plus-minus the log-Hastings factor.
    #     # log_q already includes the tempering.
    #     if ci != cj
    #         log_acceptance += log_q
    #     elseif ci == cj
    #         log_acceptance -= log_q
    #     end

    #     log_acceptance = min(0.0, log_acceptance)

    #     if log(rand()) < log_acceptance
    #         append!(clusters, proposed_state)
    #         if ci != cj
    #             hyperparams.diagnostics.accepted_merge += 1
    #         elseif ci == cj
    #             hyperparams.diagnostics.accepted_split += 1
    #         end
    #     else
    #         append!(clusters, initial_state)
    #         if ci != cj
    #             hyperparams.diagnostics.rejected_merge += 1
    #         elseif ci == cj
    #             hyperparams.diagnostics.rejected_split += 1
    #         end
    #     end

    #     return clusters

    # end

    # function advance_hyperparams_adaptive!(
    #     clusters::Vector{Cluster},
    #     hyperparams::MNCRPHyperparams,
    #     base2original::Dict{Vector{Float64}, Vector{Float64}},
    #     diagnostics::Diagnostics;
    #     amwg_batch_size=40, acceptance_target=0.44,
    #     nb_ffjord_am=1, am_safety_probability=0.05, am_safety_sigma=0.1,
    #     hyperparams_chain=nothing, temperature=1.0)

    #     di = diagnostics
    #     d = dimension(hyperparams)

    #     slice_mu = 2:1+d
    #     slice_psi = (2 + d + 1):(2 + d + div(d * (d + 1), 2))

    #     idx_nu = 3 + d + div(d * (d + 1), 2)

    #     idx_nn_alpha = idx_nu + 1
    #     idx_nn_scale = idx_nn_alpha + 1

    #     clear_diagnostics!(di, clearhyperparams=true, clearsplitmerge=false, clearnn=false, keepstepscale=true)

    #     nn_D = hyperparams.nn_params === nothing ? 0 : size(hyperparams.nn_params, 1)

    #     for i in 1:amwg_batch_size
    #             advance_alpha!(clusters, hyperparams, step_size=exp(di.amwg_logscales[1]))
    #             advance_mu!(clusters, hyperparams, step_size=exp.(di.amwg_logscales[slice_mu]))
    #             advance_lambda!(clusters, hyperparams, step_size=exp(di.amwg_logscales[2+d]))
    #             advance_psi!(clusters, hyperparams,step_size=exp.(di.amwg_logscales[slice_psi]))
    #             advance_nu!(clusters, hyperparams, step_size=exp(di.amwg_logscales[idx_nu]))
    #             if hyperparams.nn !== nothing
    #                 advance_nn_alpha!(hyperparams, step_size=exp(di.amwg_logscales[idx_nn_alpha]))
    #                 advance_nn_scale!(hyperparams, step_size=exp(di.amwg_logscales[idx_nn_scale]))
    #             end
    #     end

    #     di.amwg_nbbatches += 1
    #     adjust_amwg_logscales!(di, acceptance_target=acceptance_target)

    #     if hyperparams.nn !== nothing && nb_ffjord_am > 0

    #         if length(hyperparams_chain) <= 4 * nn_D

    #             step_distrib = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))

    #         elseif length(hyperparams_chain) > 4 * nn_D

    #             nn_sigma = am_sigma(hyperparams.diagnostics)

    #             safety_component = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))
    #             empirical_estimate_component = MvNormal(2.38^2 / nn_D * nn_sigma)

    #             step_distrib = MixtureModel([safety_component, empirical_estimate_component], [am_safety_probability, 1 - am_safety_probability])
    #         end

    #         for i in 1:nb_ffjord_am
    #             advance_ffjord!(clusters, hyperparams, base2original,
    #                             step_distrib=step_distrib, temperature=temperature)
    #         end

    #         hyperparams.diagnostics.am_L += 1
    #         hyperparams.diagnostics.am_N .+= hyperparams.nn_params
    #         hyperparams.diagnostics.am_NN .+= hyperparams.nn_params * hyperparams.nn_params'

    #     end

    #     return hyperparams

    # end

    # function adjust_amwg_logscales!(diagnostics::Diagnostics; acceptance_target=0.44, minmax_logscale=Inf, min_delta=0.01)

    #     di = diagnostics

    #     delta_n = min(min_delta, 1/sqrt(di.amwg_nbbatches))

    #     d = size(di.accepted_mu, 1)
    #     idx_nu = 3 + d + div(d * (d + 1), 2)
    #     idx_nn_alpha = idx_nu + 1
    #     idx_nn_scale = idx_nn_alpha + 1
        
    #     if di.accepted_alpha / (di.accepted_alpha + di.rejected_alpha) < acceptance_target
    #         di.amwg_logscales[1] -= delta_n
    #     else
    #         di.amwg_logscales[1] += delta_n
    #     end

    #     for i in 1:d
    #         if di.accepted_mu[i] / (di.accepted_mu[i] + di.rejected_mu[i]) < acceptance_target
    #             di.amwg_logscales[1+i] -= delta_n
    #         else
    #             di.amwg_logscales[1+i] += delta_n
    #         end
    #     end

    #     if di.accepted_lambda / (di.accepted_lambda + di.rejected_lambda) < acceptance_target
    #         di.amwg_logscales[2+d] -= delta_n
    #     else
    #         di.amwg_logscales[2+d] += delta_n
    #     end

    #     for i in 1:size(di.accepted_flatL, 1)
    #         if di.accepted_flatL[i] / (di.accepted_flatL[i] + di.rejected_flatL[i]) < acceptance_target
    #             di.amwg_logscales[2+d+i] -= delta_n
    #         else
    #             di.amwg_logscales[2+d+i] += delta_n
    #         end
    #     end

    #     if di.accepted_nu / (di.accepted_nu + di.rejected_nu) < acceptance_target
    #         di.amwg_logscales[idx_nu] -= delta_n
    #     else
    #         di.amwg_logscales[idx_nu] += delta_n
    #     end

    #     # Bit of a hack to determine if
    #     # a neural network is present but ok
    #     # It relies on advance_hyperparams_adaptive!
    #     # skipping nn_alpha if the neural network is not present
    #     if di.accepted_nn_alpha + di.rejected_nn_alpha > 0
    #         if di.accepted_nn_alpha / (di.accepted_nn_alpha + di.rejected_nn_alpha) < acceptance_target
    #             di.amwg_logscales[idx_nn_alpha] -= delta_n
    #         else
    #             di.amwg_logscales[idx_nn_alpha] += delta_n
    #         end
    #     end

    #     if di.accepted_nn_scale + di.rejected_nn_scale > 0
    #         if di.accepted_nn_scale / (di.accepted_nn_scale + di.rejected_nn_scale) < acceptance_target
    #             di.amwg_logscales[idx_nn_scale] -= delta_n
    #         else
    #             di.amwg_logscales[idx_nn_scale] += delta_n
    #         end
    #     end

    #     di.amwg_logscales[di.amwg_logscales .< -minmax_logscale] .= -minmax_logscale
    #     di.amwg_logscales[di.amwg_logscales .> minmax_logscale] .= minmax_logscale

    #     return di

    # end

    # function advance_ffjord!(
    #     clusters::Vector{Cluster},
    #     hyperparams::MNCRPHyperparams,
    #     base2original::Dict{Vector{Float64}, Vector{Float64}};
    #     step_distrib=nothing,
    #     temperature=1.0)

    #     if hyperparams.nn === nothing
    #         return hyperparams
    #     end

    #     ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), basedist=nothing, ad=AutoForwardDiff())
    #     original_clusters = realspace_clusters(Matrix, clusters, base2original)

    #     proposed_nn_params = hyperparams.nn_params .+ rand(step_distrib)

    #     # We could have left the calculation of deltalogps
    #     # to logprobgenerative below, but we a proposal comes
    #     # a new base2original so we do both at once here and
    #     # call logprobgenerative with ffjord=false

    #     original_elements = reduce(hcat, original_clusters)
    #     proposed_base, _ = ffjord_model(original_elements, proposed_nn_params, hyperparams.nn_state)
    #     proposed_elements = Matrix{Float64}(proposed_base.z)
    #     proposed_baseclusters = chunk(proposed_elements, size.(original_clusters, 2))
    #     proposed_base2original = Dict{Vector{Float64}, Vector{Float64}}(eachcol(proposed_elements) .=> eachcol(original_elements))

    #     log_acceptance = -sum(proposed_base.delta_logp)

    #     # We already accounted for the ffjord deltalogps above
    #     # so call logprobgenerative with ffjord=false on the
    #     # proposed state.
    #     log_acceptance += (
    #           logprobgenerative(Cluster.(proposed_baseclusters), hyperparams, proposed_base2original, hyperpriors=false, ffjord=false)
    #         - logprobgenerative(clusters, hyperparams, base2original, hyperpriors=false, ffjord=true)
    #     )

    #     # We called logprobgenerative with ffjord=true on the current state
    #     # but not on the proposed state, so we need to account for the
    #     # prior on the neural network for the proposed state
    #     log_acceptance += nn_prior(hyperparams.nn, proposed_nn_params, hyperparams.nn_alpha, hyperparams.nn_scale)

    #     log_acceptance /= temperature

    #     log_acceptance = min(0.0, log_acceptance)
    #     if log(rand()) < log_acceptance
    #         hyperparams.nn_params = proposed_nn_params
    #         empty!(clusters)
    #         append!(clusters, Cluster.(proposed_baseclusters))
    #         empty!(base2original)
    #         merge!(base2original, proposed_base2original)
    #         hyperparams.diagnostics.accepted_nn .+= 1
    #         return 1
    #     else
    #         hyperparams.diagnostics.rejected_nn .+= 1
    #         return 0
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
    #                     std=dropdims(mapslices(std, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     iqr=dropdims(mapslices(iqr, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     CI95=dropdims(mapslices(_CI95, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     CI90=dropdims(mapslices(_CI90, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     quantile=q -> dropdims(mapslices(sl -> quantile(sl, q), tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     modefd=dropdims(mapslices(_modefd, tps, dims=ndims(tps)), dims=ndims(tps)),
    #                     modedoane=dropdims(mapslices(_modedoane, tps, dims=ndims(tps)), dims=ndims(tps))
    #                     )
    #         end
    #     end

    # end

end
