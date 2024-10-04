function advance_chain!(chain::FCChain, nb_steps=100;
    nb_gibbs=1, nb_splitmerge=30, splitmerge_t=3,
    nb_amwg=1, amwg_batch_size=50, nb_ffjord_am=1, ffjord_am_temperature=1.0,
    sample_every=:autocov, stop_criterion=nothing,
    checkpoint_every=-1, checkpoint_prefix="chain",
    attempt_map=true, pretty_progress=:repl)

    checkpoint_every == -1 || typeof(checkpoint_prefix) == String || throw("Must specify a checkpoint prefix string")

    # Used for printing stats #
    diagnostics = chain.diagnostics

    last_accepted_split = diagnostics.accepted.splitmerge.split
    last_rejected_split = diagnostics.rejected.splitmerge.split
    split_total = 0

    last_accepted_merge = diagnostics.accepted.splitmerge.merge
    last_rejected_merge = diagnostics.rejected.splitmerge.merge
    merge_total = 0

    # nb_fullseq_moves = 0

    nb_map_attemps = 0
    nb_map_successes = 0

    last_checkpoint = -1
    last_map_idx = chain.map_idx

    ###########################

    if pretty_progress === :repl
        progressio = stderr
    elseif pretty_progress === :file
        progressio = open("progress_pid$(getpid()).txt", "w")
    end

    if pretty_progress === :repl || pretty_progress === :file || pretty_progress
        if isnothing(nb_steps) || !isfinite(nb_steps) || nb_steps < 0
            progbar = ProgressUnknown(showspeed=true, output=progressio)
            nb_steps = Inf
            _nb_steps = typemax(Int64)
        else
            progbar = Progress(nb_steps; showspeed=true, output=progressio)
            _nb_steps = nb_steps
        end
    end

    for step in 1:_nb_steps

        for i in 1:nb_amwg
            advance_hyperparams_amwg!(
                chain.rng, 
                chain.clusters, 
                chain.hyperparams, 
                chain.diagnostics, 
                amwg_batch_size=amwg_batch_size)
        end

        for i in 1:nb_ffjord_am
            advance_adaptive!(
                chain.rng, 
                chain.clusters,
                chain.hyperparams,
                chain.diagnostics,
                nb_ffjord_am=nb_ffjord_am, temperature=ffjord_am_temperature,
                hyperparams_chain=chain.hyperparams_chain
                )
        end

        push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

        # Sequential split-merge
        if nb_splitmerge isa Int64
            for i in 1:nb_splitmerge
                advance_splitmerge_seq!(chain.rng, chain.clusters, chain.hyperparams, diagnostics, t=splitmerge_t)
            end
        elseif nb_splitmerge isa Float64 || nb_splitmerge === :matchgibbs
            # old_nb_ssuccess = chain.hyperparams.diagnostics.accepted_split
            # old_nb_msuccess = chain.hyperparams.diagnostics.accepted_merge
            # while (chain.hyperparams.diagnostics.accepted_split == old_nb_ssuccess) && (chain.hyperparams.diagnostics.accepted_merge == old_nb_msuccess)
            #     advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
            # end
        end

        # Gibbs sweep
        if nb_gibbs isa Int64
            for i in 1:nb_gibbs
                advance_gibbs!(chain.rng, chain.clusters, chain.hyperparams)
            end
        elseif (0 < nb_gibbs < 1) && (rand(chain.rng, T)r < nb_gibbs)
            advance_gibbs!(chain.rng, chain.clusters, chain.hyperparams)
        end

        push!(chain.nbclusters_chain, length(chain.clusters))
        push!(chain.largestcluster_chain, maximum(length.(chain.clusters)))

        # Stats #
        split_ratio = "$(diagnostics.accepted.splitmerge.split - last_accepted_split)/$(diagnostics.rejected.splitmerge.split- last_rejected_split)"
        merge_ratio = "$(diagnostics.accepted.splitmerge.merge - last_accepted_merge)/$(diagnostics.rejected.splitmerge.merge - last_rejected_merge)"
        split_total += diagnostics.accepted.splitmerge.split - last_accepted_split
        merge_total += diagnostics.accepted.splitmerge.merge - last_accepted_merge
        split_per_step = round(split_total/step, digits=2)
        merge_per_step = round(merge_total/step, digits=2)
        last_accepted_split = diagnostics.accepted.splitmerge.split
        last_rejected_split = diagnostics.rejected.splitmerge.split
        last_accepted_merge = diagnostics.accepted.splitmerge.merge
        last_rejected_merge = diagnostics.rejected.splitmerge.merge


        ########################

        # logprob
        logprob = logprobgenerative(chain.clusters, chain.hyperparams, chain.rng)
        push!(chain.logprob_chain, logprob)

        # MAP
        history_length = 500
        short_logprob_chain = chain.logprob_chain[max(1, end - history_length):end]

        logp_quantile95 = quantile(short_logprob_chain, 0.95)

        map_success = false # try block has its own scope

        if logprob > logp_quantile95 && attempt_map
            nb_map_attemps += 1
            try
                map_success = attempt_map!(chain, max_nb_pushes=20, verbose=false, optimize_hyperparams=false)
            catch e
                throw(e)
                map_success = false
            end

            nb_map_successes += map_success
            last_map_idx = chain.map_idx
        end

        sample_eta = -1

        start_sampling_at = 5 * modeldimension(chain.hyperparams)

        if sample_every !== nothing
            if sample_every === :autocov
                if length(chain) >= start_sampling_at
                    convergence_burnt = ess_rhat(largestcluster_chain(chain)[div(end, 2):end])
                    latest_sample_idx = length(chain.samples_idx) > 0 ? chain.samples_idx[end] : 0
                    curr_idx = length(chain)
                    sample_eta = floor(Int64, latest_sample_idx + 2 * div(curr_idx, 2) / convergence_burnt.ess + 1 - curr_idx)
                    if curr_idx - latest_sample_idx > 2 * div(curr_idx, 2) / convergence_burnt.ess
                        push!(chain.clusters_samples, deepcopy(chain.clusters))
                        push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
                        push!(chain.samples_idx, curr_idx)
                    end

                    if length(chain.samples_idx) >= 20
                        sample_ess = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples]).ess
                        while 2 * sample_ess < length(chain.samples_idx)
                            popfirst!(chain.clusters_samples)
                            popfirst!(chain.hyperparams_samples)
                            popfirst!(chain.samples_idx)
                            sample_ess = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples]).ess
                        end
                    end
                end
            elseif sample_every >= 1
                if mod(length(chain.logprob_chain), sample_every) == 0
                    push!(chain.clusters_samples, deepcopy(chain.clusters))
                    push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
                    chain.oldest_sample = length(chain.logprob_chain)
                end
            end
        end

        if checkpoint_every > 0 &&
            (mod(length(chain.logprob_chain), checkpoint_every) == 0
            || last_checkpoint == -1)

            last_checkpoint = length(chain.logprob_chain)
            mkpath(dirname("$(checkpoint_prefix)"))
            filename = "$(checkpoint_prefix)_pid$(getpid())_iter$(last_checkpoint).jld2"
            jldsave(filename; chain)
        end


        if length(chain) >= start_sampling_at
            largestcluster_convergence = ess_rhat(largestcluster_chain(chain)[div(end, 2):end])
        else
            largestcluster_convergence = (ess=0, rhat=0)
        end

        if length(chain.samples_idx) >= 20
            samples_convergence = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples])
        else
            samples_convergence = (ess=0, rhat=0)
        end

        delta_minusnn_map = 0.0
        if hasnn(chain.hyperparams)
            delta_minusnn_map += log_nn_prior(chain.map_hyperparams._.nn.params, chain.map_hyperparams._.nn.prior.alpha, chain.map_hyperparams._.nn.prior.scale)
            delta_minusnn_map += log_jeffreys_nn(chain.map_hyperparams._.nn.prior.alpha, chain.map_hyperparams._.nn.prior.scale)
        end

        if pretty_progress === :repl || pretty_progress === :file || pretty_progress
            next!(progbar;
            showvalues=[
            (:"step (#gibbs, #splitmerge, #amwg, #ffjordam)", "$(step)/$(nb_steps) ($nb_gibbs, $nb_splitmerge, $nb_amwg, $nb_ffjord_am)"),
            (:"chain length", "$(length(chain))"),
            (:"conv largestcluster chain (burn 50%)", "ess=$(round(largestcluster_convergence.ess, digits=1)), rhat=$(round(largestcluster_convergence.rhat, digits=3))$(length(chain) < start_sampling_at ? " (wait $start_sampling_at)" : "")"),
            (:"#chain samples (oldest, latest, eta) convergence", "$(pretty_progress === :repl ? "\033[37m" : "")$(length(chain.samples_idx))/$(length(chain.samples_idx.buffer)) ($(length(chain.samples_idx) > 0 ? chain.samples_idx[begin] : -1), $(length(chain.samples_idx) > 0 ? chain.samples_idx[end] : -1), $(max(0, sample_eta))) ess=$(samples_convergence.ess > 0 ? round(samples_convergence.ess, digits=1) : "wait 20") rhat=$(samples_convergence.rhat > 0 ? round(samples_convergence.rhat, digits=3) : "wait 20") (trimmed if ess < $(samples_convergence.ess > 0 ? round(length(chain.samples_idx)/2, digits=1) : "wait"))$(pretty_progress === :repl ? "\033[0m" : "")"),
            (:"logprob (best, q95)", "$(round(chain.logprob_chain[end], digits=1)) ($(round(maximum(chain.logprob_chain), digits=1)), $(round(logp_quantile95, digits=1)))"),
            (:"nb clusters, nb>1, smallest(>1), median, mean, largest", "$(length(chain.clusters)), $(length(filter(c -> length(c) > 1, chain.clusters))), $(minimum(length.(filter(c -> length(c) > 1, chain.clusters)))), $(round(median([length(c) for c in chain.clusters]), digits=0)), $(round(mean([length(c) for c in chain.clusters]), digits=0)), $(maximum([length(c) for c in chain.clusters]))"),
            (:"split #succ/#tot, merge #succ/#tot", split_ratio * ", " * merge_ratio),
            (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
            (:"MAP #attempts/#successes", "$(nb_map_attemps)/$(nb_map_successes)" * (attempt_map ? "" : " (off)")),
            (:"nb clusters, nb>1, smallest(>1), median, mean, largest", "$(length(chain.map_clusters)), $(length(filter(c -> length(c) > 1, chain.map_clusters))), $(minimum(length.(filter(c -> length(c) > 1, chain.map_clusters)))), $(round(median([length(c) for c in chain.map_clusters]), digits=0)), $(round(mean([length(c) for c in chain.map_clusters]), digits=0)), $(maximum([length(c) for c in chain.map_clusters]))"),
            (:"last MAP logprob (minus nn)", "$(round(chain.map_logprob, digits=1)) ($(round(chain.map_logprob - delta_minusnn_map, digits=1)))"),
            (:"last MAP at", last_map_idx),
            (:"last checkpoint at", last_checkpoint)
            ])
        else
            print("\r$(step)/$(nb_steps)")
            print(" (t:$(length(chain.logprob_chain)))")
            print("   lp: $(round(chain.logprob_chain[end], digits=1))")
            print("   sr:" * split_ratio * " mr:" * merge_ratio)
            print("   sps:$(split_per_step), mps:$(merge_per_step)")
            print("   #cl:$(length(chain.clusters)), #cl>1:$(length(filter(x -> length(x) > 1, chain.clusters)))")
            print("   mapattsuc:$(nb_map_attemps)/$(nb_map_successes)")
            print("   lastmap@$(last_map_idx)")
            print("   lchpt@$(last_checkpoint)")
            print("      ")
            if map_success
                print("!      #clmap:$(length(chain.map_clusters))")
                println("   $(round(chain.map_logprob, digits=1))")
            end
            flush(stdout)
        end

        if !isnothing(sample_every) && (stop_criterion === :sample_ess)
            if (chain.samples_idx.length >= chain.samples_idx.capacity
                && samples_convergence.ess >= chain.samples_idx.capacity)
                break
            end
        end

        isfile("stop") && break

    end

    finish!(progbar)

    if pretty_progress === :file
        close(progressio)
    end

    return chain

end


function attempt_map!(chain::FCChain; max_nb_pushes=15, optimize_hyperparams=false, verbose=true, force=false)

    map_clusters_attempt = deepcopy(chain.clusters)
    map_hyperparams = deepcopy(chain.hyperparams)

    # Greedy Gibbs!
    # We only use Gibbs moves in the base space 
    # to construct an approximate MAP state so both
    # map_mll and test_mll use ignoreffjord=true
    map_mll = logprobgenerative(map_clusters_attempt, chain.map_hyperparams, ignoreffjord=true)
    for p in 1:max_nb_pushes
        test_attempt = copy(map_clusters_attempt)
        advance_gibbs!(chain.rng, test_attempt, map_hyperparams; temperature=0.0)
        test_mll = logprobgenerative(test_attempt, map_hyperparams, ignoreffjord=true)
        if test_mll <= map_mll
            # We've regressed, so stop and leave
            # the previous state before this test
            # attempt as the approx. map state
            break
        else
            # the test state led to a better state
            # than the previous one, keep going
            map_clusters_attempt = test_attempt
            map_mll = test_mll
        end
    end

    # if optimize_hyperparams
    #     optimize_hyperparams!(map_clusters_attempt, map_hyperparams, verbose=verbose)
    # end

    attempt_logprob = logprobgenerative(map_clusters_attempt, map_hyperparams, chain.rng, ignorehyperpriors=false, ignoreffjord=false)
    if attempt_logprob > chain.map_logprob || force
        chain.map_logprob = attempt_logprob
        chain.map_clusters = map_clusters_attempt
        chain.map_hyperparams = map_hyperparams
        chain.map_idx = lastindex(chain.logprob_chain)
        return true
    else
        return false
    end
end