function advance_gibbs!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; temperature::T=one(T)) where {T, D, E}
    element_schedule = shuffle!(rng, availableelements(clusters))
    for element in element_schedule
        pop!(clusters, element, deleteifempty=true)
        advance_gibbs!(rng, element, clusters, hyperparams, leaveempty=true, temperature=temperature)
    end
    filter!(!isempty, clusters)
end

function advance_gibbs!(rng::AbstractRNG, element::E, clusters::AbstractVector{C}, hyperparams::AbstractFCHyperparams{T, D}; leaveempty=false, temperature::T=one(T)) where {C <: AbstractCluster{T, D, E}} where {T, D, E}

    alpha, mu, lambda, psi, nu = hyperparams._.pyp.alpha, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    _b2o = first(clusters).b2o

    nbempty = sum(isempty.(clusters))
    if nbempty < 1
        push!(clusters, (C.name.wrapper)(_b2o))
    elseif nbempty >= 2
        throw(InvalidStateException("Found more than one empty cluster before Gibbs move", :gibbssampler))
    end

    log_weights = zeros(T, length(clusters))
    for (i, cluster) in enumerate(clusters)
        log_weights[i] = log_cluster_weight(element, cluster, alpha, mu, lambda, psi, nu)
    end

    if temperature > 0.0
        unnorm_logp = log_weights / temperature
        norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
        probs = Weights(exp.(norm_logp))
        new_assignment = sample(rng, clusters, probs)
    elseif temperature <= 0.0
        _, max_idx = findmax(log_weights)
        new_assignment = clusters[max_idx]
    end

    push!(new_assignment, element)

    if !leaveempty
        filter!(!isempty, clusters)
    end

    return clusters
end


function advance_alpha!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D, E}

    step_distrib = Normal(zero(T), stepsize)

    N = sum(length.(clusters))
    K = length(clusters)

    alpha = hyperparams._.pyp.alpha
    log_alpha = log(alpha)

    proposed_logalpha = log_alpha + rand(rng, step_distrib)
    proposed_alpha = exp(proposed_logalpha)

    log_acceptance = zero(T)

    # because we propose moves on the log scale
    # but need them uniform over alpha > 0
    # before feeding them to the hyperprior

    log_acceptance += K * proposed_logalpha - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
    log_acceptance -= K * log_alpha - loggamma(alpha + N) + loggamma(alpha)

    log_acceptance += log_jeffreys_crp_alpha(proposed_alpha, N) - log_jeffreys_crp_alpha(alpha, N)

    log_hastings = proposed_logalpha - log_alpha
    log_acceptance += log_hastings

    # log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.pyp.alpha = proposed_alpha
        diagnostics.accepted.pyp.alpha += 1
    else
        diagnostics.rejected.pyp.alpha += 1
    end

    return hyperparams
end


function advance_mu!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::Vector{T}=fill(1/10, D), random_order=true) where {T, D, E}

    step_distrib = MvNormal(diagm(stepsize.^2))
    steps = rand(rng, step_distrib)

    lambda, psi, nu = hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    mu = hyperparams._.niw.mu

    if random_order
        dim_order = randperm(rng, D)
    else
        dim_order = 1:D
    end

    for k in dim_order
        proposed_mu = hyperparams._.niw.mu[:]
        proposed_mu[k] = proposed_mu[k] + steps[k]

        log_acceptance = sum([log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), proposed_mu, lambda, psi, nu) - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for c in clusters])

        # log_acceptance = min(zero(T), log_acceptance)

        if log(rand(rng, T)) < log_acceptance
            hyperparams._.niw.mu = proposed_mu
            diagnostics.accepted.niw.mu[k] += 1
        else
            diagnostics.rejected.niw.mu[k] += 1
        end
    end

    return hyperparams
end

function advance_lambda!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)/10) where {T, D, E}

    step_distrib = Normal(zero(T), stepsize)

    mu, lambda, psi, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    proposed_loglambda = log(lambda) + rand(rng, step_distrib)
    proposed_lambda = exp(proposed_loglambda)

    log_acceptance = sum([log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), mu, proposed_lambda, psi, nu) - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for c in clusters])

    # Jeffreys prior over lambda is the logarithmic
    # prior and moves are symmetric on the log-scale
    # so the Hastings factor and hyperprior on lambda
    # cancel each others out.

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.niw.lambda = proposed_lambda
        diagnostics.accepted.niw.lambda += 1
    else
        diagnostics.rejected.niw.lambda += 1
    end

    return hyperparams

end

function advance_psi!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D};
                      stepsize::Vector{T}=fill(1/10, div(D * (D + 1), 2)), random_order=true) where {T, D, E}

    flatL_d = div(D * (D + 1), 2)

    step_distrib = MvNormal(diagm(stepsize.^2))
    steps = rand(rng, step_distrib)

    if random_order
        dim_order = randperm(rng, flatL_d)
    else
        dim_order = 1:flatL_d
    end

    mu, lambda, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.nu

    for k in dim_order

        # We need L for the Hastings factor
        # so we don't use foldpsi()
        L = LowerTriangular(hyperparams._.niw.flatL)
        psi = L * L'

        proposed_flatL = hyperparams._.niw.flatL[:]
        proposed_flatL[k] = proposed_flatL[k] + steps[k]

        # We need proposed_L for the Hastings factor
        # so we don't use foldpsi()
        proposed_L = LowerTriangular(proposed_flatL)
        proposed_psi = proposed_L * proposed_L'

        log_acceptance = sum([log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, proposed_psi, nu) - log_Zniw(cluster, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for cluster in clusters])

        # Go from symmetric and uniform in L to uniform in psi
        # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
        # 2^d's cancel in the Hastings ratio
        log_hastings = sum((D:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
        log_acceptance += log_hastings

        log_acceptance += D * (logdetpsd(psi) - logdetpsd(proposed_psi))

        # log_acceptance = min(zero(T), log_acceptance)

        if log(rand(rng, T)) < log_acceptance
            hyperparams._.niw.flatL = proposed_flatL
            diagnostics.accepted.niw.flatL[k] += 1
        else
            diagnostics.rejected.niw.flatL[k] += 1
        end

    end

    return hyperparams
end

function advance_nu!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D, E}

    step_distrib = Normal(zero(T), stepsize)

    mu, lambda, psi, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    # x = nu - (d - 1)
    # we use moves on the log of x
    # so as to always keep nu > d - 1
    current_logx = log(nu - (D - 1))
    proposed_logx = current_logx + rand(rng, step_distrib)
    proposed_nu = D - 1 + exp(proposed_logx)

    log_acceptance = sum([log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, proposed_nu) - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for c in clusters])

    # Convert back to uniform moves on the positive real line nu > d - 1
    log_hastings = proposed_logx - current_logx
    log_acceptance += log_hastings

    log_acceptance += log_jeffreys_nu(proposed_nu, D) - log_jeffreys_nu(nu, D)

    # log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.niw.nu = proposed_nu
        diagnostics.accepted.niw.nu += 1
    else
        diagnostics.rejected.niw.nu += 1
    end

    return hyperparams
end


function advance_nn_alpha!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D}

    hasnn(hyperparams) || return 0

    step_distrib = Normal(zero(T), stepsize)

    nn_alpha = hyperparams._.nn.prior.alpha
    nn_scale = hyperparams._.nn.prior.scale
    nn_params = hyperparams._.nn.params

    log_nn_alpha = log(nn_alpha)

    proposed_log_nn_alpha = log_nn_alpha + rand(rng, step_distrib)
    proposed_nn_alpha = exp(proposed_log_nn_alpha)

    # Hastings factor on log-scale
    log_acceptance = proposed_log_nn_alpha - log_nn_alpha

    log_acceptance += log_nn_prior(nn_params, proposed_nn_alpha, nn_scale) - log_nn_prior(nn_params, nn_alpha, nn_scale)
    log_acceptance += log_jeffreys_nn(proposed_nn_alpha, nn_scale) - log_jeffreys_nn(nn_alpha, nn_scale)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.nn.prior.alpha = proposed_nn_alpha
        diagnostics.accepted.nn.prior.alpha += 1
    else
        diagnostics.rejected.nn.prior.alpha += 1
    end

    return hyperparams
end

function advance_nn_scale!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D}

    hasnn(hyperparams) || return 0

    step_distrib = Normal(zero(T), stepsize)

    nn_alpha = hyperparams._.nn.prior.alpha
    nn_scale = hyperparams._.nn.prior.scale
    nn_params = hyperparams._.nn.params

    log_nn_scale = log(nn_scale)

    proposed_log_nn_scale = log_nn_scale + rand(rng, step_distrib)
    proposed_nn_scale = exp(proposed_log_nn_scale)

    # Hastings factor for log-scale moves
    log_acceptance = proposed_log_nn_scale - log_nn_scale

    log_acceptance += log_nn_prior(nn_params, nn_alpha, proposed_nn_scale) - log_nn_prior(nn_params, nn_alpha, nn_scale)
    log_acceptance += log_jeffreys_nn(nn_alpha, proposed_nn_scale) - log_jeffreys_nn(nn_alpha, nn_scale)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.nn.prior.scale = proposed_nn_scale
        diagnostics.accepted.nn.prior.scale += 1
    else
        diagnostics.rejected.nn.prior.scale += 1
    end

    return hyperparams
end


# Sequential splitmerge from Dahl & Newcomb
function advance_splitmerge_seq!(rng::AbstractRNG, clusters::AbstractVector{C}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; t::Int=3, temperature::T=one(T))::AbstractVector{<:AbstractCluster{T, D, E}} where {T, D, C <: AbstractCluster{T, D, E}} where E

    @assert t >= 0

    alpha = hyperparams._.pyp.alpha
    mu, lambda, flatL, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.flatL, hyperparams._.niw.nu
    psi = foldpsi(flatL)

    _b2o = first(clusters).b2o

    # cluster_indices = Tuple{Int, E}[(ce, e) for (ce, cluster) in enumerate(clusters) for e in cluster]

    # (ci, ei), (cj, ej) = sample(rng, cluster_indices, 2, replace=false)

    clusterprob = length.(clusters) ./ sum(length.(clusters))
    ci = rand(rng, Categorical(clusterprob))
    ei = rand(rng, collect(clusters[ci]))
    @assert pop!(clusters[ci], ei) === ei

    # New weights with ei removed from the partition
    clusterprob = length.(clusters) ./ sum(length.(clusters))
    cj = rand(rng, Categorical(clusterprob))
    ej = rand(rng, collect(clusters[cj]))
    @assert pop!(clusters[cj], ej) === ej
    
    if ci == cj

        scheduled_elements = E[e for e in clusters[ci] if !(e === ei) && !(e === ej)]

        # Isolate the current merged state
        current_state = C[push!(push!(clusters[ci], ei), ej)]
        # and remove it from the current partition
        deleteat!(clusters, ci)

    elseif ci != cj

        scheduled_elements = E[e for cl in C[clusters[ci], clusters[cj]] for e in cl if !(e === ei) && !(e === ej)]
        
        # Isolate the current split state
        current_state = C[push!(clusters[ci], ei), push!(clusters[cj], ej)]
        # and remove it from the current partition
        deleteat!(clusters, sort([ci, cj]))

    end

    #filter!(!isempty, clusters)

    ClusterConstructor = C.name.wrapper

    # We force a split state. This state is used for both
    # - split moves, in which case it's the proposed state,
    #   and we accumulate the log_q appearing in the numerator
    #   of the  Metropolis ratio, or
    # - merge moves, in which case it's to accumulate log_q
    #   appearing in the denominator of the Metropolis ratio
    #   using a fictitious split state. We do not use the
    #   current state for that purpose because we'd have to
    #   undo it. We could, but it's tricky and annoyhing and
    #   stationarity is not affected by this choice.
    #   Any well mixed split state will do. Neat.
    split_state = [ClusterConstructor([ei], _b2o), ClusterConstructor([ej], _b2o)]

    log_q = 0.0

    # step=0 is the sequential Gibbs step
    # step=1:t are the Gibbs "shuffling" steps
    # step=t+1 is the step during which we accumulate log_q
    # for the inverse move (1 -> 2 clusters) in a merge (2 -> 1 clusters)
    # this step is skipped when accumulating log_q for a split (1 -> 2 clusters)
    for step in 0:t+1

        # if step == t+1
        #     if ci == cj
        #         # Do a last past to the proposed
        #         # split state to accumulate
        #         # q(proposed|launch)
        #         # remember that
        #         # q(launch|proposed) = q(merged|some fictional split state) = 1
        #         append!(launch_state, proposed_state)
        #     elseif ci != cj
        #         # Don't perform last step in a merge,
        #         # keep log_q as the transition probability
        #         # to the launch state, i.e.
        #         # q(launch|proposed) = q(launch|launch-1)
                
        #         # launch_state = proposed_state
        #         append!(launch_state, proposed_state)
        #         break
        #     end
        # end

        log_q = zero(T)
 
        for el in shuffle!(rng, scheduled_elements)

            # Does nothing during first, sequential step
            delete!(split_state, el)

            # Should be true by construction, just
            # making sure we didn't mess up (yet)
            @assert all([!isempty(c) for c in split_state])
            @assert length(split_state) == 2

            #############

            log_weights = zeros(T, length(split_state))

            log_weights = T[log_cluster_weight(el, split_state[1], alpha, mu, lambda, psi, nu),
                            log_cluster_weight(el, split_state[2], alpha, mu, lambda, psi, nu)]

            if temperature > 0
                unnorm_logp = log_weights / T(temperature)
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                new_assignment = rand(rng, Categorical(exp.(norm_logp)))
                log_q += norm_logp[new_assignment]
            elseif temperature <= 0
                _, new_assignment = findmax(log_weights)
                log_q += 0 # symbolic, transition is certain
            end

            push!(split_state[new_assignment], el)

        end

    end

    # At this point if we are doing a split move
    # then log_q = q(split* | single cluster launch state)
    # and if we are doing a merge state
    # then log_q = q(launch|merge)=  q(launch|launch-1)

    if ci != cj
        # Create proposed merge state
        # The previous loop was only to get
        # q(launch|proposed) = q(launch|launch-1)
        # and at this point launch_state = proposed_state
        proposed_state = [ClusterConstructor([e for cl in current_state for e in cl], _b2o)]
    elseif ci == cj
        # In a merge move the proposed state is
        # the split state we just constructed
        # using restricted Gibbs moves
        proposed_state = split_state
    end

    
    if temperature > 0

        log_acceptance = (logprobgenerative(proposed_state, hyperparams, ignorehyperpriors=true, ignoreffjord=true)
                        - logprobgenerative(current_state, hyperparams, ignorehyperpriors=true, ignoreffjord=true))
        log_acceptance /= T(temperature)
    
        # Hastings factor
        if ci != cj
            #   q(proposed | current) / q(current | proposed)
            # = q(split | merge) / q(merge | split)
            # = q / 1
            log_acceptance += log_q
        elseif ci == cj
            #   q(proposed | current) / q(current | proposed)
            # = q(merge | split) / q(split | merge)
            # = 1 / q
            log_acceptance -= log_q
        end

    elseif temperature <= 0
       
        log_acceptance = zero(T)
    
    end


    if log(rand(rng, T)) < log_acceptance
        append!(clusters, proposed_state)
        if ci != cj
            diagnostics.accepted.splitmerge.merge += 1
        elseif ci == cj
            diagnostics.accepted.splitmerge.split += 1
        end
    else
        append!(clusters, current_state)
        if ci != cj
            diagnostics.rejected.splitmerge.merge += 1
        elseif ci == cj
            diagnostics.rejected.splitmerge.split += 1
        end
    end

    return clusters

end


function advance_ffjord!(
    rng::AbstractRNG,
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    hyperparams::AbstractFCHyperparams{T, D},
    diagnostics::AbstractDiagnostics{T, D};
    step_distrib) where {T, D, E}

    hasnn(hyperparams) && !isnothing(step_distrib) || return hyperparams

    proposed_hparray = copy(hyperparams._)
    proposed_hparray.nn.params .= proposed_hparray.nn.params .+ rand(rng, step_distrib)

    proposed_clusters, proposed_delta_logps = reflow(rng, clusters, proposed_hparray, hyperparams.ffjord)

    # reflow() already computes the logprob coming
    # from the jacobian of the FFJORD transformation
    # so we do it here to avoid recomputing it
    # in logprobgenerative()
    log_acceptance = logprobgenerative(proposed_clusters, proposed_hparray, ignorehyperpriors=true)
    log_acceptance -= sum(proposed_delta_logps)
    log_acceptance += log_nn_prior(proposed_hparray.nn.params, hyperparams._.nn.prior.alpha, hyperparams._.nn.prior.scale)

    log_acceptance -= logprobgenerative(rng, clusters, hyperparams._, hyperparams.ffjord, ignorehyperpriors=true)

    if log(rand(rng, T)) < log_acceptance
        empty!(clusters)
        append!(clusters, proposed_clusters)
        hyperparams._ .= proposed_hparray
        diagnostics.accepted.nn.params += 1
    else
        diagnostics.rejected.nn.params += 1
    end

    return hyperparams

end

function advance_hyperparams_adaptive!(
    rng::AbstractRNG,
    clusters::Vector{<:AbstractCluster{T, D, E}},
    hyperparams::AbstractFCHyperparams{T, D},
    diagnostics::AbstractDiagnostics{T, D};
    amwg_batch_size=40, acceptance_target::T=T(0.44),
    nb_ffjord_am=1, am_safety_probability::T=T(0.05), am_safety_sigma::T=T(0.3),
    hyperparams_chain=nothing) where {T, D, E}


    di = diagnostics
    # by default only resets hyperparams acceptance rates
    clear_diagnostics!(di)

    nn_D = hasnn(hyperparams) ? size(hyperparams._.nn.params, 1) : 0

    for i in 1:amwg_batch_size
        advance_alpha!(rng, clusters, hyperparams, di, stepsize=exp(di.amwg.logscales.pyp.alpha))
        advance_mu!(rng, clusters, hyperparams, di, stepsize=exp.(di.amwg.logscales.niw.mu))
        advance_lambda!(rng, clusters, hyperparams, di, stepsize=exp(di.amwg.logscales.niw.lambda))
        advance_psi!(rng, clusters, hyperparams, di, stepsize=exp.(di.amwg.logscales.niw.flatL))
        advance_nu!(rng, clusters, hyperparams, di, stepsize=exp(di.amwg.logscales.niw.nu))
        if hasnn(hyperparams)
            # advance_nn_alpha!(rng, hyperparams, di, stepsize=exp(di.amwg.logscales.nn.prior.alpha))
            # advance_nn_scale!(rng, hyperparams, di, stepsize=exp(di.amwg.logscales.nn.prior.scale))
        end
    end
    di.amwg.nbbatches += 1

    adjust_amwg_logscales!(di, acceptance_target=acceptance_target)

    if hasnn(hyperparams) && nb_ffjord_am > 0

        if length(hyperparams_chain) <= 2 * nn_D

            step_distrib = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))

        else

            nn_sigma = am_sigma(diagnostics)

            safety_component = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))
            empirical_estimate_component = MvNormal(2.38^2 / nn_D * nn_sigma)

            step_distrib = MixtureModel([safety_component, empirical_estimate_component], [am_safety_probability, 1 - am_safety_probability])
        end

        for i in 1:nb_ffjord_am
            advance_ffjord!(rng, clusters, hyperparams, di, step_distrib=step_distrib)
        end

        diagnostics.am.L += 1
        diagnostics.am.x .+= hyperparams._.nn.params
        diagnostics.am.xx .+= hyperparams._.nn.params * hyperparams._.nn.params'

    end

    return hyperparams

end


# function advance_gibbs!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; temperature::T=one(T)) where {T, D, E}
#     return advance_gibbs!(default_rng(), clusters, hyperparams, temperature=temperature)
# end
# function advance_splitmerge_seq!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; t::Int=3, temperature::T=one(T)) where {T, D, E}
#     return advance_splitmerge_seq!(default_rng(), clusters, hyperparams, diagnostics, t=t, temperature=temperature)
# end

# function advance_alpha!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D, E}
#     return advance_alpha!(default_rng(), clusters, hyperparams, diagnostics, stepsize=stepsize)
# end
# function advance_mu!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::Vector{T}=fill(1/10, D), random_order=true) where {T, D, E}
#     return advance_mu!(default_rng(), clusters, hyperparams, diagnostics, stepsize=stepsize, random_order=random_order)
# end
# function advance_lambda!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)/10) where {T, D, E}
#     return advance_lambda!(default_rng(), clusters, hyperparams, diagnostics, stepsize=stepsize)
# end
# function advance_psi!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::Vector{T}=fill(1/10, div(D * (D + 1), 2)), random_order=true) where {T, D, E}
#     return advance_psi!(default_rng(), clusters, hyperparams, diagnostics, stepsize=stepsize, random_order=random_order)
# end
# function advance_nu!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D, E}
#     return advance_nu!(default_rng(), clusters, hyperparams, diagnostics, stepsize=stepsize)
# end
# function advance_nn_alpha!(hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D}
#     return advance_nn_alpha!(default_rng(), hyperparams, diagnostics, stepsize=stepsize)
# end
# function advance_nn_scale!(hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; stepsize::T=one(T)) where {T, D}
#     return advance_nn_scale!(default_rng(), hyperparams, diagnostics, stepsize=stepsize)
# end
# function advance_ffjord!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, diagnostics::AbstractDiagnostics{T, D}; step_distrib=nothing) where {T, D, E}
#     return advance_ffjord!(default_rng(), clusters, hyperparams, diagnostics, step_distrib=step_distrib)
# end

function am_sigma(L::Real, x::AbstractVector{T}, xx::AbstractMatrix{T}; correction=true, eps::T=one(T)e-10) where T
    sigma = (xx - x * x' / L) / (L - 1)
    if correction
        sigma = (sigma + sigma') / 2 + eps * I
    end
    return sigma
end

am_sigma(diagnostics::DiagnosticsFFJORD{T, D}; correction=true, eps::T=T(1e-10)) where {T, D} = am_sigma(diagnostics.am.L, diagnostics.am.x, diagnostics.am.xx, correction=correction, eps=eps) #: zeros(Float64, 0, 0)

function adjust_amwg_logscales!(diagnostics::AbstractDiagnostics{T, D}; acceptance_target::T=0.44, min_delta::T=0.01) where {T, D} #, minmax_logscale::T=one(T)0.0)
    delta_n = min(min_delta, 1/sqrt(diagnostics.amwg.nbbatches))

    contparams = hasnn(diagnostics) ? (:pyp, :niw, :nn) : (:pyp, :niw)
    acc_rates = diagnostics.accepted[contparams] ./ (diagnostics.accepted[contparams] .+ diagnostics.rejected[contparams])
    diagnostics.amwg.logscales .+= delta_n .* (acc_rates .> acceptance_target) .- delta_n .* (acc_rates .<= acceptance_target)

    # diagnostics.amwg_logscales[diagnostics.amwg_logscales .< -minmax_logscale] .= -minmax_logscale
    # diagnostics.amwg_logscales[diagnostics.amwg_logscales .> minmax_logscale] .= minmax_logscale

    return diagnostics
end