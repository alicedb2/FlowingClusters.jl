function advance_gibbs!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; temperature::T=one(T)) where {T, D, E}
    element_schedule = shuffle!(rng, availableelements(clusters))
    for element in element_schedule
        pop!(clusters, element, delete_empty=true)
        advance_gibbs!(rng, element, clusters, hyperparams, temperature=temperature)
    end
    return filter!(!isempty, clusters)
end

function advance_gibbs!(rng::AbstractRNG, element::E, clusters::AbstractVector{C}, hyperparams::AbstractFCHyperparams{T, D}; temperature::T=one(T)) where {C <: AbstractCluster{T, D, E}} where {T, D, E}

    alpha, mu, lambda, psi, nu = hyperparams._.pyp.alpha, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    _b2o = first(clusters).b2o

    nbempty =  sum(isempty.(clusters))
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

    # nbempty = sum(isempty.(clusters))
    # if nbempty < 1
    #     push!(clusters, (C.name.wrapper)(_b2o))
    # elseif nbempty >= 2
    #     throw(InvalidStateException("Found more than one empty cluster after Gibbs move", :gibbssampler))
    # end
end


function advance_alpha!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D, E}

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

    log_acceptance += log(jeffreys_alpha(proposed_alpha, N)) - log(jeffreys_alpha(alpha, N))

    log_hastings = proposed_logalpha - log_alpha
    log_acceptance += log_hastings

    log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.pyp.alpha = proposed_alpha
        return 1
    else
        return 0
    end

end


function advance_mu!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::Vector{T}=fill(1/10, D), random_order=true)::Vector{Int} where {T, D, E}

    step_distrib = MvNormal(diagm(stepsize.^2))
    steps = rand(rng, step_distrib)

    lambda, psi, nu = hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    mu = hyperparams._.niw.mu
    accepted_mu = zeros(Int, D)

    if random_order
        dim_order = randperm(D)
    else
        dim_order = 1:D
    end

    for i in dim_order
        proposed_mu = hyperparams._.niw.mu[:]
        proposed_mu[i] = proposed_mu[i] + steps[i]

        log_acceptance = sum([log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), proposed_mu, lambda, psi, nu) - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for c in clusters])

        log_acceptance = min(zero(T), log_acceptance)

        if log(rand(rng, T)) < log_acceptance
            hyperparams._.niw.mu = proposed_mu
            accepted_mu[i] = 1
        end
    end

    return accepted_mu
    
end

function advance_lambda!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T)/10)::Int where {T, D, E}

    step_distrib = Normal(zero(T), stepsize)

    mu, lambda, psi, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    proposed_loglambda = log(lambda) + rand(rng, step_distrib)
    proposed_lambda = exp(proposed_loglambda)

    log_acceptance = sum([log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), mu, proposed_lambda, psi, nu) - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for c in clusters])

    # We leave loghastings = 0 because the
    # Jeffreys prior over lambda is the logarithmic
    # prior and moves are symmetric on the log scale.

    log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.niw.lambda = proposed_lambda
        return 1
    else
        return 0
    end

end

function advance_psi!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D};
                      stepsize::Vector{T}=fill(1/10, div(D * (D + 1), 2)), random_order=true)::Vector{Int} where {T, D, E}

    flatL_d = div(D * (D + 1), 2)

    step_distrib = MvNormal(diagm(stepsize.^2))
    steps = rand(rng, step_distrib)

    if random_order
        dim_order = randperm(flatL_d)
    else
        dim_order = 1:flatL_d
    end

    mu, lambda, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.nu

    accepted_flatL = zeros(Int, flatL_d)

    for k in dim_order

        L = LowerTriangular(hyperparams._.niw.flatL)
        psi = L * L'
        
        proposed_flatL = hyperparams._.niw.flatL[:]
        proposed_flatL[k] = proposed_flatL[k] + steps[k]

        proposed_L = LowerTriangular(proposed_flatL)
        proposed_psi = proposed_L * proposed_L'

        log_acceptance = sum([log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, proposed_psi, nu) - log_Zniw(cluster, mu, lambda, psi, nu) + log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu) for cluster in clusters])

        # Go from symmetric and uniform in L to uniform in psi
        # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
        # 2^d's cancel in the Hastings ratio
        log_hastings = sum((D:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
        log_acceptance += log_hastings

        log_acceptance += D * (logdetpsd(psi) - logdetpsd(proposed_psi))

        log_acceptance = min(zero(T), log_acceptance)

        if log(rand(rng, T)) < log_acceptance
            hyperparams._.niw.flatL = proposed_flatL
            accepted_flatL[k] = 1
        end

    end
    
    return accepted_flatL

end

function advance_nu!(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D, E}

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

    log_acceptance += log(jeffreys_nu(proposed_nu, D)) - log(jeffreys_nu(nu, D))

    log_acceptance = min(zero(T), log_acceptance)
    if log(rand(rng, T)) < log_acceptance
        hyperparams._.niw.nu = proposed_nu
        return 1
    else
        return 0
    end

end


function advance_nn_alpha!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D}

    hasnn(hyperparams) || return 0

    step_distrib = Normal(zero(T), stepsize)

    nn_alpha = hyperparams._.nn.t.alpha
    nn_scale = hyperparams._.nn.t.scale
    nn_params = hyperparams._.nn.params    

    log_nn_alpha = log(nn_alpha)

    proposed_log_nn_alpha = log_nn_alpha + rand(rng, step_distrib)
    proposed_nn_alpha = exp(proposed_log_nn_alpha)

    log_acceptance = nn_prior(nn_params, proposed_nn_alpha) - nn_prior(nn_params, nn_alpha)

    # Hastings factor on log-scale
    log_acceptance += proposed_log_nn_alpha - log_nn_alpha

    # log_acceptance += log(jeffreys_t_alpha(proposed_nn_alpha)) - log(jeffreys_t_alpha(nn_alpha))
    log_acceptance += log_jeffreys_t(proposed_nn_alpha, nn_scale) - log_jeffreys_t(nn_alpha, nn_scale)

    log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams.nn_alpha = proposed_nn_alpha
        return 1
    else
        return 0
    end

end

function advance_nn_scale!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D}

    hasnn(hyperparams) || return 0

    step_distrib = Normal(zero(T), stepsize)

    nn_alpha = hyperparams._.nn.t.alpha
    nn_scale = hyperparams._.nn.t.scale
    nn_params = hyperparams._.nn.params
    
    log_nn_scale = log(nn_scale)

    proposed_log_nn_scale = log_nn_scale + rand(rng, step_distrib)
    proposed_nn_scale = exp(proposed_log_nn_scale)

    log_acceptance = nn_prior(nn_params, nn_alpha, proposed_nn_scale) - nn_prior(nn_params, nn_alpha, nn_scale)
 
    # Comment next two line for independence Jeffreys prior on nn_scale
    log_acceptance += proposed_log_nn_scale - log_nn_scale # Hastings factor
    log_acceptance += log_jeffreys_t(nn_alpha, proposed_nn_scale) - log_jeffreys_t(nn_alpha, nn_scale)
    
    log_acceptance = min(zero(T), log_acceptance)

    if log(rand(rng, T)) < log_acceptance
        hyperparams._.nn.t.scale = proposed_nn_scale
        return 1
    else
        return 0
    end

end


# # Sequential splitmerge from Dahl & Newcomb
# function advance_splitmerge_seq!(clusters::AbstractVector{C}, hyperparams::AbstractFCHyperparams{T, D}; t::Int=3, temperature::T=one(T)) where {T, D, C <: AbstractCluster{T, D, E}}

#     @assert t >= 0

#     alpha, mu, lambda, psi, nu = hyperparams._.pyp.alpha, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu
#     d = dimension(hyperparams)

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
#             return [-1, 1]
#         elseif ci == cj
#             return [1, -1]
#         end
#     else
#         append!(clusters, initial_state)
#         if ci != cj
#             return [-1, 0]
#         elseif ci == cj
#             return [0, -1]
#         end
#     end

# end



# function advance_ffjord!(
#     rng::AbstractRNG,
#     clusters::AbstractVector{<:AbstractCluster{T, D, E}},
#     hyperparams::AbstractFCHyperparams{T, D},
#     base2original::Dict{Vector{T}, Vector{T}};
#     step_distrib=nothing,
#     temperature::T=one(T))::Int where {T, D, E}

#     hasnn(hyperparams) && !isnothing(step_distrib) || return 0

#     ffjord_model = FFJORD(hyperparams.nn, (0.0, 1.0), (dimension(hyperparams),), Tsit5(), basedist=nothing, ad=AutoForwardDiff())    
#     original_clusters = realspace_clusters(Matrix, clusters, base2original)

#     proposed_nn_params = hyperparams._.nn.params .+ rand(step_distrib)

#     # We could have left the calculation of deltalogps
#     # to logprobgenerative below, but we a proposal comes
#     # a new base2original so we do both at once here and
#     # call logprobgenerative with ffjord=false

#     original_elements = reduce(hcat, original_clusters)
#     proposed_base, _ = ffjord_model(original_elements, proposed_nn_params, hyperparams.nns)
#     proposed_elements = Matrix{Float64}(proposed_base.z)
#     proposed_baseclusters = chunk(proposed_elements, size.(original_clusters, 2))
#     proposed_base2original = Dict{Vector{Float64}, Vector{Float64}}(eachcol(proposed_elements) .=> eachcol(original_elements))

#     log_acceptance = -sum(proposed_base.delta_logp)

#     # We already accounted for the ffjord deltalogps above
#     # so call logprobgenerative with ffjord=false on the
#     # proposed state.
#     log_acceptance += logprobgenerative(Cluster.(proposed_baseclusters), hyperparams, proposed_base2original, hyperpriors=false, ffjord=false) - logprobgenerative(clusters, hyperparams, base2original, hyperpriors=false, ffjord=true)

#     # We called logprobgenerative with ffjord=true on the current state
#     # but not on the proposed state, so we need to account for the
#     # prior on the neural network for the proposed state
#     log_acceptance += nn_prior(proposed_nn_params, hyperparams._.nn.t.alpha, hyperparams._.nn.t.scale)

#     log_acceptance /= temperature

#     log_acceptance = min(0.0, log_acceptance)
#     if log(rand()) < log_acceptance
#         hyperparams._.nn.params = proposed_nn_params
#         empty!(clusters)
#         append!(clusters, Cluster.(proposed_baseclusters))
#         empty!(base2original)
#         merge!(base2original, proposed_base2original)
#         return 1
#     else
#         return 0
#     end

# end


# function advance_hyperparams_adaptive!(
#     clusters::Vector{<:AbstractCluster{T, D, E}},
#     hyperparams::AbstractFCHyperparams{T, D};
#     amwg_batch_size=40, acceptance_target::T=0.44,
#     nb_ffjord_am=1, am_safety_probability::T=0.05, am_safety_sigma::T=0.1,
#     hyperparams_chain=nothing, temperature::T=one(T))::Int where {T, D, E}

#     # di = diagnostics
#     # d = dimension(hyperparams)

#     # by default only resets hyperparams acceptance rates
#     clear_diagnostics!(di)

#     nn_D = hasnn(hyperparams) ? size(hyperparams._.nn.params, 1) : 0

#     for i in 1:amwg_batch_size
#             advance_alpha!(clusters, hyperparams, stepsize=exp(di.amwg_logscales.pyp.alpha))
#             advance_mu!(clusters, hyperparams, stepsize=exp.(di.amwg_logscales.niw.mu))
#             advance_lambda!(clusters, hyperparams, stepsize=exp(di.amwg_logscales.niw.lambda))
#             advance_psi!(clusters, hyperparams,stepsize=exp.(di.amwg_logscales.niw.flatL))
#             advance_nu!(clusters, hyperparams, stepsize=exp(di.amwg_logscales.niw.nu))
#             if hasnn(hyperparams)
#                 advance_nn_alpha!(hyperparams, stepsize=exp(di.amwg_logscales.nn.t.alpha))
#                 advance_nn_scale!(hyperparams, stepsize=exp(di.amwg_logscales.nn.t.scale))
#             end
#     end

#     di.amwg_nbbatches += 1

#     adjust_amwg_logscales!(di, acceptance_target=acceptance_target)

#     if hasnn(hyperparams) && nb_ffjord_am > 0

#         if length(hyperparams_chain) <= 4 * nn_D

#             step_distrib = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))

#         elseif length(hyperparams_chain) > 4 * nn_D

#             nn_sigma = am_sigma(diagnostics)

#             safety_component = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))
#             empirical_estimate_component = MvNormal(2.38^2 / nn_D * nn_sigma)

#             step_distrib = MixtureModel([safety_component, empirical_estimate_component], [am_safety_probability, 1 - am_safety_probability])
#         end

#         if hasnn(hyperparams)
#             for i in 1:nb_ffjord_am
#                 advance_ffjord!(clusters, hyperparams, base2original,
#                                 step_distrib=step_distrib, temperature=temperature)
#             end

#             diagnostics.am.L += 1
#             diagnostics.am.x .+= hyperparams._.nn.params
#             diagnostics.am.xx .+= hyperparams._.nn.params * hyperparams._.nn.params'
#         end
#     end

#     return hyperparams

# end


function advance_gibbs!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; temperature::T=one(T)) where {T, D, E}
    return advance_gibbs!(default_rng(), clusters, hyperparams, temperature=temperature)
end
function advance_alpha!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D, E}
    return advance_alpha!(default_rng(), clusters, hyperparams, stepsize=stepsize)
end
function advance_mu!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::Vector{T}=fill(1/10, D), random_order=true)::Vector{Int} where {T, D, E}
    return advance_mu!(default_rng(), clusters, hyperparams, stepsize=stepsize, random_order=random_order)
end
function advance_lambda!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T)/10)::Int where {T, D, E}
    return advance_lambda!(default_rng(), clusters, hyperparams, stepsize=stepsize)
end
function advance_psi!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::Vector{T}=fill(1/10, div(D * (D + 1), 2)), random_order=true)::Vector{Int} where {T, D, E}
    return advance_psi!(default_rng(), clusters, hyperparams, stepsize=stepsize, random_order=random_order)
end
function advance_nu!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D, E}
    return advance_nu!(default_rng(), clusters, hyperparams, stepsize=stepsize)
end
function advance_nn_alpha!(hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D}
    return advance_nn_alpha!(default_rng(), hyperparams, stepsize=stepsize)
end
function advance_nn_scale!(hyperparams::AbstractFCHyperparams{T, D}; stepsize::T=one(T))::Int where {T, D}
    return advance_nn_scale!(default_rng(), hyperparams, stepsize=stepsize)
end
# function advance_ffjord!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; step_distrib=nothing, temperature::T=one(T))::Int where {T, D, E}
#     return advance_ffjord!(default_rng(), clusters, hyperparams, step_distrib=step_distrib, temperature=temperature)
# end