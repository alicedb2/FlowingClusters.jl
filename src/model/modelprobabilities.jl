function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}; ignorehyperpriors=false, temperature::T=one(T)) where {T, D, E}

    sum(length.(clusters)) > 1 || return -Inf

    hpa = hyperparamsarray

    alpha = hpa.pyp.alpha
    mu, lambda, psi, nu = niwparams(hpa)

    N = sum(length.(clusters))
    K = length(clusters)

    # psi is always valid by construction
    # except perhaps when logdet is too small/large
    if alpha <= 0 || lambda <= 0 || nu <= D - 1
        return -Inf
    end

    # Log-probability associated with the Chinese Restaurant Process
    log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum(loggamma.(length.(clusters)))

    # Log-probability associated with the data likelihood
    # and Normal-Inverse-Wishart base distribution of the CRP
    log_niw = zero(T)
    for cluster in clusters
        log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - length(cluster) * D/2 * log(2pi)
    end
    log_niw -= K * log_Zniw(EmptyCluster{T, D, E}(), mu, lambda, psi, nu)

    log_hyperpriors = zero(T)

    if !ignorehyperpriors
        # mu0 has a flat hyperpriors
        # alpha hyperprior
        log_hyperpriors += log_jeffreys_crp_alpha(alpha, N)

        # NIW hyperpriors
        log_hyperpriors += log_jeffreys_lambda(lambda)
        log_hyperpriors += log_jeffreys_psi(psi)
        log_hyperpriors += log_jeffreys_nu(nu, D)

    end

    logprob = (log_crp + log_niw + log_hyperpriors) / temperature

    return isfinite(logprob) ? logprob : -Inf

end

function logprobgenerative(rng::AbstractRNG, clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; ignorehyperpriors=false, temperature::T=one(T)) where {T, D, E}

    hpa = hyperparamsarray

    hasnn(hpa) || throw(ArgumentError("You must provide a neural network when using FFJORD"))

    logprob_noffjord = logprobgenerative(clusters, hyperparamsarray; ignorehyperpriors=ignorehyperpriors)

    if !isfinite(logprob_noffjord)
        return -Inf
    end

    ret = forwardffjord(rng, Matrix(clusters, orig=true), hpa, ffjord)
    logprob_ffjord = -sum(ret.deltalogpxs)
    # logprob_ffjord += log_nn_prior(hpa.nn.params, hpa.nn.prior...)
    # logprob_ffjord += log_nn_prior(ret.z)

    if !ignorehyperpriors
        # logprob_ffjord += log_nn_hyperprior(hpa.nn.prior...)
    end

    logprob = (logprob_noffjord + logprob_ffjord) / temperature

    return isfinite(logprob) ? logprob : -Inf

end

function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}, rng::Union{Nothing, AbstractRNG}=default_rng(); ignorehyperpriors::Bool=false, ignoreffjord::Bool=false, temperature::T=one(T)) where {T, D, E}
    if hasnn(hyperparams) && !ignoreffjord
        (rng isa AbstractRNG)|| throw(ArgumentError("You must provide a random number generator when using FFJORD"))
        return logprobgenerative(rng, clusters, hyperparams._, hyperparams.ffjord, ignorehyperpriors=ignorehyperpriors, temperature=temperature)
    else
        return logprobgenerative(clusters, hyperparams._, ignorehyperpriors=ignorehyperpriors, temperature=temperature)
    end
end