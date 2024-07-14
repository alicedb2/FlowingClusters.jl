function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparams::AbstractFCHyperparams{T, D}; ignorehyperpriors=false, ignoreffjord=false, temperature::T=one(T))::T where {T, D, E}
    if hasnn(hyperparams)
        return logprobgenerative(clusters, hyperparams._, hyperparams.nn, ignorehyperpriors=ignorehyperpriors, ignoreffjord=ignoreffjord, temperature=temperature)
    else
        return logprobgenerative(clusters, hyperparams._, ignorehyperpriors=ignorehyperpriors, temperature=temperature)
    end
end

function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}; ignorehyperpriors=false, temperature::T=one(T))::T where {T, D, E}

    sum(length.(clusters)) > 1 || return -Inf

    hpa = hyperparamsarray

    alpha = hpa.pyp.alpha
    mu, lambda, psi, nu = niwparams(hpa)

    N = sum(length.(clusters))
    K = length(clusters)

    # psi is always valid by construction
    # except perhaps when logdet is too small/large
    if alpha <= 0 || lambda <= 0 || nu <= D - 1 || (hasnn(hpa) && (hpa.nn.t.alpha <= 0 || hpa.nn.t.scale <= 0))
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
        log_hyperpriors += log(jeffreys_alpha(alpha, N))

        # NIW hyperpriors
        log_hyperpriors += -log(lambda)
        log_hyperpriors += -D * logdetpsd(psi)
        log_hyperpriors += log(jeffreys_nu(nu, D))

    end

    log_p = log_crp + log_niw + log_hyperpriors

    return isfinite(log_p) ? log_p / temperature : -Inf

end

function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, nn::NamedTuple; ignorehyperpriors=false, ignoreffjord=false, temperature::T=one(T))::T where {T, D, E}

    hpa = hyperparamsarray

    logprob_noffjord = logprobgenerative(clusters, hyperparamsarray; ignorehyperpriors=ignorehyperpriors, temperature=temperature)

    logprob_ffjord = zero(T)

    if hasnn(hpa) && !ignoreffjord
        logprob_ffjord -= sum(forwardffjord(Matrix(clusters, orig=true), hpa, nn).deltalogpxs)
        logprob_ffjord += nn_prior(hpa.nn.params, hpa.nn.t.alpha, hpa.nn.t.scale)
    end

    ## Still debating whether nn_prior is an hyperprior or not
    if hasnn(hpa) && !ignoreffjord && !ignorehyperpriors
        # Independence Jeffreys prior
        # log_hyperpriors += log(jeffreys_t_alpha(hpa.nn.t.alpha))
        # log_hyperpriors -= log(hpa.nn.t.scale)
        
        # Bivariate Jeffreys prior
        logprob_ffjord += log_jeffreys_t(hpa.nn.t.alpha, hpa.nn.t.scale)
    end

    logprob = logprob_noffjord + logprob_ffjord / temperature

    return isfinite(logprob) ? logprob : -Inf

end
