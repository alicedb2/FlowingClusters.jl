function advance_gibbs!(clusters::Vector{Cluster}, hyperparams::FCHyperparams; temperature=1.0)

    scheduled_elements = shuffle!(elements(clusters))

    for element in scheduled_elements
        pop!(clusters, element)
        advance_gibbs!(element, clusters, hyperparams, temperature=temperature)
    end

    return clusters

end

function advance_gibbs!(element::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::FCHyperparams; temperature=1.0)

    alpha, mu, lambda, psi, nu = hyperparams._.pyp.alpha, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu
    d = length(mu)

    if sum(isempty.(clusters)) < 1
        push!(clusters, Cluster(d))
    end

    log_weights = zeros(length(clusters))
    for (i, cluster) in enumerate(clusters)
        log_weights[i] = log_cluster_weight(element, cluster, alpha, mu, lambda, psi, nu)
    end

    if temperature > 0.0
        unnorm_logp = log_weights / temperature
        norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
        probs = Weights(exp.(norm_logp))
        new_assignment = sample(clusters, probs)
    elseif temperature <= 0.0
        _, max_idx = findmax(log_weights)
        new_assignment = clusters[max_idx]
    end

    push!(new_assignment, element)

    filter!(!isempty, clusters)

end


function advance_alpha!(clusters::Vector{Cluster}, hyperparams::FCHyperparams; step_size=0.5)

    step_distrib = Normal(0.0, step_size)

    N = sum([length(c) for c in clusters])
    K = length(clusters)

    alpha = hyperparams._.pyp.alpha
    log_alpha = log(alpha)

    proposed_logalpha = log_alpha + rand(step_distrib)
    proposed_alpha = exp(proposed_logalpha)

    log_acceptance = 0.0

    # because we propose moves on the log scale
    # but need them uniform over alpha > 0
    # before feeding them to the hyperprior

    log_acceptance += K * proposed_logalpha - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
    log_acceptance -= K * log_alpha - loggamma(alpha + N) + loggamma(alpha)

    log_acceptance += log(jeffreys_alpha(proposed_alpha, N)) - log(jeffreys_alpha(alpha, N))

    log_hastings = proposed_logalpha - log_alpha
    log_acceptance += log_hastings

    log_acceptance = min(0.0, log_acceptance)

    if log(rand()) < log_acceptance
        hyperparams._.pyp.alpha = proposed_alpha
        return 1
    else
        return 0
    end

end

function advance_mu!(clusters::Vector{Cluster}, hyperparams::FCHyperparams;
                     random_order=true, step_size=fill(0.1, dimension(hyperparams)))

    d = dimension(hyperparams)

    step_distrib = MvNormal(diagm(step_size.^2))
    steps = rand(step_distrib)

    lambda, psi, nu = hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    mu = hyperparams._.niw.mu
    accepted_mu = zeros(Int64, d)

    if random_order
        dim_order = randperm(d)
    else
        dim_order = 1:d
    end

    for i in dim_order
        proposed_mu = hyperparams._.niw.mu[:]
        proposed_mu[i] = proposed_mu[i] + steps[i]

        log_acceptance = (
        sum(log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(nothing, proposed_mu, lambda, psi, nu)
            - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu)
        for c in clusters)
        )

        log_acceptance = min(0.0, log_acceptance)

        if log(rand()) < log_acceptance
            hyperparams._.niw.mu = proposed_mu
            accepted_mu[i] = 1
        end
    end

    return accepted_mu
    
end

function advance_lambda!(clusters::Vector{Cluster}, hyperparams::FCHyperparams; step_size=0.1)

    step_distrib = Normal(0.0, step_size)

    mu, lambda, psi, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu

    proposed_loglambda = log(lambda) + rand(step_distrib)
    proposed_lambda = exp(proposed_loglambda)

    log_acceptance = sum(log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(nothing, mu, proposed_lambda, psi, nu)
                       - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu)
                    for c in clusters)

    # We leave loghastings = 0.0 because the
    # Jeffreys prior over lambda is the logarithmic
    # prior and moves are symmetric on the log scale.

    log_acceptance = min(0.0, log_acceptance)

    if log(rand()) < log_acceptance
        hyperparams._.niw.lambda = proposed_lambda
        return 1
    else
        return 0
    end

end

function advance_psi!(clusters::Vector{Cluster}, hyperparams::FCHyperparams;
                      random_order=true, step_size=fill(0.1, length(hyperparams.flatL)))

    flatL_d = length(hyperparams._.niw.flatL)

    step_distrib = MvNormal(diagm(step_size.^2))
    steps = rand(step_distrib)

    if random_order
        dim_order = randperm(flatL_d)
    else
        dim_order = 1:flatL_d
    end

    mu, lambda, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.nu

    accepted_flatL = zeros(Int64, flatL_d)

    d = length(mu)

    for k in dim_order

        L = foldL(hyperparams._.niw.flatL)
        psi = L * L'
        
        proposed_flatL = hyperparams._.niw.flatL[:]
        proposed_flatL[k] = proposed_flatL[k] + steps[k]

        proposed_L = foldL(proposed_flatL)
        proposed_psi = proposed_L * proposed_L'

        log_acceptance = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                           - log_Zniw(cluster, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu)
                        for cluster in clusters)

        # Go from symmetric and uniform in L to uniform in psi
        # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
        # 2^d's cancel in the Hastings ratio
        log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
        log_acceptance += log_hastings

        log_acceptance += d * (logdetpsd(psi) - logdetpsd(proposed_psi))

        log_acceptance = min(0.0, log_acceptance)

        if log(rand()) < log_acceptance
            hyperparams._.niw.flatL = proposed_flatL
            accepted_flatL[k] = 1
        end

    end
    
    return accepted_flatL

end

function advance_nu!(clusters::Vector{Cluster}, hyperparams::FCHyperparams; step_size=1.0)

    step_distrib = Normal(0.0, step_size)

    mu, lambda, psi, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu
    d = length(mu)

    # x = nu - (d - 1)
    # we use moves on the log of x
    # so as to always keep nu > d - 1
    current_logx = log(nu - (d - 1))
    proposed_logx = current_logx + rand(step_distrib)
    proposed_nu = d - 1 + exp(proposed_logx)

    log_acceptance = sum(log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(nothing, mu, lambda, psi, proposed_nu)
                       - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu)
                for c in clusters)

    # Convert back to uniform moves on the positive real line nu > d - 1
    log_hastings = proposed_logx - current_logx
    log_acceptance += log_hastings

    log_acceptance += log(jeffreys_nu(proposed_nu, d)) - log(jeffreys_nu(nu, d))

    log_acceptance = min(0.0, log_acceptance)

    if log(rand()) < log_acceptance
        hyperparams.nu = proposed_nu
        return 1
    else
        return 0
    end

end

function advance_nn_alpha!(hyperparams::FCHyperparams; step_size=1.0)

    hasnn(hyperparams) || return 0

    step_distrib = Normal(0.0, step_size)

    nn_alpha = hyperparams._.nn.t.alpha
    nn_scale = hyperparams._.nn.t.scale

    log_nn_alpha = log(nn_alpha)

    proposed_log_nn_alpha = log_nn_alpha + rand(step_distrib)
    proposed_nn_alpha = exp(proposed_log_nn_alpha)

    log_acceptance = nn_prior(hyperparams.nn_params, proposed_nn_alpha) - nn_prior(hyperparams.nn_params, nn_alpha)

    # Hastings factor on log-scale
    log_acceptance += proposed_log_nn_alpha - log_nn_alpha

    # log_acceptance += log(jeffreys_t_alpha(proposed_nn_alpha)) - log(jeffreys_t_alpha(nn_alpha))
    log_acceptance += log_jeffreys_t(proposed_nn_alpha, nn_scale) - log_jeffreys_t(nn_alpha, nn_scale)

    log_acceptance = min(0.0, log_acceptance)

    if log(rand()) < log_acceptance
        hyperparams.nn_alpha = proposed_nn_alpha
        return 1
    else
        return 0
    end

end

function advance_nn_scale!(hyperparams::FCHyperparams; step_size=1.0)

    hasnn(hyperparams) || return 0

    step_distrib = Normal(0.0, step_size)

    nn_alpha = hyperparams._.nn.t.alpha
    nn_scale = hyperparams._.nn.t.scale

    log_nn_scale = log(nn_scale)

    proposed_log_nn_scale = log_nn_scale + rand(step_distrib)
    proposed_nn_scale = exp(proposed_log_nn_scale)

    log_acceptance = (nn_prior(hyperparams._.nn.params, nn_alpha, proposed_nn_scale) 
                    - nn_prior(hyperparams._.nn.params, nn_alpha, nn_scale))

    # Hastings factor
    log_acceptance += proposed_log_nn_scale - log_nn_scale

    log_acceptance += log_jeffreys_t(nn_alpha, proposed_nn_scale) - log_jeffreys_t(nn_alpha, nn_scale)

    log_acceptance = min(0.0, log_acceptance)

    if log(rand()) < log_acceptance
        hyperparams._.nn.t.scale = proposed_nn_scale
        return 1
    else
        return 0
    end

end