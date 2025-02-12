function log_logarithmic(x::T; lowerbound=T(10^-5), upperbound=T(1000)) where T
    if lowerbound < x <= upperbound
        ret = -log(abs(x))
        if lowerbound > 0 && isfinite(upperbound)
            ret -= log(log(upperbound) - log(lowerbound))
        end
        return ret
    else
        return -Inf
    end
end

# Very cute!
function log_jeffreys_crp_alpha(alpha::T, n::Int) where T

    return 1/2 * log((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

end

# Funky stuff happens sometimes where lambda
# shoots up to very large values when you have
# clusters with only 1 element and it screws
# up the computation of log_cluster_weight
# and logprobgenerative. This is a quick fix,
# I don't like it but it works. I don't think
# it should be happening.
function log_jeffreys_lambda(lambda::T) where T
    return log_logarithmic(lambda)
end

function log_jeffreys_psi(psi::AbstractMatrix{T}; loglowerbound=-size(psi, 1) * log(T(10^-3))) where T
    D = size(psi, 1)
    ret = -D * logdetpsd(psi)
    if isfinite(ret)
        return ret
    else
        return -Inf
    end
end
# function log_jeffreys_psi(psi::AbstractMatrix{T}; loglowerbound=-size(psi, 1) * log(T(10^-3))) where T
#     D = size(psi, 1)
#     ret = -D * logdetpsd(psi)
#     if ret >= loglowerbound
#         return ret
#     else
#         return -Inf
#     end
# end

# Very cute as well!
function log_jeffreys_nu(nu::T, d::Int; lowerbound=T(0), upperbound=T(1000)) where T
    if T(d-1) + lowerbound < nu <= upperbound
        return -log(2) + 1/2 * log(sum([polygamma(1, nu/2 + (1 - i)/2) for i in 1:d]))
    else
        return -Inf
    end

end

function log_nn_prior_normalinvgamma(nn_params::ComponentArray{T}, mu0::T, lambda0::T, alpha0::T, beta0::T; lastlayer=true, flat=true) where T

    if flat
        return zero(T)
    end

    lambda0, alpha0, beta0 = exp(lambda0), exp(alpha0), exp(beta0)

    if lastlayer
        weights = nn_params[keys(nn_params)[end]].weight
    else
        # All layers
        weights = reduce(vcat, [nn_params[layername].weight[:] for layername in keys(nn_params)])
    end

    W = sum(weights)
    WW = sum(weights.^2)
    n = length(weights)

    mu_u = (lambda0 * mu0 + W) / (lambda0 + n)
    lambda_u = lambda0 + n
    alpha_u = alpha0 + n / 2
    beta_u = beta0 + 1/2 * (WW + lambda0 * mu0^2 - lambda_u * mu_u^2)

    if beta_u <= beta0
        @warn "beta_u <= beta0, the posterior is improper over the alpha0 hyperparameter. This is a function of the data, you're simply unlucky. Proceed at your own risk, but it will eventually crash-and-burn." maxlog=1
    end

    # 1/Z0
    A0 = 1/2 * log(lambda0) + alpha0 * log(beta0) - loggamma(alpha0)

    # 1/Zu
    Au = 1/2 * log(lambda_u) + alpha_u * log(beta_u) - loggamma(alpha_u)

    return A0 - Au

end

function log_nn_hyperprior(_mu0::T, lambda0::T, alpha0::T, beta0::T; cauchy=true) where T

    if cauchy
        return sum(logpdf(Cauchy(), [_mu0, lambda0, alpha0, beta0]))
    end

    lambda0, alpha0, beta0 = exp(lambda0), exp(alpha0), exp(beta0)

    ret = zero(T) # flat over mu0
    ret += log_logarithmic(lambda0)
    ret += log_logarithmic(beta0)

    # If beta0 >= beta_u this will cause
    # the posterior to be improper over
    # the alpha0 hyperparameter
    ret += 1/2 * log(polygamma(1, alpha0))

    return ret

end

# log_nn_prior(args...; kwargs...) = log_nn_prior_normalinvgamma(args...; kwargs...)
log_nn_prior(args...; kwargs...) = 0
