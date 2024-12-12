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
    loglambda = log(abs(lambda))
    if loglambda > log(1000)
        return -Inf
    end
    return -loglambda
end

function log_jeffreys_psi(psi::AbstractMatrix{T}) where T
    D = size(psi, 1)
    return -D * logdetpsd(psi)
end

# Very cute as well!
function log_jeffreys_nu(nu::T, d::Int) where T

    return -log(2) + 1/2 * log(sum([polygamma(1, nu/2 + (1 - i)/2) for i in 1:d]))

end

# The Cauchy distribution integrated over the Jeffreys prior
# for its scale parameter leads to the symmetric
# logarithmic prior. Terrible mixing for some reason,
# much worse than the product of univariate t-distribrutions
# with very small alpha and large scale such that
# sqrt(alpha) * scale = 1. The two should in theory
# be roughtly equivalent but for some reason they are not.
# I blame maybe the singularity at zero.
function log_nn_prior_logarithmic(nn_params::ComponentArray{T}, alpha::T, scale::T) where {T}

    # return zero(T)

    weights = reduce(vcat, [nn_params[layername].weight[:] for layername in keys(nn_params)])

    any(iszero.(weights)) && return -Inf

    # return sum(-(1 + alpha)/2 * log.(1 .+ abs.(weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))
    return -sum(log.(abs.(weights)))

end

# Product of univariate Student t-distributions
function log_nn_prior_univariate_tdists(nn_params::ComponentArray{T}, alpha::T, scale::T) where {T}

    # return zero(T)

    # Stable t-distribution of index alpha on weights of last hidden layer.
    # (Neal - 1996 - Bayesian Learning for Neural Networks)
    # weights = nn_params[keys(nn_params)[end]].weight
    # return sum(-(1 + alpha)/2 * log.(1 .+ abs.(weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))

    # Stable t-distribution of index alpha on all weights.
    # When alpha=1 this becomes the Cauchy distribution
    weights = reduce(vcat, [nn_params[layername].weight[:] for layername in keys(nn_params)])

    # return sum(-(1 + alpha)/2 * log.(1 .+ abs.(weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))
    return sum(-(1 + alpha)/2 * log1pexp.(2 * log.(abs.(weights)) .- 2 * log(abs(scale)) .- log(alpha)) .- 1/2 * log(pi * alpha) .- log(scale) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))

end

# Isotropic multivariate Student t-distribution
function log_nn_prior_multivariate_tdist(nn_params::ComponentArray{T}, alpha::T, scale::T) where {T}

    # return zero(T)

    last_weights = nn_params[keys(nn_params)[end]].weight
    p = length(last_weights)

    return -(alpha + p)/2 * log(1 + 1/alpha * sum(last_weights.^2 ./ scale^2)) - p/2 * log(pi * alpha) - p * log(scale) - loggamma(alpha/2) + loggamma((p + alpha)/2)

end

# log_nn_prior(args...; kwargs...) = log_nn_prior_logarithmic(args...; kwargs...)
log_nn_prior(args...; kwargs...) = log_nn_prior_univariate_tdists(args...; kwargs...)

# Neat!
function log_jeffreys_t_alpha(alpha::T) where T
    # Otherwise weird stuff happens with polygamma
    if alpha < 10000
        return -log(2) + 1/2 * log(polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2) - 2 * (5 + alpha) / alpha / (alpha^2 + 4 * alpha  + 3))
    else
        return -Inf
    end
end

# Independence Jeffreys prior
# of scale parameter for scaled t-distribution
# alpha=3 by default so that alpha dependence
# goes to zero by default. It doesn't mean
# much because log_jeffreys_t_scale is used
# as P(scale) in the indenpendce prior
# P(scale, alpha) = P(scale)P(alpha)
function log_jeffreys_t_scale(scale::T, alpha::T=T(3)) where T
    return -log(abs(scale))
    # return -log(abs(scale)) + log(2 * alpha) - log(3 + alpha)
end

# Bivariate Jeffreys prior of
# product of univariate t-distribution
# with unique identical scale and degree.
function log_jeffreys_t(alpha::T, scale::T) where T
    # Otherwise weird stuff happens with polygamma
    if alpha < 10000# && scale < 10000
        # Just for fun the 2.0984 is the normalization 
        # constant of the alpha part of the Jeffreys
        # prior which is independent of the improper scale part
        return -log(2.0984) - log(abs(scale)) + 1/2 * log(alpha / 2 / (3 + alpha) * (polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2)) - 1 / (1 + alpha)^2)
    else
        return -Inf
    end
end

function log_jeffreys_nn(alpha::T, scale::T; independence=true) where T
    if independence
        return log_jeffreys_t_alpha(alpha) + log_jeffreys_t_scale(scale)
    else
        return log_jeffreys_t(alpha, scale)
    end
end