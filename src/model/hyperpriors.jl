# Very cute!
function log_jeffreys_crp_alpha(alpha::T, n::Int) where T

    return 1/2 * log((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

end

# Very cute as well!
function log_jeffreys_nu(nu::T, d::Int) where T

    return -log(2) + 1/2 * log(sum([polygamma(1, nu/2 + (1 - i)/2) for i in 1:d]))

end

function log_nn_prior(nn_params::ComponentArray{T}, alpha::T, scale::T) where {T}
    return zero(T)
    
    # Stable t-distribution of index alpha on weights of last hidden layer.
    # (Neal - 1996 - Bayesian Learning for Neural Networks)

    last_weights = nn_params[keys(nn_params)[end]].weight

    return sum(-(1 + alpha)/2 * log.(1 .+ abs.(last_weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))

end

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
    return -log(abs(scale)) + log(2 * alpha) - log(3 + alpha)
end

# Bivariate Jeffreys prior of scaled t-distribution, how neat is that!
function log_jeffreys_t(alpha::T, scale::T) where T
    # Otherwise weird stuff happens with polygamma
    if alpha < 10000
        # Just for fun the 2.0984 is the normalization 
        # constant of the alpha part of the Jeffreys
        # prior which is independent of the improper scale part
        return -log(2.0984) - log(abs(scale)) + 1/2 * log(alpha / 2 / (3 + alpha) * (polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2)) - 1 / (1 + alpha)^2)
    else
        return -Inf
    end
end

function log_jeffreys_nn(alpha::T, scale::T; independence=false) where T
    if independence
        return log_jeffreys_t_alpha(alpha) + log_jeffreys_t_scale(scale)
    else
        return log_jeffreys_t(alpha, scale)
    end
end