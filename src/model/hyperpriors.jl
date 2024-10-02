# Very cute!
function jeffreys_crp_alpha(alpha::T, n::Int) where T

    return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

end

# Very cute as well!
function jeffreys_nu(nu::T, d::Int) where T

    return 1/2 * sqrt(sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

end

function nn_prior(nn_params::ComponentArray{T}, alpha::T=one(T), scale::T=one(T)) where T

    # Stable t-distribution of index alpha on weights of last hidden layer.
    # (Neal - 1996 - Bayesian Learning for Neural Networks)

    # scale *= nn.layers[end].in_dims
    last_weights = nn_params[keys(nn_params)[end]].weight

    return sum(-(1 + alpha)/2 * log.(1 .+ abs.(last_weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))

end

# Neat!
function jeffreys_t_alpha(alpha::T) where T
    # Otherwise weird stuff happens in nn_prior
    if alpha < 10000
        return 1/2 * sqrt(polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2) - 2 * (5 + alpha) / alpha / (alpha^2 + 4 * alpha  + 3))
    else
        return zero(T)
    end
end

# Independence Jeffreys prior
# of scale parameter for scaled t-distribution
function log_jeffreys_t_scale(scale::T, alpha::T=one(T)) where T
    return -log(abs(scale)) + log(2 * alpha) - log(3 + alpha)
end

# Bivariate Jeffreys prior of scaled t-distribution, how neat is that!
function log_jeffreys_t(alpha::T, scale::T) where T
    # Otherwise weird stuff happens in nn_prior
    if alpha < 10000
        return -log(abs(scale)) + 1/2 * log(alpha / 2 / (3 + alpha) * (polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2)) - 1 / (1 + alpha)^2)
    else
        return T(-Inf)
    end
end