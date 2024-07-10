# Very cute!
function jeffreys_alpha(alpha::Float64, n::Int64)

    return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

end

# Very cute as well!
function jeffreys_nu(nu::Float64, d::Int64)

    return 1/2 * sqrt(sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

end

function nn_prior(nn_params::AbstractArray{Float64}, alpha::Float64=1.0, scale::Float64=1.0)

    # Stable t-distribution of index alpha on weights of last hidden layer.
    # (Neal - 1996 - Bayesian Learning for Neural Networks)

    # scale *= nn.layers[end].in_dims
    last_weights = nn_params[keys(nn_params)[end]].weight

    return sum(-(1 + alpha)/2 * log.(1 .+ abs.(last_weights ./ scale).^2 ./ alpha) .- 1/2 * log(pi * alpha * scale^2) .- loggamma(alpha/2) .+ loggamma((1 + alpha)/2))

end

function jeffreys_t_alpha(alpha::Float64)
    # Otherwise weird stuff happens in nn_prior
    if alpha < 10000.0
        # return 1/2 * sqrt(polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2) - (5 + alpha) / 2 / alpha * exp(loggamma((1 + alpha) / 2) - loggamma((5 + alpha) / 2)))
        return 1/2 * sqrt(polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2) - 2 * (5 + alpha) / alpha / (alpha^2 + 4 * alpha  + 3))
    else
        return 0.0
    end
end

function log_jeffreys_t_scale(scale::Float64)
    return -log(abs(scale))
end

# Bivariate Jeffreys prior of scaled t-distribution
function log_jeffreys_t(alpha::Float64, scale::Float64)
    # Otherwise weird stuff happens in nn_prior
    if alpha < 10000.0
        return -log(abs(scale)) + 1/2 * log(alpha / 2 / (3 + alpha) * (polygamma(1, alpha / 2) - polygamma(1, (1 + alpha) / 2)) - 1 / (1 + alpha)^2)
    else
        return 0.0
    end
end