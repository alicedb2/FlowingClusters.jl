function updated_niw_hyperparams(clusters::Cluster, hyperparams::AbstractFCHyperparams)::Tuple{AbstractVector{Float64}, Float64, AbstractMatrix{Float64}, Float64}
    return updated_niw_hyperparams(clusters, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu)
end

function updated_niw_hyperparams(
    ::Nothing,
    mu::AbstractVector{Float64},
    lambda::Float64,
    psi::AbstractMatrix{Float64},
    nu::Float64
    )

    return (mu, lambda, psi, nu)

end

function updated_niw_hyperparams(cluster::Cluster,
    mu::AbstractVector{Float64},
    lambda::Float64,
    psi::AbstractMatrix{Float64},
    nu::Float64
    )

    if isempty(cluster)
        return (mu, lambda, psi, nu)
    else

        d = length(mu)
        n = length(cluster)

        lambda_c = lambda + n
        nu_c = nu + n

        # mu_c = Array{Float64}(undef, d)
        # psi_c = Array{Float64}(undef, d, d)
        mu_c = cluster.mu_c_volatile
        psi_c = cluster.psi_c_volatile

        @inbounds for i in 1:d
            mu_c[i] = (lambda * mu[i] + cluster.sum_x[i]) / (lambda + n)
        end

        @inbounds for j in 1:d
            @inbounds for i in 1:j
                psi_c[i, j] = psi[i, j]
                psi_c[i, j] += cluster.sum_xx[i, j]
                psi_c[i, j] += lambda * mu[i] * mu[j]
                psi_c[i, j] -= lambda_c * mu_c[i] * mu_c[j]
                psi_c[j, i] = psi_c[i, j]
            end
        end

        return (mu_c, lambda_c, psi_c, nu_c)

    end

end

function log_Zniw(
    ::Nothing,
    mu::AbstractVector{Float64},
    lambda::Float64,
    psi::AbstractMatrix{Float64},
    nu::Float64)::Float64

    d = length(mu)
    log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi)
    return log_numerator - log_denominator

end

function log_Zniw(
    ::Nothing,
    mu::AbstractVector{Float64},
    lambda::Float64,
    L::LowerTriangular{Float64},
    nu::Float64)

    d = length(mu)
    log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    log_denominator = d/2 * log(lambda) + nu * sum(log.(diag(L)))
    return log_numerator - log_denominator

end


# Return the normalization constant
# of the normal-inverse-Wishart distribution,
# possibly in the presence of data if cluster isn't empty
function log_Zniw(
    cluster::Cluster,
    mu::AbstractVector{Float64},
    lambda::Float64,
    psi::AbstractMatrix{Float64},
    nu::Float64)::Float64

    d = length(mu)

    mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

    log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)

    log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi)

    return log_numerator - log_denominator

end

function log_cluster_weight(element::Vector{Float64}, cluster::Cluster, alpha::Float64, mu::AbstractVector{Float64}, lambda::Float64, psi::AbstractMatrix{Float64}, nu::Float64; N::Union{Int64, Nothing}=nothing)

    @assert !(element in cluster) "$(element)"

    d = length(mu)

    if isempty(cluster)
        log_weight = log(alpha) # - log(alpha + Nminus1)
    else
        log_weight = log(length(cluster)) # - log(alpha + Nminus1)
    end

    if N !== nothing
        log_weight -= log(alpha + N)
    end

    push!(cluster, element)
    log_weight += log_Zniw(cluster, mu, lambda, psi, nu)
    pop!(cluster, element)
    log_weight -= log_Zniw(cluster, mu, lambda, psi, nu)
    log_weight -= d/2 * log(2pi)

    return log_weight

end