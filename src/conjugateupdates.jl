function updated_niw_hyperparams(cluster::AbstractCluster{T, D}, hyperparams::AbstractFCHyperparams{T, D}) where {T, D}
    return updated_niw_hyperparams(cluster, hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu)
end

function updated_niw_hyperparams(::EmptyCluster{T, D}, mu::Vector{T}, lambda::T, psi::Matrix{T}, nu::T) where {T, D}
    return (;mu_c=mu, lambda_c=lambda, psi_c=psi, nu_c=nu)
end

function updated_niw_hyperparams(cluster::AbstractCluster{T, D}, mu::Vector{T}, lambda::T, psi::Matrix{T}, nu::T)::@NamedTuple{mu_c::Vector{T}, lambda_c::T, psi_c::Matrix{T}, nu_c::T} where {T, D}

    if isempty(cluster)
        return mu, lambda, psi, nu
    else

        n = length(cluster)

        lambda_c = lambda + n
        nu_c = nu + n

        # mu_c = Array{Float64}(undef, D)
        # psi_c = Array{Float64}(undef, D, D)
        mu_c = cluster.mu_c_volatile
        psi_c = cluster.psi_c_volatile

        @inbounds for i in 1:D
            mu_c[i] = (lambda * mu[i] + cluster.sum_x[i]) / (lambda + n)
        end

        @inbounds for j in 1:D
            @inbounds for i in 1:j
                psi_c[i, j] = psi[i, j]
                psi_c[i, j] += cluster.sum_xx[i, j]
                psi_c[i, j] += lambda * mu[i] * mu[j]
                psi_c[i, j] -= lambda_c * mu_c[i] * mu_c[j]
                psi_c[j, i] = psi_c[i, j]
            end
        end

        return (;mu_c, lambda_c, psi_c, nu_c)

    end

end

# Return the normalization constant
# of the normal-inverse-Wishart distribution,
# possibly in the presence of data if cluster isn't empty
function log_Zniw(cluster::AbstractCluster{T, D}, mu::Vector{T}, lambda::T, psi::Matrix{T}, nu::T)::T where {T, D}

    mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

    log_numerator = D/2 * log(2*T(pi)) + nu * D/2 * log(T(2)) + logmvgamma(D, nu/2)

    log_denominator = D/2 * log(lambda) + nu/2 * logdetpsd(psi)

    return log_numerator - log_denominator

end

function log_cluster_weight(element::AbstractVector{T}, cluster::AbstractCluster{T, D}, alpha::T, mu::Vector{T}, lambda::T, psi::Matrix{T}, nu::T, N::Int=-1)::T where {T, D}

    !(element in cluster) || error(KeyError("Duplicate $(element)"))

    if isempty(cluster)
        log_weight = log(alpha) # - log(alpha + Nminus1)
    else
        log_weight = log(T(length(cluster))) # - log(alpha + Nminus1)
    end

    # Not really useful because weights are normalized
    # during Gibbs samples
    if N >= 0
        log_weight -= log(alpha + N)
    end

    push!(cluster, element)
    log_weight += log_Zniw(cluster, mu, lambda, psi, nu)
    pop!(cluster, element)
    log_weight -= log_Zniw(cluster, mu, lambda, psi, nu)
    log_weight -= D/2 * log(2*T(pi))

    return log_weight

end