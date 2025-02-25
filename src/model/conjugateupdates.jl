function updated_niw_hyperparams(cluster::AbstractCluster{T, D, E}, hyperparams::AbstractFCHyperparams{T, D})::@NamedTuple{mu_c::Vector{T}, lambda_c::T, psi_c::Matrix{T}, nu_c::T}  where {T, D, E}
    return updated_niw_hyperparams(cluster, hyperparams._.crp.niw.mu, hyperparams._.crp.niw.lambda, foldpsi(hyperparams._.crp.niw.flatL), hyperparams._.crp.niw.nu)
end

function updated_niw_hyperparams(::EmptyCluster{T, D, E}, mu::AbstractVector{T}, lambda::T, psi::Matrix{T}, nu::T)::@NamedTuple{mu_c::Vector{T}, lambda_c::T, psi_c::Matrix{T}, nu_c::T}  where {T, D, E}
    return (;mu_c=mu, lambda_c=lambda, psi_c=psi, nu_c=nu)
end

function updated_niw_hyperparams(cluster::AbstractCluster{T, D, E}, mu::AbstractVector{T}, lambda::T, psi::Matrix{T}, nu::T)::@NamedTuple{mu_c::Vector{T}, lambda_c::T, psi_c::Matrix{T}, nu_c::T} where {T, D, E}

    if isempty(cluster)
        return (;mu_c=mu, lambda_c=lambda, psi_c=psi, nu_c=nu)
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
function log_Zniw(cluster::AbstractCluster{T, D, E}, mu::AbstractVector{T}, lambda::T, psi::AbstractMatrix{T}, nu::T)::T where {T, D, E}

    mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

    log_numerator = D/2 * log(2*T(pi)) + nu * D/2 * log(T(2)) + logmvgamma(D, nu/2)

    log_denominator = D/2 * log(lambda) + nu/2 * logdetpsd(psi)

    return log_numerator - log_denominator

end


function log_cluster_weight(element::E, cluster::AbstractCluster{T, D, E}, alpha::T, mu::AbstractVector{T}, lambda::T, psi::AbstractMatrix{T}, nu::T; N::Int=-1)::T where {T, D, E}

    !(element in cluster) || error(KeyError("Duplicate $(element)"))

    if isempty(cluster)
        log_weight = log(T(alpha)) # - log(alpha + Nminus1)
    else
        log_weight = log(T(length(cluster))) # - log(alpha + Nminus1)
    end

    # Not really useful because weights are normalized
    # during Gibbs samples
    if N >= 0
        log_weight -= log(T(alpha + N))
    end

    push!(cluster, element)
    log_weight += log_Zniw(cluster, mu, lambda, psi, nu)
    pop!(cluster, element)
    log_weight -= log_Zniw(cluster, mu, lambda, psi, nu)
    log_weight -= D/2 * log(2*T(pi))

    return log_weight

end


function updated_mvstudent_params(
    ::EmptyCluster{T, D, E},
    mu::AbstractVector{T},
    lambda::T,
    psi::AbstractMatrix{T},
    nu::T
    )::@NamedTuple{df_c::T, mu_c::Vector{T}, sigma_c::Matrix{T}} where {T, D, E}

    # d = length(mu)

    return (df_c=nu - D + 1, mu_c=Vector{T}(mu), sigma_c=Matrix{T}((lambda + 1)/lambda/(nu - D + 1) * psi))

end

function updated_mvstudent_params(
    cluster::AbstractCluster{T, D, E},
    mu::AbstractVector{T},
    lambda::T,
    psi::AbstractMatrix{T},
    nu::T
    )::@NamedTuple{df_c::T, mu_c::Vector{T}, sigma_c::Matrix{T}} where {T, D, E}

    mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

    return (df_c=nu_c - D + 1, mu_c=mu_c, sigma_c=(lambda_c + 1)/lambda_c/(nu_c - D + 1) * psi_c)

end

function updated_mvstudent_params(
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    mu::AbstractVector{T},
    lambda::T,
    psi::AbstractMatrix{T},
    nu::T;
    add_empty=true
    )::Vector{@NamedTuple{df_c::T, mu_c::Vector{T}, sigma_c::Matrix{T}}} where {T, D, E}

    updated_mvstudent_degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]
    if add_empty
        push!(updated_mvstudent_degs_mus_sigs, updated_mvstudent_params(EmptyCluster{T, D, E}(), mu, lambda, psi, nu))
    end

    return updated_mvstudent_degs_mus_sigs
end

function updated_mvstudent_params(clusters::AbstractVector{<:AbstractCluster}, hyperparams::AbstractFCHyperparams; add_empty=true)
    return updated_mvstudent_params(clusters, hyperparams._.crp.niw.mu, hyperparams._.crp.niw.lambda, foldpsi(hyperparams._.crp.niw.flatL), hyperparams._.crp.niw.nu, add_empty=add_empty)
end
