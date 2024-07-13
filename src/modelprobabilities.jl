# function logprobgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, base2original::Union{Nothing, Dict{Vector{Float64}, Vector{Float64}}}=nothing; hyperpriors=true, ffjord=false, temperature=1.0)

#     alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
#     nn, nn_params, nn_state, nn_alpha, nn_scale = hyperparams.nn, hyperparams.nn_params, hyperparams.nn_state, hyperparams.nn_alpha, hyperparams.nn_scale

#     logp = logprobgenerative(clusters, alpha, mu, lambda, psi, nu, base2original, nn, nn_params, nn_state, nn_alpha, nn_scale; hyperpriors=hyperpriors, ffjord=ffjorT, Demperature=temperature)

#     return logp
# end

# # Theta is assumed to be a concatenated vector of coordinates
# # i.e. vcat(log(alpha), mu, log(lambda), flatL, log(nu -d + 1))
# function logprobgenerative(clusters::Vector{Cluster}, theta::Vector{Float64}; hyperpriors=true, backtransform=true, jacobian=false, temperature=1.0)
#     alpha, mu, lambda, _flatL, L, psi, nu = pack(theta, backtransform=backtransform)
#     log_p = logprobgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors, temperature=temperature)
#     if jacobian
#         d = length(mu)
#         log_p += log(alpha)
#         log_p += log(lambda)
#         log_p += sum((d:-1:1) .* log.(abs.(diag(L))))
#         log_p += log(nu - d + 1)
#     end
#     return log_p
# end

# # Return the log-likelihood of the model
# function logprobgenerative(
#     clusters::Vector{Cluster},
#     alpha::Float64,
#     mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64,
#     base2original::Union{Nothing, Dict{Vector{Float64}, Vector{Float64}}}=nothing,
#     nn::Union{Nothing, Chain}=nothing,
#     nn_params::Union{Nothing, ComponentArray}=nothing,
#     nn_state::Union{Nothing, NamedTuple}=nothing,
#     nn_alpha::Float64=1.0, nn_scale::Float64=1.0;
#     hyperpriors=true, ffjord=false, temperature=1.0)

#     @assert all(length(c) > 0 for c in clusters)

#     N = sum([length(c) for c in clusters])
#     K = length(clusters)
#     d = length(mu)

#     if alpha <= 0.0 || lambda <= 0.0 || nu <= d - 1 || !isfinite(logdetpsd(psi)) || nn_alpha <= 0.0
#         return -Inf
#     end

#     # Log-probability associated with the Chinese Restaurant Process
#     log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum([loggamma(length(c)) for c in clusters])

#     # Log-probability associated with the data likelihood
#     # and Normal-Inverse-Wishart base distribution of the CRP
#     log_niw = 0.0
#     for cluster in clusters
#         log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - length(cluster) * d/2 * log(2pi)
#     end
#     log_niw -= K * log_Zniw(nothing, mu, lambda, psi, nu)

#     log_nn = 0.0
#     if ffjord && nn !== nothing
#         ffjord_model = FFJORD(nn, (0.0f0, 1.0f0), (d,), Tsit5(), ad=AutoForwardDiff())
#         origmat = reduce(hcat, values(base2original), init=zeros(Float64, d, 0))
#         ret, _ = ffjord_model(origmat, nn_params, nn_state)
#         log_nn -= sum(ret.delta_logp)

#         log_nn += nn_prior(nn, nn_params, nn_alpha, nn_scale)
#     end

#     log_hyperpriors = 0.0

#     if hyperpriors
#         # mu0 has a flat hyperpriors
#         # alpha hyperprior
#         log_hyperpriors += log(jeffreys_alpha(alpha, N))
#         # lambda hyperprior
#         log_hyperpriors += -log(lambda)
#         # psi hyperprior
#         log_hyperpriors += -d * logdetpsd(psi)
#         # log_hyperpriors += -d * logdet(psi)
#         # log_hyperpriors += -d * log(det(psi))
#         # nu hyperprior
#         log_hyperpriors += log(jeffreys_nu(nu, d))

#         if ffjord && nn !== nothing
#             # log_hyperpriors += log(jeffreys_t_alpha(nn_alpha))
#             # log_hyperpriors -= log(nn_scale)
#             log_hyperpriors += log_jeffreys_t(nn_alpha, nn_scale)
#         end
#     end

#     log_p = log_crp + log_niw + log_nn + log_hyperpriors

#     return isfinite(log_p) ? log_p / temperature : -Inf

# end

function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D}}, hyperparams::AbstractFCHyperparams{T, D}; ignorehyperpriors::Bool=false, ignoreffjord::Bool=false, temperature::T=one(T))::T where {T, D}
    return logprobgenerative(clusters, hyperparams._, ignorehyperpriors=ignorehyperpriors, ignoreffjord=ignoreffjord, temperature=temperature)
end

function logprobgenerative(clusters::AbstractVector{<:AbstractCluster{T, D}}, hyperparamsarray::ComponentArray{T}; ignorehyperpriors::Bool=false, ignoreffjord::Bool=false, temperature::T=one(T))::T where {T, D}

    # @assert all(length(c) > 0 for c in clusters)

    hpa = hyperparamsarray

    alpha, mu, lambda, psi, nu = hpa.pyp.alpha, hpa.niw.mu, hpa.niw.lambda, foldpsi(hpa.niw.flatL), hpa.niw.nu

    N = sum(length.(clusters))
    K = length(clusters)

    # psi is always valid by construction
    # except perhaps when logdet is too small/large
    if alpha <= 0 || lambda <= 0 || nu <= D - 1 || (hasnn(hpa) && hpa.nn.t.alpha <= 0 && hpa.nn.t.scale <= 0)
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
    log_niw -= K * log_Zniw(EmptyCluster{T, D}(), mu, lambda, psi, nu)

    log_nn = zero(T)

    if !ignoreffjord && hasnn(hpa)

        # ffjord_model = FFJORD(nn, (0.0, 1.0), (d,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
        # origmat = reduce(hcat, values(base2original), init=zeros(Float64, d, 0))
        # ret, _ = ffjord_model(origmat, hpa.nn.params, nn_state)
        
        # test later now that ffjord assumes a
        # flat improper base distribution when basedist=nothing
        # log_nn = sum(ret.logpx) 
        
        # if C isa SetCluster
            # origmat = Matrix(

        log_nn -= sum(forwardffjord(Matrix(clusters, orig=true), hpa).delta_logps)

        log_nn += nn_prior(hpa.nn.params, hpa.nn.t.alpha, hpa.nn.t.scale)
    end

    log_hyperpriors = zero(T)

    if !ignorehyperpriors
        # mu0 has a flat hyperpriors
        # alpha hyperprior
        log_hyperpriors += log(jeffreys_alpha(alpha, N))

        # NIW hyperpriors
        log_hyperpriors += -log(lambda)
        log_hyperpriors += -D * logdetpsd(psi)
        log_hyperpriors += log(jeffreys_nu(nu, D))

        if !ignoreffjord && hasnn(hpa)
            # Independence Jeffreys prior
            # log_hyperpriors += log(jeffreys_t_alpha(hpa.nn.t.alpha))
            # log_hyperpriors -= log(hpa.nn.t.scale)
            
            # Bivariate Jeffreys prior
            log_hyperpriors += log_jeffreys_t(hpa.nn.t.alpha, hpa.nn.t.scale)
        end
    end

    # println("$(round(log_crp, digits=4)) $(round(log_niw, digits=4)) $(round(log_nn, digits=4)) $(round(log_hyperpriors, digits=4))")
    log_p = log_crp + log_niw + log_nn + log_hyperpriors

    return isfinite(log_p) ? log_p / temperature : -Inf

end

# function logprobgenerative(
#     clusters::Vector{Cluster},
#     hyperparamsarray::ComponentArray{Float64},
#     base2original::Dict{Vector{Float64}, Vector{Float64}},
#     hyperpriors=true, ffjord=false, temperature=1.0)
#     return logprobgenerative(clusters, hyperparamsarray, base2original, nothing, nothing; hyperpriors=hyperpriors, ffjord=ffjorT, Demperature=temperature)
# end


# function logprobgenerative(clusters::Vector{Cluster},
#     hyperparamsarray::ComponentArray{Float64},
#     hyperpriors=true, temperature=1.0)

#     hpa = hyperparamsarray

#     alpha, mu, lambda, psi, nu = hpa.pyp.alpha, hpa.niw.mu, hpa.niw.lambda, foldpsi(hpa.niw.flatL), hpa.niw.nu

#     N = sum([length(c) for c in clusters])
#     K = length(clusters)
#     d = size(hpa.niw.mu, 1)

#     if alpha <= 0.0 || lambda <= 0.0 || nu <= d - 1 || !isfinite(logdetpsd(psi))
#         return -Inf
#     end

#     # Log-probability associated with the Chinese Restaurant Process
#     log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum([loggamma(length(c)) for c in clusters])

#     # Log-probability associated with the data likelihood
#     # and Normal-Inverse-Wishart base distribution of the CRP
#     log_niw = 0.0
#     for cluster in clusters
#         log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - length(cluster) * d/2 * log(2pi)
#     end
#     log_niw -= K * log_Zniw(nothing, mu, lambda, psi, nu)

#     log_nn = 0.0

#     log_hyperpriors = 0.0

#     if hyperpriors
#         # mu0 has a flat hyperpriors
#         # alpha hyperprior
#         log_hyperpriors += log(jeffreys_alpha(alpha, N))

#         # NIW hyperpriors
#         log_hyperpriors += -log(lambda)
#         log_hyperpriors += -d * logdetpsd(psi)
#         log_hyperpriors += log(jeffreys_nu(nu, d))

#     end

#     log_p = log_crp + log_niw + log_hyperpriors

#     return isfinite(log_p) ? log_p / temperature : -Inf
# end
