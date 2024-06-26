export MNDPVariational
export advance_variational!, advance_variational_stochastic!
export advance_gamma!, advance_eta!, advance_phi!
export randomize_phi!, reset_gamma!, randomize_eta!, heat_phi!
export mu_k, lambda_k, psi_k, nu_k, naturalize, canonize
export variational_objective
export draw_partition
export draw_Q, log_Q, log_Ptrunc

mutable struct MNDPVariational
    data::Matrix{Float64}
    
    d::Int64
    T::Int64
    N::Int64
    current_iteration::Int64

    hyperparams::MNCRPHyperparams

    s1::Float64
    s2::Float64

    w1::Float64
    w2::Float64
    
    logphi_nk::Matrix{Float64}
    gamma1_k::Vector{Float64}
    gamma2_k::Vector{Float64}

    eta1_k::Matrix{Float64}
    eta2_k::Vector{Float64}
    eta3_k::Array{Float64, 3}
    eta4_k::Vector{Float64}
end

function Base.show(io::IO, ::MIME"text/plain", var::MNDPVariational)
    print("MNDPVariational(N=$(var.N), d=$(var.d), T=$(var.T))")
end

function MNDPVariational(data::Vector{Vector{Float64}}, T::Int64; randomize_phi=true)
    data = copy(reduce(hcat, data)')
    N, d = size(data)

    hyperparams = MNCRPHyperparams(d)
    hyperparams.alpha = 100.0 / log(length(data))
    
    s1, s2 = 0.5, 1.0
    w1, w2 = 0.5, 1.0

    if randomize_phi
        logphi_nk = log.(copy(rand(Dirichlet(ones(T)), N)'))
    else
        logphi_nk = fill(-log(T), N, T)
    end

    gamma1_k = fill(1.0, T-1)
    gamma2_k = fill(hyperparams.alpha, T-1)

    eta1_k = permutedims(repeat(hyperparams.lambda * hyperparams.mu, 1, T), (2, 1))
    eta2_k = fill(hyperparams.lambda, T)
    eta3_k = permutedims(repeat(hyperparams.psi + hyperparams.lambda * hyperparams.mu * hyperparams.mu', 1, 1, T), (3, 1, 2))
    eta4_k = fill(hyperparams.nu, T)

    return MNDPVariational(data, d, T, N, 0, hyperparams, s1, s2, w1, w2, logphi_nk, gamma1_k, gamma2_k, eta1_k, eta2_k, eta3_k, eta4_k)
end

function MNDPVariational(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; T=nothing)
    
    hyperparams = deepcopy(hyperparams)

    data = copy(reduce(hcat, elements(clusters))')

    N, d = size(data)
    if T === nothing || T < 2 * length(clusters)
        T = 2 * length(clusters)
    end
    deltaT = T - length(clusters)

    clusters_naturals = map(x -> naturalize(x...), updated_niw_hyperparams.(clusters, Ref(hyperparams)))

    eta1_k = copy(cat(getindex.(clusters_naturals, 1)..., 
                      [copy(hyperparams.lambda * hyperparams.mu) for _ in 1:deltaT]..., dims=2)')

    eta2_k = cat(getindex.(clusters_naturals, 2)..., 
                 [hyperparams.lambda for _ in 1:deltaT]..., dims=1)

    eta3_k = permutedims(cat(getindex.(clusters_naturals, 3)..., 
                             [hyperparams.psi + hyperparams.lambda * hyperparams.mu * hyperparams.mu' for _ in 1:deltaT]..., dims=3), [3, 1, 2])
                             
    eta4_k = cat(getindex.(clusters_naturals, 4)..., 
                 [hyperparams.nu for _ in 1:deltaT]..., dims=1)

    gamma1_k = 1.0 .+ length.(clusters)
    gamma1_k = vcat(gamma1_k, fill(1.0, deltaT - 1))

    gamma2_k = hyperparams.alpha .+ reverse(cumsum(reverse(length.(clusters)))) .- length.(clusters)
    gamma2_k = vcat(gamma2_k, fill(hyperparams.alpha, deltaT - 1))

    s1, s2 = 0.5, 0.01
    w1, w2 = 0.5, 0.01

    logphi_nk = fill(-log(T), N, T)
    
    mndp = MNDPVariational(data, d, T, N, 0, hyperparams, s1, s2, w1, w2, logphi_nk, gamma1_k, gamma2_k, eta1_k, eta2_k, eta3_k, eta4_k)
    
    advance_phi!(mndp, nothing, 1.0)

    return mndp

end

function randomize_phi!(mndp::MNDPVariational)
    mndp.logphi_nk = log.(copy(rand(Dirichlet(ones(mndp.T)), mndp.N)'))
    return mndp.logphi_nk
end

function heat_phi!(mndp::MNDPVariational, temperature::Float64=1.0)
    for n in 1:mndp.N
        mndp.logphi_nk[n, :] = mndp.logphi_nk[n, :] / temperature .- logsumexp(mndp.logphi_nk[n, :] / temperature)
    end
    return mndp
end

function reset_gamma!(mndp::MNDPVariational)
    mndp.gamma1_k = fill(1.0, mndp.T-1)
    mndp.gamma2_k = fill(mndp.hyperparams.alpha, mndp.T-1)
    return mndp
end

function randomize_eta!(mndp::MNDPVariational)
    mndp.eta2_k = rand(mndp.T)
    mndp.eta1_k = mndp.eta2_k .* rand(mndp.T, mndp.d)
    mndp.eta3_k = permutedims(cat(((x, y, z) -> LowerTriangular(x) * LowerTriangular(x)' .+ 1/y * z * z' ).([rand(mndp.d, mndp.d) for _ in 1:mndp.T], mndp.eta2_k, eachslice(mndp.eta1_k, dims=1))..., dims=3), [3, 1, 2])
    mndp.eta4_k = mndp.d .- 1.0 .+ 3.0 * rand(mndp.T)
    return mndp
end

function advance_variational!(mndp::MNDPVariational, nb_steps::Int64)

    for step in 1:nb_steps
        print("\r$step/$nb_steps")
        advance_hyper_w!(mndp)
        advance_gamma!(mndp)
        advance_eta!(mndp)
        advance_phi!(mndp, nothing, 1.0)
    end    
    println()

    return mndp
end


function advance_variational_stochastic!(mndp::MNDPVariational, nb_steps::Int64; minibatch_size::Int64=1, kappa::Float64=1.0, tau::Float64=1.0)

    # random_order = shuffle(1:mndp.N)
    for step in 1:nb_steps
        print("\r$step/$nb_steps")

        rho = (mndp.current_iteration + step + tau)^-kappa

        print("   (iter $(mndp.current_iteration), rho=$(round(rho, digits=10)))")
        
        minibatch = sample(1:mndp.N, minibatch_size, replace=false)
        # minibatch = [random_order[mod(step - 1, mndp.N) + 1]]
        # advance_hyper_w!(mndp, rho)
        advance_gamma!(mndp, minibatch, rho)
        advance_eta!(mndp, minibatch, rho)
        advance_phi!(mndp, minibatch, 1.0)

        # if mod(step, mndp.N) == 0
        #     shuffle!(random_order)
        # end

        mndp.current_iteration += 1

    end    
    println()

    return mndp
end

function advance_hyper_w!(mndp::MNDPVariational)::MNDPVariational
    mndp.w1 = mndp.s1 + mndp.T - 1
    mndp.w2 = mndp.s2 - sum([polygamma(0, mndp.gamma2_k[k]) - polygamma(0, mndp.gamma1_k[k] + mndp.gamma2_k[k]) for k in 1:mndp.T-1])
    return mndp
end

function advance_hyper_w!(mndp::MNDPVariational, rho::Float64)::MNDPVariational
    intermediate_w1 = mndp.s1 + mndp.T - 1
    intermediate_w2 = mndp.s2 - sum([polygamma(0, mndp.gamma2_k[k]) - polygamma(0, mndp.gamma1_k[k] + mndp.gamma2_k[k]) for k in 1:mndp.T-1])
    mndp.w1 = (1.0 - rho) * mndp.w1 + rho * intermediate_w1
    mndp.w2 = (1.0 - rho) * mndp.w2 + rho * intermediate_w2
    return mndp
end

function advance_gamma!(mndp::MNDPVariational)::MNDPVariational
    
    for k in 1:mndp.T-1
        mndp.gamma1_k[k] = 1.0
        for n in 1:mndp.N
            mndp.gamma1_k[k] += exp(mndp.logphi_nk[n, k])
        end
    end

    for k in 1:mndp.T-1
        mndp.gamma2_k[k] = mndp.w1/mndp.w2
        # mndp.gamma2_k[k] = mndp.hyperparams.alpha
        for j in k+1:mndp.T
            for n in 1:mndp.N
                mndp.gamma2_k[k] += exp(mndp.logphi_nk[n, j])
            end
        end
    end

    return mndp
end

function advance_gamma!(mndp::MNDPVariational, minibatch::Int64, rho::Float64)::MNDPVariational
    return advance_gamma!(mndp::MNDPVariational, Int64[minibatch], rho)
end

function advance_gamma!(mndp::MNDPVariational, minibatch::Vector{Int64}, rho::Float64)::MNDPVariational
    
    for k in 1:mndp.T-1
        intermediate_sum_minibatch = 0.0
        for s in minibatch
            intermediate_sum_minibatch += 1.0 + mndp.N * exp(mndp.logphi_nk[s, k])
        end
        mndp.gamma1_k[k] = (1.0 - rho) * mndp.gamma1_k[k] + rho * intermediate_sum_minibatch / length(minibatch)
    end

    for k in 1:mndp.T-1
        intermediate_sum_minibatch = 0.0
        for s in minibatch
            intermediate_sum_minibatch += mndp.w1/mndp.w2
            # intermediate_sum_minibatch += mndp.hyperparams.alpha
            for j in k+1:mndp.T
                intermediate_sum_minibatch += mndp.N * exp(mndp.logphi_nk[s, j])
            end
        end
        mndp.gamma2_k[k] = (1.0 - rho) * mndp.gamma2_k[k] + rho * intermediate_sum_minibatch / length(minibatch)
    end

    return mndp
end

function advance_eta!(mndp::MNDPVariational)::MNDPVariational
    
    for i in 1:mndp.d
        for k in 1:mndp.T            
            mndp.eta1_k[k, i] = mndp.hyperparams.lambda * mndp.hyperparams.mu[i]
            for n in 1:mndp.N
                mndp.eta1_k[k, i] += exp(mndp.logphi_nk[n, k]) * mndp.data[n, i]
            end
        end
    end

    for k in 1:mndp.T
        mndp.eta2_k[k] = mndp.hyperparams.lambda
        for n in 1:mndp.N
            mndp.eta2_k[k] += exp(mndp.logphi_nk[n, k])
        end
    end

    for j in 1:mndp.d
        for i in 1:j
            for k in 1:mndp.T
                mndp.eta3_k[k, i, j] = mndp.hyperparams.psi[i, j] + mndp.hyperparams.lambda * mndp.hyperparams.mu[i] * mndp.hyperparams.mu[j]
                
                for n in 1:mndp.N
                    mndp.eta3_k[k, i, j] += exp(mndp.logphi_nk[n, k]) * mndp.data[n, i] * mndp.data[n, j]
                end

                mndp.eta3_k[k, j, i] = mndp.eta3_k[k, i, j]
            end
        end
    end

    for k in 1:mndp.T
        mndp.eta4_k[k] = mndp.hyperparams.nu
        for n in 1:mndp.N
            mndp.eta4_k[k] += exp(mndp.logphi_nk[n, k])
        end
    end

    return mndp
end

function advance_eta!(mndp::MNDPVariational, minibatch::Vector{Int64}, rho::Float64)::MNDPVariational
    
    for i in 1:mndp.d
        for k in 1:mndp.T
            intermediate_sum_minibatch = 0.0
            for s in minibatch
                intermediate_sum_minibatch += mndp.hyperparams.lambda * mndp.hyperparams.mu[i]
                intermediate_sum_minibatch += mndp.N * exp(mndp.logphi_nk[s, k]) * mndp.data[s, i]
            end
            mndp.eta1_k[k, i] = (1.0 - rho) * mndp.eta1_k[k, i] + rho * intermediate_sum_minibatch / length(minibatch)
        end
    end

    for k in 1:mndp.T
        intermediate_sum_minibatch = 0.0
        for s in minibatch
            intermediate_sum_minibatch += mndp.hyperparams.lambda + mndp.N * exp(mndp.logphi_nk[s, k])
        end
        mndp.eta2_k[k] = (1.0 - rho) * mndp.eta2_k[k] + rho * intermediate_sum_minibatch / length(minibatch)
    end

    for j in 1:mndp.d
        for i in 1:j
            for k in 1:mndp.T
                intermediate_sum_minibatch = 0.0
                for s in minibatch
                    intermediate_sum_minibatch += mndp.hyperparams.psi[i, j] + mndp.hyperparams.lambda * mndp.hyperparams.mu[i] * mndp.hyperparams.mu[j]
                    intermediate_sum_minibatch += mndp.N * exp(mndp.logphi_nk[s, k]) * mndp.data[s, i] * mndp.data[s, j]
                end

                mndp.eta3_k[k, i, j] = (1.0 - rho) * mndp.eta3_k[k, i, j] + rho * intermediate_sum_minibatch / length(minibatch)

                mndp.eta3_k[k, j, i] = mndp.eta3_k[k, i, j]
            end
        end
    end

    for k in 1:mndp.T
        intermediate_sum_minibatch = 0.0
        for s in minibatch
            intermediate_sum_minibatch += mndp.hyperparams.nu + mndp.N * exp(mndp.logphi_nk[s, k])
        end
        mndp.eta4_k[k] = (1.0 - rho) * mndp.eta4_k[k] + rho * intermediate_sum_minibatch / length(minibatch)
    end

    return mndp
end

function advance_phi!(mndp::MNDPVariational, minibatch::Union{Nothing, Int64, Vector{Int64}}, rho::Float64)::MNDPVariational

    if minibatch === nothing
        n_subset = shuffle(1:mndp.N)
    elseif minibatch isa Int64
        n_subset = minibatch:minibatch
    elseif minibatch isa Vector{Int64}
        n_subset = minibatch
    end


    intermediate_logphi = Array{Float64}(undef, length(n_subset), mndp.T)

    for k in 1:mndp.T
        logvk_contrib = k == mndp.T ? 0.0 : polygamma(0, mndp.gamma1_k[k]) - polygamma(0, mndp.gamma1_k[k] + mndp.gamma2_k[k])
        log1mvk_contrib = sum([polygamma(0, mndp.gamma2_k[j]) - polygamma(0, mndp.gamma1_k[j] + mndp.gamma2_k[j]) for j in 1:k-1])

        psi_k = @views mndp.eta3_k[k, :, :] - 1.0 / mndp.eta2_k[k] * mndp.eta1_k[k, :] * mndp.eta1_k[k, :]'
        logdetsigma_contrib = @views mndp.d/2 * log(2) - 1/2 * logdetpsd(psi_k) + 1/2 * sum(polygamma(0, mndp.eta4_k[k]/2 + (1-j)/2) for j in 1:mndp.d)
        musigmamu_contrib = @views -mndp.d / 2 / mndp.eta2_k[k] - mndp.eta4_k[k] / 2 / mndp.eta2_k[k]^2 * mndp.eta1_k[k, :]' * (psi_k \ mndp.eta1_k[k, :])
        
        for n in 1:length(n_subset)
            intermediate_logphi[n, k] = (logvk_contrib + log1mvk_contrib)
            intermediate_logphi[n, k] += logdetsigma_contrib + musigmamu_contrib
            
            invpsi_xn = psi_k \ mndp.data[n_subset[n], :]
            
            # inv(sigma) contribution
            intermediate_logphi[n, k] += @views (-1/2 * mndp.eta4_k[k] * mndp.data[n_subset[n], :]' * invpsi_xn)
            # mu' * inv(sigma) contribution
            intermediate_logphi[n, k] += @views (mndp.eta4_k[k] / mndp.eta2_k[k] * mndp.eta1_k[k, :]' * invpsi_xn)
            intermediate_logphi[n, k] -= mndp.d/2 * log(2pi)


            # mndp.logphi_nk[n, k] = (logvk_contrib + log1mvk_contrib)
            # mndp.logphi_nk[n, k] += logdetsigma_contrib + musigmamu_contrib
            
            # invpsi_xn = psi_k \ mndp.data[n, :]
            
            # # inv(sigma) contribution
            # mndp.logphi_nk[n, k] += @views (-1/2 * mndp.eta4_k[k] * mndp.data[n, :]' * invpsi_xn)
            # # mu' * inv(sigma) contribution
            # mndp.logphi_nk[n, k] += @views (mndp.eta4_k[k] / mndp.eta2_k[k] * mndp.eta1_k[k, :]' * invpsi_xn)
            # mndp.logphi_nk[n, k] -= mndp.d/2 * log(2pi)
        end
    end

    for n in 1:length(n_subset)
        intermediate_logphi[n, :] .-= logsumexp(intermediate_logphi[n, :])
        mndp.logphi_nk[n_subset[n], :] .= logsumexp.(log(1 - rho) .+ mndp.logphi_nk[n_subset[n], :], log(rho) .+ intermediate_logphi[n, :])
        # mndp.logphi_nk[n_subset[n], :] .= (1 - rho) * mndp.logphi_nk[n_subset[n], :] + rho * intermediate_logphi[n, :]
        # mndp.logphi_nk[n_subset[n], :] .-= logsumexp(mndp.logphi_nk[n_subset[n], :])
    end
    # for n in n_subset
    #     mndp.logphi_nk[n, :] .= mndp.logphi_nk[n, :] .- logsumexp(mndp.logphi_nk[n, :])
    # end

    return mndp
end

function naturalize(mu::AbstractVector{Float64}, lambda::Float64, psi::AbstractMatrix{Float64}, nu::Float64)
    return lambda * mu, lambda, psi + lambda * mu * mu', nu
end

function canonize(eta1::AbstractVector{Float64}, eta2::Float64, eta3::AbstractMatrix{Float64}, eta4::Float64)
    return eta1 / eta2, eta2, eta3 - eta1 * eta1' / eta2, eta4
end

function mu_k(eta1_k::AbstractMatrix{Float64}, eta2_k::AbstractVector{Float64})
    return eta1_k ./ eta2_k
end

function mu_k(mndp::MNDPVariational)
    return mu_k(mndp.eta1_k, mndp.eta2_k)
end

function lambda_k(eta2_k::AbstractVector{Float64})
    return eta2_k[:]
end
function lambda_k(mndp::MNDPVariational)
    return lambda_k(mndp.eta2_k)
end

function psi_k(eta1_k::AbstractMatrix{Float64}, eta2_k::AbstractVector{Float64}, eta3_k::AbstractArray{Float64, 3})
    na = [CartesianIndex()]
    psi0_k = eta3_k - 1.0 ./ eta2_k[:, na, na] .* eta1_k[:, :, na] .* eta1_k[:, na, :]
    # There's some minute machine precision 
    # differences between upper/lower diagonal elements >:(
    # so let's fix that
    return (psi0_k + permutedims(psi0_k, [1, 3, 2])) / 2
end

function psi_k(mndp::MNDPVariational)
    return psi_k(mndp.eta1_k, mndp.eta2_k, mndp.eta3_k)
end

function nu_k(eta4_k::AbstractVector{Float64})
    return eta4_k[:]
end

function nu_k(mndp::MNDPVariational)
    return nu_k(mndp.eta4_k)
end

function draw_partition(mndp::MNDPVariational; map=false)::Vector{Cluster}
    clusters = Dict{Int64, Cluster}()

    for n in 1:mndp.N
        if map
            _, cluster_idx = findmax(mndp.logphi_nk[n, :])
        else
            cluster_idx = findfirst(rand(Multinomial(1, exp.(mndp.logphi_nk[n, :]))) .== 1)
        end
        
        if !(cluster_idx in keys(clusters))
            clusters[cluster_idx] = Cluster(mndp.d)
        end

        push!(clusters[cluster_idx], mndp.data[n, :])
    end

    return collect(values(clusters))
end


function variational_objective(mndp::MNDPVariational)
    return variational_objective(
        mndp.gamma1_k,
        mndp.gamma2_k,
        mndp.eta1_k,
        mndp.eta2_k,
        mndp.eta3_k,
        mndp.eta4_k,
        mndp.logphi_nk,
        mndp.hyperparams.alpha,
        mndp.hyperparams.mu,
        mndp.hyperparams.lambda,
        mndp.hyperparams.psi,
        mndp.hyperparams.nu)
end

function variational_objective(
    gamma1_k::AbstractVector{Float64},
    gamma2_k::AbstractVector{Float64}, 
    eta1_k::AbstractMatrix{Float64},
    eta2_k::AbstractVector{Float64},
    eta3_k::AbstractArray{Float64, 3},
    eta4_k::AbstractVector{Float64},
    logphi_nk::AbstractMatrix{Float64},
    alpha::Float64,
    mu0::AbstractVector{Float64},
    lambda0::Float64,
    psi0::AbstractMatrix{Float64},
    nu0::Float64
    )

    d = length(mu0)
    N, T = size(logphi_nk)
    na = [CartesianIndex()]

    mu0_k = mu_k(eta1_k, eta2_k)
    lambda0_k = eta2_k
    psi0_k = psi_k(eta1_k, eta2_k, eta3_k)
    nu0_k = eta4_k

    Eq_log_p_DP = sum([(alpha - 1)*(polygamma(0, gamma1_k[k]) - polygamma(0, gamma1_k[k] + gamma2_k[k])) + log(alpha) for k in 1:T-1])

    
    phi_nk_nsums = exp.(logsumexp(logphi_nk, dims=1))[1, :]
    # sum_n sum_j=k+1^T phi_nj
    phi_nk_nrck = reverse(cumsum(reverse(phi_nk_nsums))) - phi_nk_nsums
    
    Eq_log_p_slice = sum(phi_nk_nrck[1:end-1] .* (polygamma.(0, gamma1_k)) - polygamma.(0, gamma1_k + gamma2_k))
    Eq_log_p_slice += sum(phi_nk_nsums[1:end-1] .* (polygamma.(0, gamma2_k)) - polygamma.(0, gamma1_k + gamma2_k))

    Eq_log_p_niw = 0.0
    

    Eq_log_p_llk = 0.0



    Eq_log_q = 0.0

    Eq_log_q += sum((gamma1_k .- 1) .* (polygamma.(0, gamma1_k) .- polygamma.(0, gamma1_k + gamma2_k)))
    Eq_log_q += sum((gamma2_k .- 1) .* (polygamma.(0, gamma2_k) .- polygamma.(0, gamma1_k + gamma2_k)))
    Eq_log_q -= sum(logbeta.(gamma1_k, gamma2_k))

    # Contribution from -1/2 (nu0_k + d + 2) * Eq[logdet(Sigma_k)]
    Eq_log_q += sum(
        -1/2 * (nu0_k .+ d .+ 2) .* (
            -d * log(2)
            .+ logdetpsd.(eachslice(psi0_k, dims=1))
            .- sum(polygamma.(0, nu0_k[:, na]/2 .+ 1/2 .- (1:d)[na, :]/2), dims=2)[1, :]
            )
        )
    # Contribution from 
    # -1/2 Eq[trace(psi0_k * Sigma_k^-1)] = -1/2 trace(psi0_k * Eq[Sigma_k^-1])
    Eq_log_q -= d/2 * sum(nu0_k)
    # Combined contributions from
    # -lambda0_k/2 (Eq[Mu_k' * Sigma_k^-1 * Mu_k]
    #              - 2 mu0_k' Eq[Sigma_k^-1 Mu_k]
    #                + mu0_k' Eq[Sigma_k^-1] mu0_k)
    # (lots of cancellations)
    Eq_log_q -= T * d/2

    Eq_log_q -= sum(log_Zniw.(nothing, Vector{Float64}.(eachslice(mu0_k, dims=1)), lambda0_k, Matrix{Float64}.(eachslice(psi0_k, dims=1)), nu0_k))
    
    Eq_log_q += sum(exp.(logphi_nk) .* logphi_nk)

    return Eq_log_p_DP + Eq_log_p_slice - Eq_log_q
end

function log_Q(
    V_k::AbstractVector{Float64}, 
    Mu_k::AbstractMatrix{Float64}, 
    Sigma_k::AbstractArray{Float64, 3}, 
    Z_nk::AbstractMatrix{Int64}, 
    mndp::MNDPVariational
    )
   return log_Q(
        V_k, Mu_k, Sigma_k, Z_nk, 
        mndp.gamma1_k, mndp.gamma2_k, 
        mndp.eta1_k, mndp.eta2_k, mndp.eta3_k, mndp.eta4_k,
        mndp.logphi_nk
    )
end

function log_Q(
    V_k::AbstractVector{Float64}, 
    Mu_k::AbstractMatrix{Float64}, 
    Sigma_k::AbstractArray{Float64, 3}, 
    Z_nk::AbstractMatrix{Int64}, 
    gamma1_k::AbstractVector{Float64}, 
    gamma2_k::AbstractVector{Float64},
    eta1_k::AbstractMatrix{Float64}, 
    eta2_k::AbstractVector{Float64}, 
    eta3_k::AbstractArray{Float64, 3},
    eta4_k::AbstractVector{Float64},
    logphi_nk::AbstractMatrix{Float64}
    )

    d = size(eta1_k, 2)

    mu0_k = mu_k(eta1_k, eta2_k)
    lambda0_k = eta2_k
    psi0_k = psi_k(eta1_k, eta2_k, eta3_k)
    nu0_k = eta4_k

    log_q = 0.0
    log_q = sum((gamma1_k .- 1) .* log.(V_k) + (gamma2_k .- 1) .* log.(1 .- V_k) - logbeta.(gamma1_k, gamma2_k))
    log_q += sum(
        - log_Zniw.(nothing, Vector{Float64}.(eachslice(mu0_k, dims=1)), lambda0_k, Matrix{Float64}.(eachslice(psi0_k, dims=1)), nu0_k)
        - 1/2 * (nu0_k .+ d .+ 2) .* logdetpsd.(eachslice(Sigma_k, dims=1))
        - 1/2 * tr.(eachslice(psi0_k, dims=1) ./ eachslice(Sigma_k, dims=1))
        - 1/2 * lambda0_k .* dot.(eachslice(Mu_k - mu0_k, dims=1), eachslice(Sigma_k, dims=1) .\ eachslice(Mu_k - mu0_k, dims=1))
    )
    log_q += sum(Z_nk .* logphi_nk) 

    return log_q
    
end

function draw_Q(mndp::MNDPVariational)
    return draw_Q(
        mndp.gamma1_k, mndp.gamma2_k, 
        mndp.eta1_k, mndp.eta2_k, mndp.eta3_k, mndp.eta4_k,
        mndp.logphi_nk)
    
end

function draw_Q(
    gamma1_k::AbstractVector{Float64}, 
    gamma2_k::AbstractVector{Float64}, 
    eta1_k::AbstractMatrix{Float64},
    eta2_k::AbstractVector{Float64},
    eta3_k::AbstractArray{Float64, 3},
    eta4_k::AbstractVector{Float64},
    logphi_nk::AbstractMatrix{Float64}
    )

    mu0_k = mu_k(eta1_k, eta2_k)
    lambda0_k = eta2_k
    psi0_k = psi_k(eta1_k, eta2_k, eta3_k)
    nu0_k = eta4_k

    V_k = rand.(Beta.(gamma1_k, gamma2_k))

    MuSig_ks = drawNIW.(eachslice(mu0_k, dims=1), lambda0_k, Matrix{Float64}.(eachslice(psi0_k, dims=1)), nu0_k)
    Mu_k = copy(reduce(hcat, getindex.(MuSig_ks, 1))')
    Sigma_k = permutedims(cat(getindex.(MuSig_ks, 2)..., dims=3), [3, 1, 2])

    Z_nk = rand.(Multinomial.(1, Vector{Float64}.(eachslice(exp.(logphi_nk), dims=1))))
    Z_nk = copy(reduce(hcat, Z_nk)')


    return V_k, Mu_k, Sigma_k, Z_nk
end

function log_P(
    V_k::AbstractVector{Float64}, 
    Mu_k::AbstractMatrix{Float64}, 
    Sigma_k::AbstractArray{Float64, 3}, 
    Z_nk::AbstractMatrix{Int64}, 
    hyperparams::MNCRPHyperparams,
    )
   return log_P(
        V_k, Mu_k, Sigma_k, Z_nk, 
        hyperparams
    )
end

function log_Ptrunc(
    V_k::AbstractVector{Float64}, 
    Mu_k::AbstractMatrix{Float64}, 
    Sigma_k::AbstractArray{Float64, 3}, 
    Z_nk::AbstractMatrix{Int64}, 
    hyperparams::MNCRPHyperparams
    )

    na = [CartesianIndex()]
    alpha, mu0, lambda0, psi0, nu0 = collect(hyperparams)
    d = length(mu0)
    N, T = size(Z_nk)
    @assert length(V_k) + 1 == size(Mu_k, 1) == size(Sigma_k, 1) == T
    
    log_p = 0.0
    
    # Truncated DP stick-breaking contribution
    log_p += (alpha - 1) * sum(log.(1 .- V_k)) + T * log(alpha)

    log_p += sum(
        - 1/2 * (nu0 + d + 2) .* logdetpsd.(eachslice(Sigma_k, dims=1))
        - 1/2 * tr.([psi0 / sig for sig in eachslice(Sigma_k, dims=1)])
        - 1/2 * lambda0 * dot.(eachslice(Mu_k .- mu0[na, :], dims=1), eachslice(Sigma_k, dims=1) .\ eachslice(Mu_k .- mu0[na, :], dims=1))
    )
    log_p -= T * log_Zniw(nothing, mu0, lambda0, psi0, nu0)



    return log_p
end

function predictive_logpdf(mndp::MNDPVariational)
    return predictive_logpdf(mndp.gamma1_k, mndp.gamma2_k, mndp.eta1_k, mndp.eta2_k, mndp.eta3_k, mndp.eta4_k)
end

function predictive_logpdf(
    gamma1_k::AbstractVector{Float64}, gamma2_k::AbstractVector{Float64}, 
    eta1_k::AbstractMatrix{Float64}, eta2_k::AbstractVector{Float64}, eta3_k::AbstractArray{Float64, 3}, eta4_k::AbstractVector{Float64}
    )

    mvstudent_degs_mus_sigs = updated_mvstudent_params.(nothing, 
                                                        eachslice(mu_k(eta1_k, eta2_k), dims=1), 
                                                        eta2_k, 
                                                        eachslice(psi_k(eta1_k, eta2_k, eta3_k), dims=1), 
                                                        eta4_k)

    log_Eq_pik = cumsum(log.(gamma2_k) - log.(gamma1_k + gamma2_k)) - (log.(gamma2_k) - log.(gamma1_k + gamma2_k))
    log_Eq_pik += log.(gamma1_k) - log.(gamma1_k + gamma2_k)
    weights = exp.(log_Eq_pik)
    push!(weights, exp(sum(log.(gamma2_k) - log.(gamma1_k + gamma2_k))))

    return predictive_logpdf(weights, mvstudent_degs_mus_sigs)

end

function tail_probability(mndp::MNDPVariational; rejection_samples=10000)
    return tail_probability(mndp.gamma1_k, mndp.gamma2_k, mndp.eta1_k, mndp.eta2_k, mndp.eta3_k, mndp.eta4_k, rejection_samples=rejection_samples)
end

function tail_probability(
    gamma1_k::AbstractVector{Float64}, gamma2_k::AbstractVector{Float64}, 
    eta1_k::AbstractMatrix{Float64}, eta2_k::AbstractVector{Float64}, eta3_k::AbstractArray{Float64, 3}, eta4_k::AbstractVector{Float64};
    rejection_samples=10000)::Function

    mvstudent_degs_mus_sigs = updated_mvstudent_params.(nothing, 
                                                        eachslice(mu_k(eta1_k, eta2_k), dims=1), 
                                                        eta2_k, 
                                                        eachslice(psi_k(eta1_k, eta2_k, eta3_k), dims=1), 
                                                        eta4_k)

    log_Eq_pik = cumsum(log.(gamma2_k) - log.(gamma1_k + gamma2_k)) - (log.(gamma2_k) - log.(gamma1_k + gamma2_k))
    log_Eq_pik += log.(gamma1_k) - log.(gamma1_k + gamma2_k)
    weights = exp.(log_Eq_pik)
    push!(weights, exp(sum(log.(gamma2_k) - log.(gamma1_k + gamma2_k))))

    return tail_probability(weights, mvstudent_degs_mus_sigs, rejection_samples=rejection_samples)
    
end

function tail_probability(coordinates::AbstractVector{Vector{Float64}}, v::MNDPVariational; rejection_samples=10000)
    return tail_probability(v, rejection_samples=rejection_samples).(coordinates)
end