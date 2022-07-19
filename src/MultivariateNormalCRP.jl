module MultivariateNormalCRP
    using Distributions: logpdf, MvNormal, InverseWishart, Normal, Cauchy, Uniform, Binomial
    using Random: randperm, shuffle, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, Weights
    using LinearAlgebra: det, LowerTriangular, cholesky, diag, tr, diagm
    using SpecialFunctions: loggamma, polygamma
    using Base.Iterators: cycle
    using ColorSchemes: Paired_12
    using Plots: plot, vline!, hline!, scatter!, @layout, grid
    # using Optim: optimize, minimizer, summary, minimum, BFGS

    export advance_chain!, initiate_chain, plot_pi_state, drawNIW, stats, plot_chain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain

    # gamma hyperprior on (nu - (d - 1)) with mean 3, so that mean(nu) = d + 2
    # and thus mean(Sigma) = Psi (from Inverse-Wishart mean(Sigma) = Psi/(nu - d - 1))
    # We choose a = 1.0 and b = 0.333 because it gives the gamma distribution
    # with the largest entropy given a/b = 3
    # ............a = 1.0 mean it's an exponential distribution *facepalm*
    
    # a_nu0 = 0.5 to mimick the sqrt(nu - d + 1) dependence of the jeffreys prior
    # b_nu > 0 to reign in runaway fluctuations 
    # a_nu0/b_nu0 = 3 so that mean(sigma) = psi
    a_nu0 = 0.5
    b_nu0 = 0.166

    a_lambda0 = 0.001 # 1/lambda near 0
    b_lambda0 = 0.001 # Mimic long tail at larger lambda
                      # mean(lambda0) = 1.0

    # So annoying, can't find a good hyperprior for nu

    mutable struct MNCRPparams
        alpha::Float64
        mu::Vector{Float64}
        lambda::Float64
        flatL::Vector{Float64}
        L::LowerTriangular{Float64, Matrix{Float64}}
        psi::Matrix{Float64}
        nu::Float64
    end

    mutable struct MNCRPchain

        # Current state of the partition over samples
        # from the Chinese Restaurant Process
        pi_state::Vector{Set{Vector{Float64}}}
        # Current value of hyperparameters
        params::MNCRPparams

        # Some chains of interests
        nbclusters_chain::Vector{Int64}
        params_chain::Vector{MNCRPparams}
        logprob_chain::Vector{Float64}

        # Maximum a-posteriori state and location
        map_pi::Vector{Set{Vector{Float64}}}
        map_params::MNCRPparams
        map_logprob::Float64
        map_idx::Int64

    end

    alpha_chain(chain::MNCRPchain) = [p.alpha for p in chain.params_chain]
    mu_chain(chain::MNCRPchain) = [p.mu for p in chain.params_chain]
    mu_chain(chain::MNCRPchain, i) = [p.mu[i] for p in chain.params_chain]
    lambda_chain(chain::MNCRPchain) = [p.lambda for p in chain.params_chain]
    psi_chain(chain::MNCRPchain) = [p.psi for p in chain.params_chain]
    psi_chain(chain::MNCRPchain, i, j) = [p.psi[i, j] for p in chain.params_chain]
    nu_chain(chain::MNCRPchain) = [p.nu for p in chain.params_chain]
    
    function flatten(L::LowerTriangular{Float64})
        
        d = size(L, 1)
    
        flatL = zeros(Int64(d * (d + 1) / 2))
    
        idx = 1
        for i in 1:d
            for j in 1:i
                flatL[idx] = L[i, j]
                idx += 1
            end
        end
    
        return flatL
    end

    function foldflat(flatL::Vector{Float64})
        
        n = size(flatL, 1)

        # Recover the dimension of a matrix
        # from the length of the vector
        # containing elements of the diag + lower triangular
        # part of the matrix. The condition is that
        # length of vector == #els diagonal + #els lower triangular part
        # i.e N == d + (d² - d) / 2 
        # Will fail at Int64() if this condition
        # cannot be satisfied for N and d integers
        d = Int64((sqrt(1 + 8 * n) - 1) / 2)
    
        L = LowerTriangular(zeros(d, d))
    
        # The order is row major
        idx = 1
        for i in 1:d
            for j in 1:i
                L[i, j] = flatL[idx]
                idx += 1
            end
        end
    
        return L
    end

    function MNCRPparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, flatL::Vector{Float64}, nu::Float64)
        d = size(mu, 1)
        flatL_d = Int64(d * (d + 1) / 2)
        if size(flatL, 1) != flatL_d
            error("Dimension mismatch, flatL should have length $flatL_d")
        end

        L = foldflat(flatL)
        psi = L * L'
    
        return MNCRPparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, L::LowerTriangular{Float64}, nu::Float64)
        d = size(mu, 1)
        if !(d == size(L, 1))
            error("Dimension mismatch, L should have dimension $d x $d")
        end
    
        psi = L * L'
        flatL = flatten(L)
    
        return MNCRPparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)
        d = size(mu, 1)
        if !(d == size(psi, 1) == size(psi, 2))
            error("Dimension mismatch, L should have dimension $d x $d")
        end
    
        L = cholesky(psi).L
        flatL = flatten(L)
    
        return MNCRPparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPparams(d::Int64)
        return MNCRPparams(1.0, zeros(d), 1.0, LowerTriangular(diagm(fill(1.0, d))), 1.0 * d)
    end

    function dimension(params::MNCRPparams)
        @assert size(params.mu, 1) == size(params.psi, 1) == size(params.psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
        return size(params.mu, 1)
    end

    function flatL!(params::MNCRPparams, value::Vector{Float64})
        d = dimension(params)
        if (2 * size(value, 1) != d * (d + 1))
            error("Dimension mismatch, value should have length $(Int64(d*(d+1)/2))")
        end
        params.flatL = copy(value) # Just making sure?
        params.L = foldflat(params.flatL)
        params.psi = params.L * params.L'
    end

    function L!(params::MNCRPparams, value::T) where {T <: AbstractMatrix{Float64}}
        d = dimension(params)
        if !(d == size(value, 1) == size(value, 2))
            error("Dimension mismatch, value shoud have size $d x $d")
        end
        params.L = LowerTriangular(value)
        params.flatL = flatten(params.L)
        params.psi = params.L * params.L'
    end

    function psi!(params::MNCRPparams, value::Matrix{Float64})
        d = dimension(params)
        if !(d == size(value, 1) == size(value, 2))
            error("Dimension mismatch, value shoud have size $d x $d")
        end
        params.L = cholesky(value).L
        params.flatL = flatten(params._L)
        params.psi = value
    end

    function drawNIW(
        mu::Vector{Float64}, 
        lambda::Float64, 
        psi::Matrix{Float64}, 
        nu::Float64)
    
        invWish = InverseWishart(nu, psi)
        sigma = rand(invWish)
    
        multNorm = MvNormal(mu, sigma/lambda)
        mu = rand(multNorm)
    
        return mu, sigma
    end

    function drawNIW(params::MNCRPparams)
        return drawNIW(params.mu, params.lambda, params.psi, params.nu)
    end


    # From conjugacy between NIW and MvNormal:
    # Update parameters of the normal-inverse-Wishart distribution
    # when joined with its data likelihood (mv normal distributions)
    function updated_niw_params(
        cluster::Union{Nothing, Set{Vector{Float64}}},
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)
    
        @assert size(mu, 1) == size(psi, 1) == size(psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
  
        if cluster === nothing || length(cluster) == 0
            return (mu, lambda, psi, nu)
        else
            d = size(mu, 1)
            n = length(cluster)
            lambda_c = lambda + n
            nu_c = nu + n

            # Slow but explicit
            # mean_x = mean(cluster)
            # psi_c = psi
            # psi_c += sum((x - mean_x) * (x - mean_x)' for x in cluster)
            # psi_c += lambda * n / (lambda + n) * (mean_x - mu) * (mean_x - mu)'

            ###############################
            # Optimized type-stableish(?) version of the 4 lines above
            ###############################

            X::Matrix{Float64} = Array{Float64}(undef, n, d)
            @inbounds for (k, x::Vector{Float64}) in enumerate(cluster)
                for i in 1:d
                    X[k, i] = x[i]
                end
            end

            # mean_x::Vector{Float64} = mean(X, dims=1)[1, :]
            mean_x::Vector{Float64} = sum(X, dims=1)[1, :] ./ n
            # mean_x = Array{Float64}(undef, d)
            # @inbounds for i in 1:d
            #     mean_x[i] = 0.0
            #     for k in 1:n
            #         mean_x[i] += X[k, i]
            #     end
            #     mean_x[i] /= n
            # end
                        
            
            prepsi_c = Array{Float64}(undef, n, d, d)
            # It is somehow marginally faster to allocate 
            # psi_c here than after the loop over prepsi_c
            psi_c = Array{Float64}(undef, d, d)

            @inbounds for i in 1:d
                for j in 1:i
                    for k in 1:n
                        prepsi_c[k, i, j] = (X[k, i] - mean_x[i]) * (X[k, j] - mean_x[j])
                        prepsi_c[k, j, i] = prepsi_c[k, i, j]
                    end
                end
            end

            # psi_c = psi
            # psi_c += lambda * n / (lambda + n) * (mean_x - mu) * (mean_x - mu)'
            # psi_c = Array{Float64}(undef, d, d)
            @inbounds for i in 1:d
                for j in 1:i
                    psi_c[i, j] = psi[i, j] + lambda * n / (lambda + n) * (mean_x[i] - mu[i]) * (mean_x[j] - mu[j])
                    psi_c[j, i] = psi_c[i, j]
                end
            end

            # psi_c += sum(prepsi_c, dims=1)[1, :, :]
            @inbounds for i in 1:d
                for j in 1:i-1
                    for k in 1:n
                        psi_c[i, j] += prepsi_c[k, i, j]
                    end
                    psi_c[j, i] = psi_c[i, j]
                end
                for k in 1:n
                    psi_c[i, i] += prepsi_c[k, i, i]
                end
            end
            ###############################

            # mu_c = (lambda * mu + n * mean_x) / (lambda + n)
            mu_c = Array{Float64}(undef, d)
            @inbounds for i in 1:d
                mu_c[i] = (lambda * mu[i] + n * mean_x[i]) / (lambda + n)
            end

            return (mu_c, lambda_c, psi_c, nu_c)
        end
    
    end

    # Return the normalization constant 
    # of the normal-inverse-Wishart distribution,
    # possibly in the presence of data if cluster isn't empty
    function log_Zniw(
        cluster::Union{Nothing, Set{Vector{Float64}}},
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)
        
        @assert size(mu, 1) == size(psi, 1) == size(psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"

        d = size(mu, 1)
    
        mu, lambda, psi, nu = updated_niw_params(cluster, mu, lambda, psi, nu)
        
        lognum = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    
        logdenum = d/2 * log(lambda) + nu/2 * log(det(psi))

        return lognum - logdenum
    
    end

    # Return the log-likelihood of the model
    function log_prob(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams)
    
        N = sum(length(c) for c in list_of_clusters)
        K = length(list_of_clusters)
    
        alpha, mu, lambda, psi, nu = params.alpha, params.mu, params.lambda, params.psi, params.nu
        d = size(mu, 1)
    
        # Log-probability associated with the Chinese Restaurant Process
        crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum(loggamma(length(c)) for c in list_of_clusters)
        
        # Log-probability associated with the data likelihood
        # and Normal-Inverse-Wishart base distribution of the CRP
        
        log_z_niw = 0.0
        for c in list_of_clusters
            log_z_niw += log_Zniw(c, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - length(c) * d/2 * log(2pi) 
        end

        log_hyperpriors = 0.0 # mu0 has a flat hyperpriors
        
        # alpha hyperprior
        log_hyperpriors += log(niw_unnorm_jeffreys_alpha(alpha, N))
        # lambda hyperprior
        log_hyperpriors += 1/2 * log(d/2) - log(lambda)
        # psi hyperprior
        log_hyperpriors += 1/2 * log(nu/2) - log(det(psi))
        # nu hyperprior
        log_hyperpriors += log(niw_unnorm_jeffreys_nu(nu, d))
        
        # sqrt(nu/2)/det(psi) hyperprior
        # log_hyperpriors += 1/2 * log(nu/2) - log(det(psi))

        # log_hyperpriors -= log(alpha)  # 1/alpha hyperprior
        # log_hyperpriors -= 1/2 * log(alpha) # 1/sqrt(alpha) hyperprior

        # log_hyperpriors -= log(lambda) # 1/lambda0 hyperprior
        # log_hyperpriors += a_lambda0 * log(b_lambda0) - loggamma(a_lambda0) + (a_lambda0 - 1) * log(lambda) - b_lambda0 * lambda

        # nu - (d - 1) ~ gamma(1, 0.333) hyperprior (max entropy with mean 3)
        # log_hyperpriors += a_nu0 * log(b_nu0) - loggamma(a_nu0) + (a_nu0 - 1) * log(nu - (d - 1)) - b_nu0 * (nu - (d - 1)) 
        
        # nu hyperprior, works for smaller nu in synthetic data
        # byt diverges again when nu is too big

        return crp + log_z_niw + log_hyperpriors
    
    end

    function add_to_state_gibbs!(element::Vector{Float64}, list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams)#; restricted=false)

        # Clean up just in case to make sure
        # the only empty cluster will be the one from
        # the CRP with probability prop to alpha
        # filter!(c -> length(c) > 0, list_of_clusters)

        @assert all([!isempty(c) for c in list_of_clusters])

        element_set = Set(Vector{Float64}[element])
        Nminus1 = sum([length(c) for c in list_of_clusters])        
        alpha, mu, lambda, psi, nu = params.alpha, params.mu, params.lambda, params.psi, params.nu
        d = size(mu, 1)
    
        crp_log_weights = [log(length(c)) - log(alpha + Nminus1) for c in list_of_clusters]    
            
        niw_log_weights = [(log_Zniw(union(element_set, c), mu, lambda, psi, nu) 
                          - log_Zniw(c, mu, lambda, psi, nu)
                          - d/2 * log(2pi)) for c in list_of_clusters]
        
        # Actually slower
        # niw_log_weights = Array{Float64}(undef, length(list_of_clusters))
        # for (i, c) in enumerate(list_of_clusters)
        #     push!(c, element)
        #     niw_log_weights[i] = log_Zniw(c, mu, lambda, psi, nu)
        #     pop!(c, element)
        #     niw_log_weights[i] -= log_Zniw(c, mu, lambda, psi, nu) + d/2 * log(2pi)
        # end
        
        # Add single empty cluster to potential choice with its associated weights
        empty_set = Set{Vector{Float64}}([])
        push!(list_of_clusters, empty_set)
        push!(crp_log_weights, log(alpha) - log(alpha + Nminus1))
        push!(niw_log_weights, log_Zniw(element_set, mu, lambda, psi, nu) 
                             - log_Zniw(nothing, mu, lambda, psi, nu) 
                             - d/2 * log(2pi))
    
        unnorm_logp = [crp + niw for (crp, niw) in zip(crp_log_weights, niw_log_weights)]
        norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
        probs = Weights(exp.(norm_logp))
    
        selected_cluster = sample(list_of_clusters, probs)
        
        # N1 = sum(length(c) for c in list_of_clusters)
        # @assert N1 == Nminus1
        push!(selected_cluster, element)
        # N2 = sum(length(c) for c in list_of_clusters)
        # @assert N2 == N1 + 1 "$N1 $N2) "
    
        # if isempty(list_of_clusters[end])
        #     pop!(list_of_clusters)
        # end

        filter!(c -> length(c) > 0, list_of_clusters)
        
    end

    function advance_clusters_gibbs!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams)
    
        elements = shuffle([e for c in list_of_clusters for e in c])
    
        for e in elements
            N = sum(length(c) for c in list_of_clusters)
            for (ci, c) in enumerate(list_of_clusters)
                if e in c
                    pop!(c, e)
                    if isempty(c)
                        deleteat!(list_of_clusters, ci)
                    end
                    break
                end
            end
            @assert sum(length(c) for c in list_of_clusters) == N - 1

            add_to_state_gibbs!(e, list_of_clusters, params)   
        
        end
    
    end


    # Split-Merge Metropolis-Hastings with restricted Gibbs sampling from Jain & Neal 2004
    # (Sorry for the ravioli code, might refactor later, might not)
    function advance_clusters_JNrestrictedsplitmerge!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams; t=5)
            
        alpha, mu, lambda, psi, nu = params.alpha, params.mu, params.lambda, params.psi, params.nu

        d = size(mu, 1)
    
        elements = [(ce, e) for (ce, c) in enumerate(list_of_clusters) for e in c]
    
        (ci, i), (cj, j) = sample(elements, 2, replace=false)
    
        # S(minus)ij will only be used for sampling without replacement
        # and shouldn't change during execution
        Smij = union(list_of_clusters[ci], list_of_clusters[cj])
        pop!(Smij, i)
        pop!(Smij, j)
        Smij = collect(Smij)
    
        # For merge moves we need the initial state
        # for the "hypothetical" restricted gibbs state
        # that would bring the launch state into the initial state.
        # Namely we need it to form q(initial | merge) = q(initial | launch)
        if ci == cj
        
            initial_S = copy(list_of_clusters[cj])
        
            launch_Si = Set{Vector{Float64}}([i])

            # Avoid one more copy
            launch_Sj = list_of_clusters[cj]
            pop!(launch_Sj, i)
        
            filter!(x -> !(x === launch_Sj), list_of_clusters) # Julia is so fucking clunky sometimes

        elseif ci != cj

            initial_Si = copy(list_of_clusters[ci])
            initial_Sj = copy(list_of_clusters[cj])

            # Avoaid two more copy
            launch_Si = list_of_clusters[ci]
            launch_Sj = list_of_clusters[cj]
        
            filter!(x -> !(x === launch_Si || x === launch_Sj), list_of_clusters)
        
        end
    

        # Shuffle elements (except i and j) with equal probability
        # of being reassigned to either launch cluster
        for e in Smij
            delete!(launch_Si, e)
            delete!(launch_Sj, e)
            choice_of_lSi_or_lSj = sample([launch_Si, launch_Sj])
            push!(choice_of_lSi_or_lSj, e)
        end
    
    
        # if t >= 1 perform t restricted Gibbs scans
        # where each element (except i and j) is reassigned
        # to either the cluster with i or the cluster with j
        for n in 1:t-1
        
            for e in sample(Smij, length(Smij), replace=false)
                
                delete!(launch_Si, e)
                delete!(launch_Sj, e)
            
                singleton_e = Set{Vector{Float64}}([e])
            
                #############

                log_weight_Si = log(length(launch_Si))

                # log_weight_Si += log_Zniw(union(singleton_e, launch_Si), mu, lambda, psi, nu) 
                push!(launch_Si, e)
                log_weight_Si += log_Zniw(launch_Si, mu, lambda, psi, nu) 
                pop!(launch_Si, e)

                log_weight_Si -= log_Zniw(launch_Si, mu, lambda, psi, nu)
                log_weight_Si -= d/2 * log(2pi) # Just for the sake of explicitness, will cancel out

                #############

                log_weight_Sj = log(length(launch_Sj))

                # log_weight_Sj += log_Zniw(union(singleton_e, launch_Sj), mu, lambda, psi, nu) 
                push!(launch_Sj, e)
                log_weight_Sj += log_Zniw(launch_Sj, mu, lambda, psi, nu)
                pop!(launch_Sj, e)

                log_weight_Sj -= log_Zniw(launch_Sj, mu, lambda, psi, nu)
                log_weight_Sj -= d/2 * log(2pi)

                #############

                unnorm_logp = [log_weight_Si, log_weight_Sj]
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))

                choice_of_lSi_or_lSj = sample([launch_Si, launch_Sj], probs)
            
                push!(choice_of_lSi_or_lSj, e)
            
            end
        
        end
    
        ##############
        ### Split! ###
        ##############
        if ci == cj
            # Perform last restricted Gibbs scan to form
            # proposed split state and keep track of assignment
            # probabilities to form the Hastings ratio
        
            log_q_proposed_from_launch = 0.0

            # We rename launch_Si and launch_Sj just
            # to make it explicit that the last restricted
            # Gibbs scan produces the proposed split state
            proposed_Si = launch_Si
            proposed_Sj = launch_Sj
        
            for e in sample(Smij, length(Smij), replace=false)

                delete!(proposed_Si, e)
                delete!(proposed_Sj, e)

                singleton_e = Set{Vector{Float64}}([e])

                #############

                log_weight_Si = log(length(proposed_Si))

                # log_weight_Si += log_Zniw(union(singleton_e, proposed_Si), mu, lambda, psi, nu) 
                push!(proposed_Si, e)
                log_weight_Si += log_Zniw(proposed_Si, mu, lambda, psi, nu) 
                pop!(proposed_Si, e)

                log_weight_Si -= log_Zniw(proposed_Si, mu, lambda, psi, nu)
                # There is no Z0 in the denominator of the predictive posterior
                log_weight_Si -= d/2 * log(2pi) # Just for the sake of explicitness, will cancel out
            
                #############

                log_weight_Sj = log(length(proposed_Sj))

                # log_weight_Sj += log_Zniw(union(singleton_e, proposed_Sj), mu, lambda, psi, nu) 
                push!(proposed_Sj, e)
                log_weight_Sj += log_Zniw(proposed_Sj, mu, lambda, psi, nu) 
                pop!(proposed_Sj, e)

                log_weight_Sj -= log_Zniw(proposed_Sj, mu, lambda, psi, nu)
                log_weight_Sj -= d/2 * log(2pi)

                #############

                unnorm_logp = [log_weight_Si, log_weight_Sj]
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))

                choice_of_propSi_or_propSj = sample([proposed_Si, proposed_Sj], probs)
                push!(choice_of_propSi_or_propSj, e)

                # Accumulate the log probability of the restricted assignment
                if choice_of_propSi_or_propSj === proposed_Si
                    log_q_proposed_from_launch += norm_logp[1]
                elseif choice_of_propSi_or_propSj === proposed_Sj
                    log_q_proposed_from_launch += norm_logp[2]
                end
            end
        
            n_proposed_Si = length(proposed_Si)
            n_proposed_Sj = length(proposed_Sj)
            n_initial_S = length(initial_S) # = n_proposed_Si + n_proposed_Sj, of course
            @assert n_initial_S == n_proposed_Si + n_proposed_Sj "$n_initial_S, $n_proposed_Si, $n_proposed_Sj"
        
            log_crp_ratio = log(alpha) + loggamma(n_proposed_Si) + loggamma(n_proposed_Sj) - loggamma(n_initial_S)
        
            # The Hastings ratio is 
            # q(initial | proposed) / q(proposed | initial)
            # Here q(initial | proposed) = 1 
            # because the initial state is a merge state with its ci = cj
            # and the proposed state is a split state, so the proposal process
            # would be a merge proposal and there is only one way to do so.
        
            # Note also that q(proposed | initial) = q(proposed | launch)
            # because every initial states gets shuffled and forgotten
            # into a launch state.
            log_hastings_ratio = -log_q_proposed_from_launch
        
            log_likelihood_ratio = log_Zniw(proposed_Si, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Si * d/2 * log(2pi)        
            log_likelihood_ratio += log_Zniw(proposed_Sj, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Sj * d/2 * log(2pi)        
        
            # initial_S is the merge state. This is not the
            # Hastings ratio formed from transition probabilities
            # !of the proposal process!. This is the actual
            # likelihood of the initial merge state in the
            # generative model.
            log_likelihood_ratio -= log_Zniw(initial_S, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_S * d/2 * log(2pi)
            
            log_acceptance = log_hastings_ratio + log_crp_ratio + log_likelihood_ratio
            log_acceptance = min(0.0, log_acceptance) # Unnecessary but whatever, lets follow the convention
        
            if log(rand()) < log_acceptance
                print("s")
                push!(list_of_clusters, proposed_Si, proposed_Sj)
                return 1
            else
                push!(list_of_clusters, initial_S)
                return 0
            end
        
        ##############
        ### Merge! ###
        ##############
        elseif ci != cj
            
            # This is q(initial | merge) = q(initial | launch)
            # and we remember than q(merge | initial) = q(merge | launch) = 1
            log_q_initial_from_launch = 0.0 
        
            proposed_S = union(launch_Si, launch_Sj)
        
            for e in sample(Smij, length(Smij), replace=false)

                delete!(launch_Si, e)
                delete!(launch_Sj, e)

                singleton_e = Set{Vector{Float64}}([e])

                #############
                log_weight_Si = log(length(launch_Si))

                # log_weight_Si += log_Zniw(union(singleton_e, launch_Si), mu, lambda, psi, nu) 
                push!(launch_Si, e)
                log_weight_Si += log_Zniw(launch_Si, mu, lambda, psi, nu) 
                pop!(launch_Si, e)

                log_weight_Si -= log_Zniw(launch_Si, mu, lambda, psi, nu)
                # There is no Z0 in the denominator of the predictive posterior
                log_weight_Si -= d/2 * log(2pi) # Just for the sake of explicitness, will cancel out
            
                #############
                
                log_weight_Sj = log(length(launch_Sj))

                # log_weight_Sj += log_Zniw(union(singleton_e, launch_Sj), mu, lambda, psi, nu) 
                push!(launch_Sj, e)
                log_weight_Sj += log_Zniw(launch_Sj, mu, lambda, psi, nu) 
                pop!(launch_Sj, e)
                
                log_weight_Sj -= log_Zniw(launch_Sj, mu, lambda, psi, nu)
                log_weight_Sj -= d/2 * log(2pi)

                #############

                unnorm_logp = [log_weight_Si, log_weight_Sj]
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

                if in(e, initial_Si)
                    log_q_initial_from_launch += norm_logp[1]
                    push!(launch_Si, e)
                elseif in(e, initial_Sj)
                    log_q_initial_from_launch += norm_logp[2]
                    push!(launch_Sj, e)
                else
                    error("(merge) element in neither initial_Si or initial_Sj")
                end
            end
        
            log_hastings_ratio = log_q_initial_from_launch
        
            n_proposed_S = length(proposed_S)
            n_initial_Si = length(initial_Si)
            n_initial_Sj = length(initial_Sj)
        
            log_crp_ratio = -log(alpha) + loggamma(n_proposed_S) - loggamma(n_initial_Si) - loggamma(n_initial_Sj)
        
            log_likelihood_ratio = log_Zniw(proposed_S, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_S * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_Si, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_Si * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_Sj, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_Sj * d/2 * log(2pi)
        
            log_acceptance = log_hastings_ratio + log_crp_ratio + log_likelihood_ratio
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                print("m")
                push!(list_of_clusters, proposed_S)
                return 1
            else
                push!(list_of_clusters, initial_Si, initial_Sj)
                return 0
            end
        
        end
    
    end

    function advance_alpha!(
        list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams;
        step_type="gaussian", step_scale=0.1)
            
        # No Cauchy because it's a very bad idea on a log scale
        if step_type == "gaussian"
            step_dist = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_dist = Uniform(-step_scale/2, step_scale/2)
        end

        N = sum(length(c) for c in list_of_clusters)

        alpha = params.alpha
    
        # 1/x improper hyperprior on alpha
        proposed_logalpha = log(alpha) + rand(step_dist)
        proposed_alpha = exp(proposed_logalpha)
        
        log_acc = 0.0

        # log_acc += -3/2 * (proposed_logalpha - log(alpha))

        log_acc = log(niw_unnorm_jeffreys_alpha(proposed_alpha, N)) - log(niw_unnorm_jeffreys_alpha(alpha, N))

        log_hastings = log(alpha) - proposed_logalpha

        ####### NO! it's (approx) logarithmic ########
        # log_hastings = log(proposed_alpha) - log(alpha)
        
        # "Jeffreys" 1/sqrt(x) improper hyperprior on alpha
        # coin_flip = 1 # 2 * Int(rand() < 0.5) - 1
        # proposed_alpha = (coin_flip * sqrt(params.alpha) + rand(step_dist))^2

        # log_acc = 1/2 * (-log(proposed_alpha) + log(alpha))
        ##############################################

        log_acc += length(list_of_clusters) * log(proposed_alpha) - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
        log_acc -= length(list_of_clusters) * log(alpha) - loggamma(alpha + N) + loggamma(alpha)
        
        log_acc += log_hastings

        log_acc = min(0, log_acc)

        if log(rand()) < log_acc
            params.alpha = proposed_alpha
            return 1
        else
            return 0
        end
    
    end

    function advance_mu!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams;
                         random_order=true, step_scale=0.1, step_type="gaussian")
    
        if step_type == "cauchy"
            step_dist = Cauchy(0.0, step_scale)
        elseif step_type == "gaussian"
            step_dist = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_dist = Uniform(-step_scale/2, step_scale/2)
        end

        mu, lambda, psi, nu = params.mu, params.lambda, params.psi, params.nu
        
        d = size(mu, 1)
        
        if random_order
            dim_order = randperm(d)
        else
            dim_order = 1:d
        end    
            
        accepted = 0
    
        for i in dim_order
            proposed_mu = copy(params.mu)
            proposed_mu[i] = proposed_mu[i] + rand(step_dist)

            log_acc = sum(log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(nothing, proposed_mu, lambda, psi, nu)
                        - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                     for c in list_of_clusters)
        
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                params.mu = proposed_mu
                accepted += 1
            end
            
        end
        
        return accepted
    
    end

    function advance_lambda!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams;
                             step_type="gaussian", step_scale=0.1)

        if step_type == "gaussian"
            step_dist = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_dist = Uniform(-step_scale/2, step_scale/2)
        end
        
        mu, lambda, psi, nu = params.mu, params.lambda, params.psi, params.nu
    
        proposed_loglambda = log(lambda) + rand(step_dist)
        proposed_lambda = exp(proposed_loglambda)
        
        log_acc = sum(log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(nothing, mu, proposed_lambda, psi, nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                     for c in list_of_clusters)
        # We leave loghastings = 0.0 because apparently the
        # Jeffreys priori of lambda is the logarithmic prior

        # No actual lets regulate lambda with a gamma prior
        # log_acc += proposed_loglambda - log(lambda)
        # log_hastings = log(lambda) - proposed_loglambda
        # log_acc += (a_lambda0 - 1) * log(proposed_lambda) - b_lambda0 * (proposed_lambda)
        # log_acc -= (a_lambda0 - 1) * log(lambda) - b_lambda0 * lambda
        
        # log_acc += log_hastings

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            params.lambda = proposed_lambda
            return 1
        else
            return 0
        end
    
    end

    function advance_psi!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams;
                          random_order=true, step_scale=0.1, step_type="gaussian")
    
        if step_type == "cauchy"
            step_dist = Cauchy(0.0, step_scale)
        elseif step_type == "gaussian"
            step_dist = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_dist = Uniform(-step_scale/2, step_scale/2)
        end
    
        flatL_d = size(params.flatL, 1)
        
        if random_order
            dim_order = randperm(flatL_d)
        else
            dim_order = 1:flatL_d
        end
    
        mu, lambda, nu = params.mu, params.lambda, params.nu
    
        d = size(mu, 1)
        
        accepted = 0
    
        for i in dim_order
            
            proposed_flatL = copy(params.flatL)
        
            proposed_flatL[i] = proposed_flatL[i] + rand(step_dist)
        
            proposed_L = foldflat(proposed_flatL)
            proposed_psi = proposed_L * proposed_L'
        
            log_acc = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                        - log_Zniw(cluster, mu, lambda, params.psi, nu) + log_Zniw(nothing, mu, lambda, params.psi, nu) 
                        for cluster in list_of_clusters)
                
            # Go from symmetric and uniform in L to uniform in psi
            # det(del psi/del L) = 2^d |L_11^d * L_22^(d-1) ... L_nn|
            # 2^d's cancel in the Hastings ratio
            log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(params.L)))))
            log_acc += log_hastings

            # Jeffreys prior in the determinant. Marvelous.
            log_acc +=  -log(det(proposed_psi)) + log(det(params.psi))
            
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                flatL!(params, proposed_flatL)
                accepted += 1
            end
            
        end
    
#         flatL!(params, temp_flatL)
    
        return accepted    
    end


    function advance_nu!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams;
                         step_type="gaussian", step_scale=0.1)

        if step_type == "gaussian"
            step_dist = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_dist = Uniform(-step_scale/2, step_scale/2)
        end
    
        d = size(params.mu, 1)
        
        mu, lambda, psi, nu = params.mu, params.lambda, params.psi, params.nu

        current_logx = log(nu - (d - 1))
        proposed_logx = current_logx + rand(step_dist)
        proposed_nu = d - 1 + exp(proposed_logx)
    
        log_acc = sum(log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(nothing, mu, lambda, psi, proposed_nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                    for c in list_of_clusters)
        
        # Convert back to uniform moves on the positive real line nu > d - 1
        # log_acc += proposed_logx - current_logx 
        log_hastings = current_logx - proposed_logx

        # log_acc += (a_nu0 - 1) * log(proposed_nu - (d - 1)) - b_nu0 * (proposed_nu - (d - 1))
        # log_acc -= (a_nu0 - 1) * log(nu - (d - 1)) - b_nu0 * (nu - (d - 1))

        # jeffreys prior is still unstable
        # for synthetic data with "large" nu
        log_acc += log(niw_unnorm_jeffreys_nu(proposed_nu, d)) - log(niw_unnorm_jeffreys_nu(nu, d))

        log_acc += log_hastings
        # approx jeffrey's sqrt(d/2)/sqrt(nu)
        # log_acc += 1/2 * (-log(proposed_nu - d + 1) + log(nu - d + 1))


        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            params.nu = proposed_nu
            return 1
        else
            return 0
        end
    
    end

    function niw_unnorm_jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha) - polygamma(1, alpha + n))

    end

    # Cute result but unfortunately lead to 
    # improper a-posteriori probability in nu and psi
    function niw_unnorm_jeffreys_nu(nu::Float64, d::Int64)

        return 1/2 * sqrt(sum(polygamma.(1, nu/2 .+ 1/2 .- 1/2 * (1:d))))

    end


    function advance_full_sequential_gibbs!(list_of_clusters::Vector{Set{Vector{Float64}}}, params::MNCRPparams)

        # data = Vector{Float64}[datum for cluster in list_of_clusters for datum in cluster]
        # data = data[randperm(size(data, 1))]
        data = shuffle([element for cluster in list_of_clusters for element in cluster])

        empty!(list_of_clusters)

        for element in data
            add_to_state_gibbs!(element, list_of_clusters, params)
        end

    end

    function initiate_chain(data::Vector{Vector{Float64}})

        @assert all(size(e, 1) == size(data[1], 1) for e in data)

        d = size(data[1], 1)
        params = MNCRPparams(d)
        chain_state = MNCRPchain([], params, [], [], [], [], params, -Inf, 1)

        chain_state.params_chain = [params]

        # will be suffled and replaced rightaway full_sequential_gibbs move
        chain_state.pi_state = [Set(data)]
        advance_full_sequential_gibbs!(chain_state.pi_state, chain_state.params)
        chain_state.nbclusters_chain = [length(chain_state.pi_state)]

        chain_state.map_pi = deepcopy(chain_state.pi_state)
        lp = log_prob(chain_state.pi_state, chain_state.params)
        chain_state.map_logprob = lp
        chain_state.logprob_chain = [lp]
        # map_params=params and map_idx=1 have already been 
        # specified when calling MNCRPchain

        chain_state.logprob_chain = [chain_state.map_logprob]

        return chain_state

    end

    function advance_chain!(chain_state::MNCRPchain; nb_steps=100, nb_splitmerge=5, splitmerge_t=5, nb_gibbs=10, nb_paramsmh=10, fullseq_prob=0.01)

        print(".."); flush(stdout)

        for step in 1:nb_steps

            ## Large moves ##
            if rand() < fullseq_prob
                
                advance_full_sequential_gibbs!(chain_state.pi_state, chain_state.params)

                print("f"); flush(stdout)
            
            end

            for i in 1:nb_splitmerge
                advance_clusters_JNrestrictedsplitmerge!(chain_state.pi_state, chain_state.params, t=splitmerge_t)
            end
            #################

            # Gibbs sweep
            for i in 1:nb_gibbs
                advance_clusters_gibbs!(chain_state.pi_state, chain_state.params)
            end

            push!(chain_state.nbclusters_chain, length(chain_state.pi_state))


            # Metropolis-Hastings over each parameter
            for i in 1:nb_paramsmh
                advance_alpha!(chain_state.pi_state, chain_state.params, 
                                step_type="gaussian", step_scale=0.7)
                
                advance_mu!(chain_state.pi_state, chain_state.params, 
                            step_type="gaussian", step_scale=1.0)
                
                advance_lambda!(chain_state.pi_state, chain_state.params, 
                                step_type="gaussian", step_scale=0.7)

                advance_psi!(chain_state.pi_state, chain_state.params, 
                            step_type="gaussian", step_scale=1.0)

                advance_nu!(chain_state.pi_state, chain_state.params, 
                            step_type="gaussian", step_scale=0.5)
            end

            push!(chain_state.params_chain, deepcopy(chain_state.params))

            # logprob and MAP
            logprob = log_prob(chain_state.pi_state, chain_state.params)
            push!(chain_state.logprob_chain, logprob)
            
            if logprob > chain_state.map_logprob
                chain_state.map_logprob = logprob
                chain_state.map_pi = deepcopy(chain_state.pi_state)
                chain_state.map_params = deepcopy(chain_state.params)
                chain_state.map_idx = lastindex(chain_state.logprob_chain)
                print("^"); flush(stdout)
            end

        end

        print("$(length(chain_state.pi_state))"); flush(stdout)

    end

    function plot_pi_state(pi_state::Vector{Set{Vector{Float64}}}; plot_kw...)
        p = plot(
            legend_position=:outertopright, grid=:no, 
            showaxis=:no, ticks=:true; 
            plot_kw...)

        pi_state = sort(pi_state, by=x -> length(x), rev=true)
        for (cluster, color) in zip(pi_state, cycle(Paired_12))
            scatter!(collect(Tuple.(cluster)), label="$(length(cluster))", 
            color=color, markerstrokewidth=0)
        end
        
        # display(p)
        return p

    end

    function plot_chain(chain_state::MNCRPchain; burn=0)
        p_map = plot_pi_state(chain_state.map_pi, title="MAP state")
        p_current = plot_pi_state(chain_state.pi_state, title="Current state")
        
        lpc = chain_state.logprob_chain
        p_logprob = plot(burn+1:length(lpc), lpc[burn+1:end], grid=:no, label=nothing, title="log probability chain")
        vline!(p_logprob, [chain_state.map_idx], label=nothing, color=:gray)
        hline!(p_logprob, [chain_state.map_logprob], label=nothing, color=:gray)

        ac = alpha_chain(chain_state)
        p_alpha = plot(burn+1:length(ac), ac[burn+1:end], grid=:no, label=nothing, title="α chain")
        vline!(p_alpha, [chain_state.map_idx], label=nothing, color=:gray)

        muc = reduce(hcat, mu_chain(chain_state))'
        p_mu = plot(burn+1:size(muc, 1), muc[burn+1:end, :], grid=:no, label=nothing, title="μ chain")
        vline!(p_mu, [chain_state.map_idx], label=nothing, color=:gray)

        lc = lambda_chain(chain_state)
        p_lambda = plot(burn+1:length(lc), lc[burn+1:end], grid=:no, label=nothing, title="λ chain")
        vline!(p_lambda, [chain_state.map_idx], label=nothing, color=:gray)
        
        pc = flatten.(LowerTriangular.(psi_chain(chain_state)))
        pc = reduce(hcat, pc)'
        p_psi = plot(burn+1:size(pc, 1), pc[burn+1:end, :], grid=:no, label=nothing, title="Ψ chain")
        vline!(p_psi, [chain_state.map_idx], label=nothing, color=:gray)

        nc = nu_chain(chain_state)
        p_nu = plot(burn+1:length(nc), nc[burn+1:end], grid=:no, label=nothing, title="ν chain")
        vline!(p_nu, [chain_state.map_idx], label=nothing, color=:gray)

        sz = (1200, 1200)
        lo = @layout [a{0.5h} b; c d; e f; g h]
        p = plot(
        p_current, p_map, 
        p_logprob, p_alpha, 
        p_mu, p_lambda, 
        p_psi, p_nu,
        size=(1200, 1200), layout=lo)

        return p
    end

    function stats(chain_state::MNCRPchain; burn=0)
        println("MAP state")
        println(" log prob: $(chain_state.map_logprob)")
        println(" #cluster: $(length(chain_state.map_pi))")
        println("    alpha: $(chain_state.map_params.alpha)")
        println("       mu: $(chain_state.map_params.mu)")
        println("   lambda: $(chain_state.map_params.lambda)")
        println("      psi:")
        display(chain_state.map_params.psi)
        println("       nu: $(chain_state.map_params.nu)")
        println()
        println("Mean..")
        println(" #cluster: $(mean(chain_state.nbclusters_chain[burn+1:end]))")
        println("    alpha: $(mean(alpha_chain(chain_state)[burn+1:end]))")
        println("       mu: $(mean(mu_chain(chain_state)[burn+1:end]))")
        println("   lambda: $(mean(lambda_chain(chain_state)[burn+1:end]))")
        println("      psi:")
        display(mean(psi_chain(chain_state)[burn+1:end]))
        println("       nu: $(mean(nu_chain(chain_state)[burn+1:end]))")
        println()
    end
end