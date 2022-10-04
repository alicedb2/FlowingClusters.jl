module MultivariateNormalCRP
    using Distributions: logpdf, MvNormal, InverseWishart, Normal, Cauchy, Uniform, Binomial
    using Random: randperm, shuffle, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, Weights, std, percentile, quantile
    using LinearAlgebra: det, LowerTriangular, cholesky, diag, tr, diagm, inv, norm, eigen, svd
    using SpecialFunctions: loggamma, polygamma
    using Base.Iterators: cycle
    using ColorSchemes: Paired_12
    using Plots: plot, vline!, hline!, scatter!, @layout, grid, scalefontsizes, mm
    using StatsPlots: covellipse!
    import RecipesBase: plot
    
    export initiate_chain, advance_chain!, attempt_map!, reset_map!
    export log_Pgenerative, drawNIW, stats
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain
    export plot, covellipses!
    export local_average_covariance, wasserstein2_distance, wasserstein1_distance_bound

    global const Cluster::Type = Set{Vector{Float64}}

    mutable struct MNCRPhyperparams
        alpha::Float64
        mu::Vector{Float64}
        lambda::Float64
        flatL::Vector{Float64}
        L::LowerTriangular{Float64, Matrix{Float64}}
        psi::Matrix{Float64}
        nu::Float64

        accepted_alpha::Int64
        rejected_alpha::Int64

        accepted_mu::Vector{Int64}
        rejected_mu::Vector{Int64}

        accepted_lambda::Int64
        rejected_lambda::Int64

        accepted_flatL::Vector{Int64}
        rejected_flatL::Vector{Int64}

        accepted_nu::Int64
        rejected_nu::Int64

        accepted_split::Int64
        rejected_split::Int64

        accepted_merge::Int64
        rejected_merge::Int64

    end

    function MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu)
        d = size(mu, 1)
        sizeflatL = Int64(d * (d + 1) / 2)

        return MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu, 
        0, 0, 
        Array{Int64}(zeros(d)), Array{Int64}(zeros(d)), 
        0, 0, 
        Array{Int64}(zeros(sizeflatL)), Array{Int64}(zeros(sizeflatL)), 
        0, 0,
        0, 0,
        0, 0)
    end

    function MNCRPhyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, flatL::Vector{Float64}, nu::Float64)
        d = size(mu, 1)
        flatL_d = Int64(d * (d + 1) / 2)
        if size(flatL, 1) != flatL_d
            error("Dimension mismatch, flatL should have length $flatL_d")
        end

        L = foldflat(flatL)
        psi = L * L'
    
        return MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPhyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, L::LowerTriangular{Float64}, nu::Float64)
        d = size(mu, 1)
        if !(d == size(L, 1))
            error("Dimension mismatch, L should have dimension $d x $d")
        end
    
        psi = L * L'
        flatL = flatten(L)
    
        return MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPhyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)
        d = size(mu, 1)
        if !(d == size(psi, 1) == size(psi, 2))
            error("Dimension mismatch, L should have dimension $d x $d")
        end
    
        L = cholesky(psi).L
        flatL = flatten(L)
    
        return MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu)
    end

    function MNCRPhyperparams(d::Int64)
        return MNCRPhyperparams(1.0, zeros(d), 1.0, LowerTriangular(diagm(fill(0.1, d))), 1.0 * d)
    end

    function clear_diagnostics!(hyperparams::MNCRPhyperparams)
        d = size(hyperparams.mu, 1)
        sizeflatL = Int64(d * (d + 1) / 2)

        hyperparams.accepted_alpha = 0
        hyperparams.rejected_alpha = 0

        hyperparams.accepted_mu = Array{Int64}(zeros(d))
        hyperparams.rejected_mu = Array{Int64}(zeros(d))

        hyperparams.accepted_lambda = 0
        hyperparams.rejected_lambda = 0
        
        hyperparams.accepted_flatL = Array{Int64}(zeros(sizeflatL))
        hyperparams.rejected_flatL = Array{Int64}(zeros(sizeflatL))

        hyperparams.accepted_nu = 0
        hyperparams.rejected_nu = 0

        hyperparams.accepted_split = 0
        hyperparams.rejected_split = 0

        hyperparams.accepted_merge = 0
        hyperparams.rejected_merge = 0
        
    end

    mutable struct MNCRPchain

        # Current state of the partition over samples
        # from the Chinese Restaurant Process
        clusters::Vector{Cluster}
        # Current value of hyperparameters
        hyperparams::MNCRPhyperparams

        # Some chains of interests
        nbclusters_chain::Vector{Int64}
        hyperparams_chain::Vector{MNCRPhyperparams}
        logprob_chain::Vector{Float64}

        # Maximum a-posteriori state and location
        map_clusters::Vector{Cluster}
        map_hyperparams::MNCRPhyperparams
        map_logprob::Float64
        map_idx::Int64

    end

    alpha_chain(chain::MNCRPchain) = [p.alpha for p in chain.hyperparams_chain]
    mu_chain(chain::MNCRPchain) = [p.mu for p in chain.hyperparams_chain]
    mu_chain(chain::MNCRPchain, i) = [p.mu[i] for p in chain.hyperparams_chain]
    lambda_chain(chain::MNCRPchain) = [p.lambda for p in chain.hyperparams_chain]
    psi_chain(chain::MNCRPchain) = [p.psi for p in chain.hyperparams_chain]
    psi_chain(chain::MNCRPchain, i, j) = [p.psi[i, j] for p in chain.hyperparams_chain]
    nu_chain(chain::MNCRPchain) = [p.nu for p in chain.hyperparams_chain]
    
    function ij(flat_k::Int64)
        i = Int64(ceil(1/2 * (sqrt(1 + 8 * flat_k) - 1)))
        j = Int64(k - i * (i - 1)/2)
        return (i, j)
    end

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
        # Basically d is the "triangular root"
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


    function dimension(hyperparams::MNCRPhyperparams)
        @assert size(hyperparams.mu, 1) == size(hyperparams.psi, 1) == size(hyperparams.psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
        return size(hyperparams.mu, 1)
    end

    function flatL!(hyperparams::MNCRPhyperparams, value::Vector{Float64})
        d = dimension(hyperparams)
        if (2 * size(value, 1) != d * (d + 1))
            error("Dimension mismatch, value should have length $(Int64(d*(d+1)/2))")
        end
        hyperparams.flatL = copy(value) # Just making sure?
        hyperparams.L = foldflat(hyperparams.flatL)
        hyperparams.psi = hyperparams.L * hyperparams.L'
    end

    function L!(hyperparams::MNCRPhyperparams, value::T) where {T <: AbstractMatrix{Float64}}
        d = dimension(hyperparams)
        if !(d == size(value, 1) == size(value, 2))
            error("Dimension mismatch, value shoud have size $d x $d")
        end
        hyperparams.L = LowerTriangular(value)
        hyperparams.flatL = flatten(hyperparams.L)
        hyperparams.psi = hyperparams.L * hyperparams.L'
    end

    function psi!(hyperparams::MNCRPhyperparams, value::Matrix{Float64})
        d = dimension(hyperparams)
        if !(d == size(value, 1) == size(value, 2))
            error("Dimension mismatch, value should have size $d x $d")
        end
        hyperparams.L = cholesky(value).L
        hyperparams.flatL = flatten(hyperparams._L)
        hyperparams.psi = value
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

    function drawNIW(hyperparams::MNCRPhyperparams)
        return drawNIW(hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
    end

    # From conjugacy between NIW and MvNormal:
    # Update parameters of the normal-inverse-Wishart distribution
    # when joined with its data likelihood (mv normal distributions)
    # function updated_niw_hyperparams(cluster::Union{Nothing, Cluster}, 
    function updated_niw_hyperparams(cluster::Cluster, 
        mu::Vector{Float64}, 
        lambda::Float64, 
        psi::Matrix{Float64}, 
        nu::Float64
        )

        # @assert size(mu, 1) == size(psi, 1) == size(psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
  
        if isempty(cluster)

            return (mu, lambda, psi, nu)
            
        else
            
            d = size(mu, 1)
            n = length(cluster)

            lambda_c = lambda + n
            nu_c = nu + n

            # We are unfortunately going to have
            # to optimize the following
            #
            #     mean_x = mean(cluster)
            #     psi_c = (psi 
            #             + sum((x - mean_x) * (x - mean_x)' for x in cluster) 
            #             + lambda * n / (lambda + n) * (mean_x - mu) * (mean_x - mu)')
            #     mu_c = (lambda * mu + n * mean_x) / (lambda + n)

            mu_c = Array{Float64}(undef, d)
            psi_c = Array{Float64}(undef, d, d)
            
            X = Array{Float64}(undef, n, d)
            for (k, x) in enumerate(cluster)
                for i in 1:d
                    X[k, i] = x[i]
                end
            end

            # mean_x::Vector{Float64} = sum(X, dims=1)[1, :] / n
            mean_x = dropdims(sum(X, dims=1), dims=1) / n

            @inbounds for j in 1:d
                @inbounds for i in 1:j
                    psi_c[i, j] = psi[i, j] + lambda * n / (lambda + n) * (mean_x[i] - mu[i]) * (mean_x[j] - mu[j])
                    @inbounds @simd for k in 1:n
                        psi_c[i, j] += (X[k, i] - mean_x[i]) * (X[k, j] - mean_x[j])
                    end
                    psi_c[j, i] = psi_c[i, j]
                end
            end

            # mu_c = (lambda * mu + n * mean_x) / (lambda + n)
            for i in 1:d
                mu_c[i] = (lambda * mu[i] + n * mean_x[i]) / (lambda + n)
            end

            return (mu_c, lambda_c, psi_c, nu_c)
        
        end
    
    end

    function log_Zniw(
        cluster::Nothing,
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)::Float64

        empty_cluster = Cluster()

        return log_Zniw(empty_cluster, mu, lambda, psi, nu)
    
    end

    # Return the normalization constant 
    # of the normal-inverse-Wishart distribution,
    # possibly in the presence of data if cluster isn't empty
    function log_Zniw(
        cluster::Cluster,
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)::Float64
        
        d = size(mu, 1)

        mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
        
        lognum = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu / 2)
    
        logdenum = d/2 * log(lambda) + nu / 2 * log(det(psi))

        return lognum - logdenum
    
    end

    # Return the log-likelihood of the model
    function log_Pgenerative(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
    
        @assert all(length(c) > 0 for c in list_of_clusters)
        
        N = sum(length(c) for c in list_of_clusters)
        K = length(list_of_clusters)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)
    
        # Log-probability associated with the Chinese Restaurant Process
        log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum(loggamma(length(c)) for c in list_of_clusters)
        
        # Log-probability associated with the data likelihood
        # and Normal-Inverse-Wishart base distribution of the CRP
        log_niw = 0.0
        for cluster in list_of_clusters
            log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - length(cluster) * d/2 * log(2pi) 
        end

        # mu0 has a flat hyperpriors
        log_hyperpriors = 0.0
        # alpha hyperprior
        log_hyperpriors += log(jeffreys_alpha(alpha, N))
        # lambda hyperprior
        log_hyperpriors += -log(lambda)
        # psi hyperprior
        log_hyperpriors += -d * log(det(psi))
        # nu hyperprior
        log_hyperpriors += log(jeffreys_nu(nu, d))

        return log_crp + log_niw + log_hyperpriors
    
    end

    function add_to_state_gibbs!(element::Vector{Float64}, list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; greedy::Bool=false)

        # Clean up just in case to make sure
        # the only empty cluster will be the one from
        # the CRP with probability prop to alpha

        @assert all(!isempty(c) for c in list_of_clusters)

        element_set = Cluster([element])

        # for c in list_of_clusters
        #     @assert !in(element, c)
        # end
        # element is not in any cluster in list_of_clusters
        # but N includes all elements including that one
        Nminus1 = sum([length(c) for c in list_of_clusters])
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)
    
        crp_log_weights = [log(length(c)) - log(alpha + Nminus1) for c in list_of_clusters]    
            
        # niw_log_weights = [
        #         (
        #          log_Zniw(union(element_set, c), mu, lambda, psi, nu) 
        #         - log_Zniw(c, mu, lambda, psi, nu)
        #         - d/2 * log(2pi)
        #         ) 
        #     for c in list_of_clusters
        #     ]
        niw_log_weights = Array{Float64}(undef, length(list_of_clusters))
        for (i, cluster) in enumerate(list_of_clusters)
            # cluster <- cluster U {element}
            push!(cluster, element)
            niw_log_weights[i] = log_Zniw(cluster, mu, lambda, psi, nu)
            pop!(cluster, element)
            # back to cluster <- cluster \ {element}
            niw_log_weights[i] -= log_Zniw(cluster, mu, lambda, psi, nu) + d/2 * log(2pi)
        end
        
        # Add single empty cluster to potential choice with its associated weights
        empty_cluster = Cluster()
        push!(list_of_clusters, empty_cluster)
        push!(crp_log_weights, log(alpha) - log(alpha + Nminus1))
        push!(niw_log_weights, log_Zniw(element_set, mu, lambda, psi, nu) 
                             - log_Zniw(nothing, mu, lambda, psi, nu) 
                             - d/2 * log(2pi))

        unnorm_logPgen = [crp + niw for (crp, niw) in zip(crp_log_weights, niw_log_weights)]
        norm_logPgen = unnorm_logPgen .- logsumexp(unnorm_logPgen)
    
        if !greedy
            # Gibbs
            probs = Weights(exp.(norm_logPgen))
            selected_cluster = sample(list_of_clusters, probs)
        else
        # greedy Gibbs
            _, max_idx = findmax(norm_logPgen)
            selected_cluster = list_of_clusters[max_idx]
        end
        
        # N1 = sum(length(c) for c in list_of_clusters)
        # @assert N1 == Nminus1
        push!(selected_cluster, element)
        # N2 = sum(length(c) for c in list_of_clusters)
        # @assert N2 == N1 + 1 "$N1 $N2) "

        filter!(c -> length(c) > 0, list_of_clusters)
        
    end

    function advance_clusters_gibbs!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; greedy::Bool=false)
    
        elements = shuffle([el for cluster in list_of_clusters for el in cluster])
    
        for e in elements

            # This is just some complicated code to go with
            # that shuffle so that we can go over every
            # element exactly once even though they are
            # spread out in the sets in list_of_clusters
            # N = sum(length(c) for c in list_of_clusters)
            for (ci, c) in enumerate(list_of_clusters)
                if e in c
                    pop!(c, e)
                    if isempty(c)
                        deleteat!(list_of_clusters, ci)
                    end
                    break
                end
            end
            # @assert sum(length(c) for c in list_of_clusters) == N - 1

            add_to_state_gibbs!(e, list_of_clusters, hyperparams; greedy=greedy)   
        end
    
    end


    # Split-Merge Metropolis-Hastings with restricted Gibbs sampling from Jain & Neal 2004
    # (nothing wrong with a little ravioli code, 
    #  might refactor later, might not)
    function advance_clusters_JNrestrictedsplitmerge!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; t=5)
            
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

        d = size(mu, 1)
    
        # just to make explicit what's going on
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
        
            launch_Si = Cluster([i])

            # Avoid one more copy
            launch_Sj = list_of_clusters[cj]
            pop!(launch_Sj, i)
        
            # Remove cluster from the active state
            filter!(x -> !(x === launch_Sj), list_of_clusters)

        elseif ci != cj

            initial_Si = copy(list_of_clusters[ci])
            initial_Sj = copy(list_of_clusters[cj])

            # Avoid two more copies
            launch_Si = list_of_clusters[ci]
            launch_Sj = list_of_clusters[cj]
        
            filter!(x -> !(x === launch_Si || x === launch_Sj), list_of_clusters)
        
        end
    

        # Shuffle elements (except i and j) with equal probability
        # of being reassigned to either launch cluster
        for e in Smij
            
            # Make sure it's in neither
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
                
                # It's in one of them.
                # Make sure it's in neither.
                delete!(launch_Si, e)
                delete!(launch_Sj, e)
            
                # singleton_e = Cluster([e])
            
                #############

                # CRP contribution
                log_weight_Si = log(length(launch_Si))

                # log_weight_Si += log_Zniw(union(singleton_e, launch_Si), mu, lambda, psi, nu) 
                push!(launch_Si, e)
                log_weight_Si += log_Zniw(launch_Si, mu, lambda, psi, nu) 
                pop!(launch_Si, e)

                log_weight_Si -= log_Zniw(launch_Si, mu, lambda, psi, nu)
                log_weight_Si -= d/2 * log(2pi)

                #############

                # CRP contribution
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

            @assert !isempty(launch_Si) && !isempty(launch_Sj)
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

                # singleton_e = Cluster([e])

                #############

                # restricted CRP contribution
                log_weight_Si = log(length(proposed_Si))

                # log_weight_Si += log_Zniw(union(singleton_e, proposed_Si), mu, lambda, psi, nu) 
                push!(proposed_Si, e)
                log_weight_Si += log_Zniw(proposed_Si, mu, lambda, psi, nu) 
                pop!(proposed_Si, e)

                log_weight_Si -= log_Zniw(proposed_Si, mu, lambda, psi, nu)
                # There is no Z0 in the denominator of the predictive posterior
                log_weight_Si -= d/2 * log(2pi) # Just for the sake of explicitness, will cancel out
            
                #############

                # restricted CRP contribution
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
            # It's possible the split proposal didn't
            # actually suggest the same merge state,
            # so check for that
            # if n_proposed_Si > 0 && n_proposed_Sj > 0
            #     log_crp_ratio = log(alpha) + loggamma(n_proposed_Si) + loggamma(n_proposed_Sj) - loggamma(n_initial_S)
            # else
            #     log_crp_ratio = 0.0
            # end

            # It might also be unnecessary, because
            # in that case the move is always "accepted"
            # because both rejecting or accepting the move
            # lead to the same outcome. The bias
            # introduced by log(alpha) is in other words
            # inconsequential
        
            # The Hastings ratio is 
            # q(initial | proposed) / q(proposed | initial)
            # Here q(initial | proposed) = 1 
            # because the initial state is a merge state with its ci = cj
            # and the proposed state is a split state, so the proposal process
            # would be a merge proposal and there is only one way to do so.
        
            # Note also that q(proposed | initial) = q(proposed | launch)
            # because every initial states gets shuffled and forgotten
            # when forming a random launch state by restricted Gibbs moves
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
                push!(list_of_clusters, proposed_Si, proposed_Sj)
                hyperparams.accepted_split += 1
            else
                push!(list_of_clusters, initial_S)
                hyperparams.rejected_split += 1
            end
        
            filter!(c -> length(c) > 0, list_of_clusters)

        ##############
        ### Merge! ###
        ##############
        elseif ci != cj
            
            # We are calculating
            # q(initial | merge) = q(initial | launch)
            # and we remember that 
            # q(merge | initial) = q(merge | launch) = 1
            log_q_initial_from_launch = 0.0 
        
            proposed_S = union(launch_Si, launch_Sj)
        
            for e in sample(Smij, length(Smij), replace=false)

                delete!(launch_Si, e)
                delete!(launch_Sj, e)

                # singleton_e = Cluster([e])

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
                # but faster (no set creation and copy by union())
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
                push!(list_of_clusters, proposed_S)
                hyperparams.accepted_merge += 1
            else
                push!(list_of_clusters, initial_Si, initial_Sj)
                hyperparams.rejected_merge += 1
            end

            filter!(c -> length(c) > 0, list_of_clusters)

        end
    
    end

    function advance_alpha!(
        list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
        step_type="gaussian", step_scale=0.4)
            
        # No Cauchy because it's a very bad idea on a log scale
        if step_type == "gaussian"
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end

        N = sum(length(c) for c in list_of_clusters)

        alpha = hyperparams.alpha
    
        # 1/x improper hyperprior on alpha
        proposed_logalpha = log(alpha) + rand(step_distrib)
        proposed_alpha = exp(proposed_logalpha)
        
        log_acc = 0.0

        # because we propose moves on the log scale
        # but need them uniform over alpha > 0
        log_hastings = proposed_logalpha - log(alpha)
        log_acc += log_hastings

        log_acc += length(list_of_clusters) * log(proposed_alpha) - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
        log_acc -= length(list_of_clusters) * log(alpha) - loggamma(alpha + N) + loggamma(alpha)

        log_acc += log(jeffreys_alpha(proposed_alpha, N)) - log(jeffreys_alpha(alpha, N))

        log_acc = min(0.0, log_acc)

        if log(rand()) < log_acc
            hyperparams.alpha = proposed_alpha
            hyperparams.accepted_alpha += 1
        else
            hyperparams.rejected_alpha += 1
        end
    
    end

    function advance_mu!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                         random_order=true, step_scale=0.3, step_type="gaussian")
    
        if step_type == "cauchy"
            step_distrib = Cauchy(0.0, step_scale)
        elseif step_type == "gaussian"
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end

        lambda, psi, nu = hyperparams.lambda, hyperparams.psi, hyperparams.nu
        
        d = size(hyperparams.mu, 1)
        
        if random_order
            dim_order = randperm(d)
        else
            dim_order = 1:d
        end    
                
        for i in dim_order
            proposed_mu = copy(hyperparams.mu)
            proposed_mu[i] = proposed_mu[i] + rand(step_distrib)

            log_acc = sum(log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(nothing, proposed_mu, lambda, psi, nu)
                        - log_Zniw(c, hyperparams.mu, lambda, psi, nu) + log_Zniw(nothing, hyperparams.mu, lambda, psi, nu) 
                     for c in list_of_clusters)
        
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                hyperparams.mu = proposed_mu
                hyperparams.accepted_mu[i] += 1
            else
                hyperparams.rejected_mu[i] += 1
            end
            
        end
            
    end

    function advance_lambda!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                             step_type="gaussian", step_scale=0.4)

        if step_type == "gaussian"
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
    
        proposed_loglambda = log(lambda) + rand(step_distrib)
        proposed_lambda = exp(proposed_loglambda)
        
        log_acc = sum(log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(nothing, mu, proposed_lambda, psi, nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                     for c in list_of_clusters)

        # We leave loghastings = 0.0 because apparently the
        # Jeffreys prior over lambda is the logarithmic prior
        # and moves are symmetric on the log scale.

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            hyperparams.lambda = proposed_lambda
            hyperparams.accepted_lambda += 1
        else
            hyperparams.rejected_lambda += 1
        end
    
    end

    function advance_psi!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                          random_order=true, step_scale=0.1, step_type="gaussian")
    
        if step_type == "cauchy"
            step_distrib = Cauchy(0.0, step_scale)
        elseif step_type == "gaussian"
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end
    
        flatL_d = size(hyperparams.flatL, 1)
        
        if random_order
            dim_order = randperm(flatL_d)
        else
            dim_order = 1:flatL_d
        end
    
        mu, lambda, nu = hyperparams.mu, hyperparams.lambda, hyperparams.nu
    
        d = size(mu, 1)
            
        for k in dim_order
            
            proposed_flatL = copy(hyperparams.flatL)
        
            proposed_flatL[k] = proposed_flatL[k] + rand(step_distrib)
        
            proposed_L = foldflat(proposed_flatL)
            proposed_psi = proposed_L * proposed_L'
        
            log_acc = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                        - log_Zniw(cluster, mu, lambda, hyperparams.psi, nu) + log_Zniw(nothing, mu, lambda, hyperparams.psi, nu) 
                        for cluster in list_of_clusters)
                
            # Go from symmetric and uniform in L to uniform in psi
            # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
            # 2^d's cancel in the Hastings ratio
            log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(hyperparams.L)))))
            log_acc += log_hastings

            log_acc += d * (log(det(hyperparams.psi)) - log(det(proposed_psi)))
            
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                flatL!(hyperparams, proposed_flatL)
                hyperparams.accepted_flatL[k] += 1
            else
                hyperparams.rejected_flatL[k] += 1
            end
            
        end
    
    end

    function _advance_psi!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                          random_order=true, step_scale=0.1)
    
        mu, lambda, nu = hyperparams.mu, hyperparams.lambda, hyperparams.nu
        d = size(mu, 1)

        size_flatL = Int64(d * (d + 1) / 2)

        step_distrib = Normal(0.0, step_scale)    
        
        
        proposed_flatL = copy(hyperparams.flatL) .+ rand(step_distrib, size_flatL)
        proposed_L = foldflat(proposed_flatL)
        proposed_psi = proposed_L * proposed_L'
        
        log_acc = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                    - log_Zniw(cluster, mu, lambda, hyperparams.psi, nu) + log_Zniw(nothing, mu, lambda, hyperparams.psi, nu) 
                    for cluster in list_of_clusters)
                
        log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(hyperparams.L)))))
        log_acc += log_hastings

        log_acc += d * (log(det(hyperparams.psi)) - log(det(proposed_psi)))
            
        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            flatL!(hyperparams, proposed_flatL)
            for i in 1:size_flatL
                hyperparams.accepted_flatL[i] += 1
            end
        else
            for i in 1:size_flatL
                hyperparams.rejected_flatL[i] += 1
            end
        end
                
    end

    function advance_nu!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                         step_type="gaussian", step_scale=0.2)

        if step_type == "gaussian"
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == "uniform"
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end
    
        d = size(hyperparams.mu, 1)
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

        # x = nu - (d - 1)
        current_logx = log(nu - (d - 1))
        proposed_logx = current_logx + rand(step_distrib)
        proposed_nu = d - 1 + exp(proposed_logx)
    
        log_acc = sum(log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(nothing, mu, lambda, psi, proposed_nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                    for c in list_of_clusters)
        
        # Convert back to uniform moves on the positive real line nu > d - 1
        log_hastings = proposed_logx - current_logx
        log_acc += log_hastings

        log_acc += log(jeffreys_nu(proposed_nu, d)) - log(jeffreys_nu(nu, d))

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            hyperparams.nu = proposed_nu
            hyperparams.accepted_nu += 1
        else
            hyperparams.rejected_nu += 1
        end
    
    end

    function jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha - polygamma(1, alpha) + polygamma(1, alpha + n))

    end

    # Cute result but unfortunately lead to 
    # improper a-posteriori probability in nu and psi
    function jeffreys_nu(nu::Float64, d::Int64)

        return sqrt(1/4 * sum(polygamma.(1, nu/2 .+ 1/2 .- 1/2 * (1:d))) - d/2/nu)

    end


    function advance_full_sequential_gibbs!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; greedy::Bool=false)

        # data = Vector{Float64}[datum for cluster in list_of_clusters for datum in cluster]
        # data = data[randperm(size(data, 1))]
        data = shuffle([element for cluster in list_of_clusters for element in cluster])

        empty!(list_of_clusters)

        for element in data
            add_to_state_gibbs!(element, list_of_clusters, hyperparams; greedy=greedy)
        end

    end

    function initiate_chain(data::Vector{Vector{Float64}})

        @assert all(size(e, 1) == size(data[1], 1) for e in data)

        d = size(data[1], 1)
        n = length(data)

        hyperparams = MNCRPhyperparams(d)

        chain_state = MNCRPchain([], hyperparams, [], [], [], [], hyperparams, -Inf, 1)

        chain_state.hyperparams_chain = [hyperparams]
        
        
        ##### 1st initialization method: fullseq
        chain_state.clusters = [Set{Vector{Float64}}(data)]
        advance_full_sequential_gibbs!(chain_state.clusters, chain_state.hyperparams)

        ####### 2nd initialization method: marriage problem
        # best_yet = -Inf
        # nb_test_suitors = 10
        # for i in Iterators.countfrom(1, 1)
        #     print("\rSuitor $(i)")
        #     advance_full_sequential_gibbs!(chain_state.clusters, chain_state.hyperparams)
        #     logprob = log_Pgenerative(chain_state.clusters, chain_state.hyperparams)
        #     if logprob > best_yet
        #         best_yet = logprob
        #         if i > nb_test_suitors
        #             println("!")
        #             break
        #         end
        #     end
        # end
        
        ####### 3rd initialization method: N clusters
        # chain_state.clusters = [Set{Vector{Float64}}([datum]) for datum in data]

        ####### 4rd initialization method: 1 cluster
        # chain_state.clusters = [Set{Vector{Float64}}(data)]
        
        ####### 5th initialization: K clusters
        # K = 3
        # chain_state.clusters = [Set{Vector{Float64}}(p)
        #                         for p in Iterators.partition(data, ceil(length(data)/K))]
        ##########################


        chain_state.nbclusters_chain = [length(chain_state.clusters)]

        chain_state.map_clusters = deepcopy(chain_state.clusters)
        lp = log_Pgenerative(chain_state.clusters, chain_state.hyperparams)
        chain_state.map_logprob = lp
        chain_state.logprob_chain = [lp]
        # map_hyperparams=hyperparams and map_idx=1 have already been 
        # specified when calling MNCRPchain

        chain_state.logprob_chain = [chain_state.map_logprob]

        return chain_state

    end

    
    function advance_chain!(chain_state::MNCRPchain; nb_steps=100,
        nb_splitmerge=5, splitmerge_t=2, nb_gibbs=1, nb_hyperparamsmh=10, fullseq_prob=0.0)
        
        _nb_splitmerge = div(nb_splitmerge, 2)
        # Used for printing stats #
        hp = chain_state.hyperparams

        last_accepted_split = hp.accepted_split
        last_rejected_split = hp.rejected_split
        split_total = 0

        last_accepted_merge = hp.accepted_merge
        last_rejected_merge = hp.rejected_merge
        merge_total = 0
        
        fullseq_total = 0
        
        map_attempt_total = 0
        ###########################

        for step in 1:nb_steps

            ## Large moves ##

            # This one might be biasing away
            # from the stationary distribution
            if rand() < fullseq_prob
                
                advance_full_sequential_gibbs!(chain_state.clusters, chain_state.hyperparams)
                fullseq_total += 1
            end

            # Split-merge
            for i in 1:_nb_splitmerge
                advance_clusters_JNrestrictedsplitmerge!(chain_state.clusters, chain_state.hyperparams, t=splitmerge_t)
            end

            _nb_splitmerge += rand([-3, -1, 0, 1, 3])
            _nb_splitmerge = max(0, min(nb_splitmerge, _nb_splitmerge))
            
            
            #################
        
            # Gibbs sweep
            for i in 1:nb_gibbs
                advance_clusters_gibbs!(chain_state.clusters, chain_state.hyperparams; greedy=false)
            end

            push!(chain_state.nbclusters_chain, length(chain_state.clusters))

            # print("  #cl:$(length(chain_state.clusters))"); flush(stdout)


            # Metropolis-Hastings moves over each parameter
            # step_scale is adjusted to roughly hit
            # an #accepted:#rejected ratio of 1
            # It was tuned with standardized data
            # namelyby subtracting the mean and dividing
            # by the standard deviation along each dimension.
            for i in 1:nb_hyperparamsmh
                advance_alpha!(chain_state.clusters, chain_state.hyperparams, 
                                step_type="gaussian", step_scale=0.4)
                
                advance_mu!(chain_state.clusters, chain_state.hyperparams, 
                            step_type="gaussian", step_scale=0.3)
                
                advance_lambda!(chain_state.clusters, chain_state.hyperparams, 
                                step_type="gaussian", step_scale=0.4)

                advance_psi!(chain_state.clusters, chain_state.hyperparams,
                             step_type="gaussian", step_scale=0.1)

                advance_nu!(chain_state.clusters, chain_state.hyperparams, 
                            step_type="gaussian", step_scale=0.2)
            end

            push!(chain_state.hyperparams_chain, deepcopy(chain_state.hyperparams))

            # Stats #
            print("\r$(step) -> $(nb_steps)"); flush(stdout)
            print("     ")
            print("s:$(hp.accepted_split - last_accepted_split)/$(hp.rejected_split - last_rejected_split)")
            print(" m:$(hp.accepted_merge - last_accepted_merge)/$(hp.rejected_merge - last_rejected_merge)")
            split_total += hp.accepted_split - last_accepted_split
            merge_total += hp.accepted_merge - last_accepted_merge
            print("  spc:$(round(split_total/step, digits=1))")
            print("  mpc:$(round(merge_total/step, digits=1))")
            if fullseq_prob > 0.0
                print("  f:$(fullseq_total)")
            end
            print("  #cl:$(length(chain_state.clusters))"); 
            last_accepted_split = hp.accepted_split
            last_rejected_split = hp.rejected_split
            last_accepted_merge = hp.accepted_merge
            last_rejected_merge = hp.rejected_merge
            ########################
    
            # logprob
            logprob = log_Pgenerative(chain_state.clusters, chain_state.hyperparams)
            push!(chain_state.logprob_chain, logprob)

            # MAP
            history_length = 1000
            short_logprob_chain = chain_state.logprob_chain[max(1, end - history_length):end]
            
            if logprob > quantile(short_logprob_chain, 0.95)

                # Summit attempt
                map_attempt_total += 1
                
                attempt_success = attempt_map!(chain_state, nb_pushes=5)
                
                if attempt_success

                    print("  a:$(map_attempt_total)  !  map#cl:$(length(chain_state.map_clusters))");
                    println()
                    flush(stdout)                

                else

                    print("  a:$(map_attempt_total)         ")

                end

            else
            
                print("  a:$(map_attempt_total)         ")

            end
            
            flush(stdout)

        end
    end

    function attempt_map!(chain_state::MNCRPchain; nb_pushes=5)
            
        map_clusters_attempt = deepcopy(chain_state.clusters)
        # Greedy Gibbs!
        for p in 1:nb_pushes
            advance_clusters_gibbs!(map_clusters_attempt, chain_state.hyperparams; greedy=true)
        end
        attempt_logprob = log_Pgenerative(map_clusters_attempt, chain_state.hyperparams)

        if attempt_logprob > chain_state.map_logprob
            chain_state.map_logprob = attempt_logprob
            chain_state.map_clusters = deepcopy(map_clusters_attempt)
            chain_state.map_hyperparams = deepcopy(chain_state.hyperparams)
            chain_state.map_idx = lastindex(chain_state.logprob_chain)
            return true
        else
            return false
        end
    end

    function reset_map!(chain_state::MNCRPchain)
        chain_state.map_logprob = log_Pgenerative(chain_state.clusters, chain_state.hyperparams)
        chain_state.map_clusters = deepcopy(chain_state.clusters)
        chain_state.map_hyperparams = deepcopy(chain_state.hyperparams)
        chain_state.map_idx = lastindex(chain_state.logprob_chain)
    end

    function plot(clusters::Vector{Cluster}; plot_kw...)
        
        p = plot(
            legend_position=:topleft, grid=:no, 
            showaxis=:no, ticks=:true;
            plot_kw...)

        clusters = sort(clusters, by=x -> length(x), rev=true)
        for (cluster, color) in zip(clusters, cycle(Paired_12))
            scatter!(collect(Tuple.(cluster)), label="$(length(cluster))", 
            color=color, markerstrokewidth=0)
        end
        
        # display(p)
        return p

    end

    function covellipses!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; n_std=2, scalematrix=nothing, offset=nothing, mode=false, lowest_weight=nothing, plot_kw...)

        mu0, lambda0, psi0, nu0 = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu0, 1)

        for c in clusters
            if lowest_weight === nothing || lowest_weight <= length(c)
                mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(c, mu0, lambda0, psi0, nu0)
                
                if mode
                    # Sigma mode of the posterior
                    sigma_c = psi_c / (nu_c + d + 1)
                else
                    # Average sigma of the posterior
                    sigma_c = psi_c / (nu_c - d - 1)
                end

                if !(scalematrix === nothing)
                    mu_c = inv(scalematrix) * mu_c
                    sigma_c = inv(scalematrix) * sigma_c * inv(scalematrix)'
                end

                if !(offset === nothing)
                    mu_c += offset
                end

                covellipse!(mu_c, sigma_c; n_std=n_std, legend=nothing, plot_kw...)
            end
        end
    end

    function plot(chain_state::MNCRPchain; burn=0)

        if size(chain_state.hyperparams.mu, 1) == 2
            p_map = plot(chain_state.map_clusters, title="MAP state ($(length(chain_state.map_clusters)) clusters)")
            p_current = plot(chain_state.clusters, title="Current state ($(length(chain_state.clusters)) clusters)", legend=false)
        end
        
        lpc = chain_state.logprob_chain
        p_logprob = plot(burn+1:length(lpc), lpc[burn+1:end], grid=:no, label=nothing, title="log probability chain")
        vline!(p_logprob, [chain_state.map_idx], label=nothing, color=:black)
        hline!(p_logprob, [chain_state.map_logprob], label=nothing, color=:black)

        ac = alpha_chain(chain_state)
        p_alpha = plot(burn+1:length(ac), ac[burn+1:end], grid=:no, label=nothing, title="α chain")
        vline!(p_alpha, [chain_state.map_idx], label=nothing, color=:black)

        muc = reduce(hcat, mu_chain(chain_state))'
        p_mu = plot(burn+1:size(muc, 1), muc[burn+1:end, :], grid=:no, label=nothing, title="μ₀ chain")
        vline!(p_mu, [chain_state.map_idx], label=nothing, color=:black)

        lc = lambda_chain(chain_state)
        p_lambda = plot(burn+1:length(lc), lc[burn+1:end], grid=:no, label=nothing, title="λ₀ chain")
        vline!(p_lambda, [chain_state.map_idx], label=nothing, color=:black)
        
        pc = flatten.(LowerTriangular.(psi_chain(chain_state)))
        pc = reduce(hcat, pc)'
        p_psi = plot(burn+1:size(pc, 1), pc[burn+1:end, :], grid=:no, label=nothing, title="Ψ₀ chain")
        vline!(p_psi, [chain_state.map_idx], label=nothing, color=:black)

        nc = nu_chain(chain_state)
        p_nu = plot(burn+1:length(nc), nc[burn+1:end], grid=:no, label=nothing, title="ν₀ chain")
        vline!(p_nu, [chain_state.map_idx], label=nothing, color=:black)

        nbc = chain_state.nbclusters_chain
        p_nbc = plot(burn+1:length(nbc), nbc[burn+1:end], grid=:no, label=nothing, title="#cluster chain")
        vline!(p_nbc, [chain_state.map_idx], label=nothing, color=:black)

        empty_plot = plot(legend=false, grid=false, foreground_color_subplot=:white)

        lo = @layout [a{0.4h} b; c d; e f; g h; i j]
        p = plot(
        p_map, p_current, 
        p_logprob, p_nbc, 
        p_mu, p_lambda, 
        p_psi, p_nu, p_alpha, empty_plot,
        size=(1500, 1500), layout=lo)

        return p
    end

    function stats(chain_state::MNCRPchain; burn=0)
        println("MAP state")
        println(" log prob: $(chain_state.map_logprob)")
        println(" #cluster: $(length(chain_state.map_clusters))")
        println("    alpha: $(chain_state.map_hyperparams.alpha)")
        println("       mu: $(chain_state.map_hyperparams.mu)")
        println("   lambda: $(chain_state.map_hyperparams.lambda)")
        println("      psi:")
        display(chain_state.map_hyperparams.psi)
        println("       nu: $(chain_state.map_hyperparams.nu)")
        println()


        nbc = chain_state.nbclusters_chain[burn+1:end]
        ac = alpha_chain(chain_state)[burn+1:end]
        muc = mu_chain(chain_state)[burn+1:end]
        lc = lambda_chain(chain_state)[burn+1:end]
        psic = psi_chain(chain_state)[burn+1:end]
        nc = nu_chain(chain_state)[burn+1:end]

        println("Mean..")
        println(" #cluster: $(mean(nbc)) [$(percentile(nbc, 25)), $(percentile(nbc, 75))])")
        println("    alpha: $(mean(ac)) ± $(std(ac))")
        println("       mu: $(mean(muc)) ± $(std(muc))")
        println("   lambda: $(mean(lc)) ± $(std(lc))")
        println("      psi:")
        display(mean(psic))
        println("       nu: $(mean(nc)) ± $(std(nc))")
        println()
    end

    function local_average_covariance(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)

        coordinate_set = Set(Vector{Float64}[coordinate])

        N = sum(length(c) for c in clusters)

        log_parts = []
        psi_c_parts = []

        clusters = vcat(clusters, [Cluster()])

        for c in clusters

            c_union_coordinate = union(coordinate_set, c)

            log_crp_weight = (length(c) > 0 ? log(length(c)) : log(alpha)) - log(alpha + N)

            log_data_weight = (log_Zniw(c_union_coordinate, mu, lambda, psi, nu) 
                               - log_Zniw(c, mu, lambda, psi, nu) 
                               - d/2 * log(2pi))

            _, _, psi_c, nu_c = updated_niw_hyperparams(c_union_coordinate, mu, lambda, psi, nu)
            
            push!(log_parts, log_crp_weight + log_data_weight)

            push!(psi_c_parts, psi_c / (nu_c - d - 1))

        end

        log_parts = log_parts .- logsumexp(log_parts)
        
        average_psi = sum(psi_c_parts .* exp.(log_parts))

        return average_psi
    end

    function wasserstein2_distance(mu1, sigma1, mu2, sigma2)
        return sqrt(norm(mu1 - mu2)^2 + tr(sigma1 + sigma2 - 2 * sqrt(sqrt(sigma2) * sigma1 * sqrt(sigma2))))
    end

    function wasserstein1_distance_bound(mu1, sigma1, mu2, sigma2)
        
        lambda1, V1 = eigen(sigma1)
        lambda2, V2 = eigen(sigma2)

        d = size(mu1, 1)

        W1 = norm(mu1 - mu2)
        W1 += sqrt(sum((sqrt(lambda1[i]) - sqrt(lambda2[i]))^2 + 2 * sqrt(lambda1[i] * lambda2[i]) * (1 - V1[:, i]' * V2[:, i]) for i in 1:d))

        return W1
    end
end