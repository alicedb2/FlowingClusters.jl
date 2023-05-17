module MultivariateNormalCRP
    using Distributions: MvNormal, MvTDist, InverseWishart, Normal, Cauchy, Uniform, Exponential, Dirichlet, Multinomial, Beta
    using Random: randperm, shuffle, shuffle!, seed!
    using StatsFuns: logsumexp, logmvgamma, logit, logistic
    using StatsBase: sample, mean, var, Weights, std, cov, percentile, quantile, median, iqr, scattermat, fit, Histogram, autocor, autocov
    using LinearAlgebra: logdet, det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd, I, diagind, dot
    using SpecialFunctions: loggamma, polygamma, logbeta
    using Base.Iterators: cycle
    import Base.Iterators: flatten
    using ColorSchemes: Paired_12, tableau_20
    using Plots: plot, plot!, vline!, hline!, scatter!, @layout, grid, scalefontsizes, mm
    using StatsPlots: covellipse!, covellipse
    using JLD2
    using ProgressMeter
    using FiniteDiff: finite_difference_hessian
    using Optim: optimize, minimizer, LBFGS, NelderMead, Options
    using DataStructures: CircularBuffer
    using FStrings
    import MCMCDiagnosticTools: ess_rhat

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, empty!
    import Base: iterate, deepcopy, copy, sort, in, first
    # import Base: show, split

    export updated_niw_hyperparams, updated_mvstudent_params
    export initiate_chain, advance_chain!, attempt_map!, burn!, advance_hyperparams!
    export log_Pgenerative, grad_log_Pgenerative
    export plot, covellipses!
    export project_clusters, project_cluster, project_hyperparams, project_mu, project_psi
    export local_covprec, local_covariance, local_precision, local_covariance_summary
    export eigen_mode, importances, local_geometry, map_cluster_assignment_idx
    # export response_correlation
    export tail_probability, tail_probability_summary, predictive_logpdf, clustered_tail_probability
    export presence_probability, presence_probability_summary
    export optimize_hyperparams, optimize_hyperparams!
    export optimal_step_scale_local
    export drawNIW
    
    include("types/diagnostics.jl")
    export Diagnostics
    export clear_diagnostics!, diagnostics

    include("types/hyperparams.jl")
    export MNCRPHyperparams, pack, unpack, ij, set_theta!, get_theta

    include("types/cluster.jl")
    export Cluster
    export elements

    include("types/chain.jl")
    export MNCRPChain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain, logprob_chain, nbclusters_chain, largestcluster_chain
    export ess_rhat, stats

    include("types/dataset.jl")
    using .Dataset
    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes, standardize_with, standardize_with!, split, standardize!

    include("naivebioclim.jl")
    using .NaiveBIOCLIM
    export bioclim_predictor

    include("helpers.jl")
    export performance_scores

    include("MNDPVariational.jl")

    # Quick and dirty but faster logdet
    # for positive-definite matrix
    function logdetpsd(A::AbstractMatrix{Float64})
        try
            chol = cholesky(Symmetric(A))
            # marginally faster than 
            # 2 * sum(log.(diag(chol.U)))
            acc = 0.0
            for i in 1:size(A, 1)
                acc += log(chol.U[i, i])
            end
            return 2 * acc
        catch e
            # dump(eigen(A))
            # throw(e)
            return -Inf
        end
    end

    function logdetflatLL(flatL::Vector{Float64})
        acc = 0.0
        i = 1
        delta = 2
        while i <= length(flatL)
           acc += log(flatL[i])
           i += delta
           delta += 1 
        end
        return 2 * acc
    end
    
    function drawNIW(
        mu::AbstractVector{Float64}, 
        lambda::Float64, 
        psi::AbstractMatrix{Float64}, 
        nu::Float64)::Tuple{Vector{Float64}, Matrix{Float64}}
    
        invWish = InverseWishart(nu, psi)
        sigma = rand(invWish)
    
        multNorm = MvNormal(mu, sigma/lambda)
        mu = rand(multNorm)
    
        return mu, sigma
    end

    # function drawNIW(hyperparams::MNCRPHyperparams)
    #     return drawNIW(hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
    # end

    function updated_niw_hyperparams(clusters::Cluster, hyperparams::MNCRPHyperparams)::Tuple{Vector{Float64}, Float64, Matrix{Float64}, Float64}
        return updated_niw_hyperparams(clusters, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
    end

    function updated_niw_hyperparams(cluster::Nothing, 
        mu::AbstractVector{Float64}, 
        lambda::Float64, 
        psi::AbstractMatrix{Float64}, 
        nu::Float64
        )::Tuple{Vector{Float64}, Float64, Matrix{Float64}, Float64}

        return (mu, lambda, psi, nu)

    end


    function updated_niw_hyperparams(cluster::Cluster, 
        mu::Vector{Float64}, 
        lambda::Float64, 
        psi::Matrix{Float64}, 
        nu::Float64
        )::Tuple{Vector{Float64}, Float64, Matrix{Float64}, Float64}

        # @assert length(mu) == size(psi, 1) == size(psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
  
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
                # mu_c[i] = (lambda * mu[i] + n * mean_x[i]) / (lambda + n)
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
    x
    end

    function log_Zniw(
        cluster::Nothing,
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)::Float64
        
        d = length(mu)
        log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
        log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi)
        return log_numerator - log_denominator

    end

    function log_Zniw(
        cluster::Nothing,
        mu::Vector{Float64},
        lambda::Float64,
        L::LowerTriangular{Float64},
        nu::Float64)::Float64
        
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
        mu::Vector{Float64},
        lambda::Float64,
        psi::Matrix{Float64},
        nu::Float64)::Float64
        
        d = length(mu)

        mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
        
        log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    
        log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi)
        return log_numerator - log_denominator
        # log_denominator = d/2 * log(lambda) + nu/2 * log(det(psi))
    end

    # Very cute!
    function jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

    end

    # Very cute as well!
    function jeffreys_nu(nu::Float64, d::Int64)

        return 1/2 * sqrt(sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

    end

    function log_Pgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; hyperpriors=true, temperature=1.0)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        return log_Pgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors, temperature=temperature)
    end

    # Theta is assumed to be a concatenated vector of coordinates
    # i.e. vcat(log(alpha), mu, log(lambda), flatL, log(nu -d + 1))
    function log_Pgenerative(clusters::Vector{Cluster}, theta::Vector{Float64}; hyperpriors=true, backtransform=true, jacobian=false, temperature=1.0)
        alpha, mu, lambda, flatL, L, psi, nu = pack(theta, backtransform=backtransform)
        log_p = log_Pgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors, temperature=temperature) 
        if jacobian
            d = length(mu)
            log_p += log(alpha)
            log_p += log(lambda)
            log_p += sum((d:-1:1) .* log.(abs.(diag(L))))
            log_p += log(nu - d + 1)
        end
        return log_p
    end

    # Return the log-likelihood of the model
    function log_Pgenerative(clusters::Vector{Cluster}, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; hyperpriors=true, temperature=1.0)

        @assert all(length(c) > 0 for c in clusters)
        
        N = sum([length(c) for c in clusters])
        K = length(clusters)
        d = length(mu)

        if alpha <= 0.0 || lambda <= 0.0 || nu <= d - 1 || !isfinite(logdetpsd(psi))
            return -Inf
        end

        # Log-probability associated with the Chinese Restaurant Process
        log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum([loggamma(length(c)) for c in clusters])
        
        # Log-probability associated with the data likelihood
        # and Normal-Inverse-Wishart base distribution of the CRP
        log_niw = 0.0
        for cluster in clusters
            log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - length(cluster) * d/2 * log(2pi)
        end
        log_niw -= K * log_Zniw(nothing, mu, lambda, psi, nu)

        log_hyperpriors = 0.0

        if hyperpriors
            # mu0 has a flat hyperpriors
            # alpha hyperprior
            log_hyperpriors += log(jeffreys_alpha(alpha, N))
            # lambda hyperprior
            log_hyperpriors += -log(lambda)
            # psi hyperprior
            log_hyperpriors += -d * logdetpsd(psi)
            # log_hyperpriors += -d * logdet(psi)
            # log_hyperpriors += -d * log(det(psi))
            # nu hyperprior
            log_hyperpriors += log(jeffreys_nu(nu, d))
        end

        log_p = log_crp + log_niw + log_hyperpriors
        return isfinite(log_p) ? log_p / temperature : -Inf
    
    end

    function grad_log_Pgenerative(clusters::Vector{Cluster}, storage::Vector{Float64}, theta::Vector{Float64}; hyperpriors=true, backtransform=true, jacobian=true)::Vector{Float64}
        g = grad_log_Pgenerative(clusters, theta, hyperpriors=true, backtransform=backtransform, jacobian=jacobian)
        storage[:] .= g
        println(storage)
        return storage
    end

    function grad_log_Pgenerative(clusters::Vector{Cluster}, theta::Vector{Float64}; hyperpriors=true, backtransform=true, jacobian=true)::Vector{Float64}
        alpha, mu, lambda, flatL, L, psi, nu = pack(theta, backtransform=backtransform)
        grad_logP = collect(grad_log_Pgenerative(clusters::Vector{Cluster}, alpha, mu, lambda, psi, nu, hyperpriors=hyperpriors))
        d = length(mu)

        grad_psi = grad_logP[4]
        grad_L = zeros(div(d * (d + 1), 2))
        # The Jacobian for L -> Psi is always included
        for k in 1:length(grad_L)
            m, n = ij(k)
            if m >= n
                for i in m:d
                    grad_L[k] += L[i, n] * grad_psi[i, m]
                end
                for j in 1:m
                    grad_L[k] += L[j, n] * grad_psi[m, j]
                end
            end

            if m == n && jacobian
                grad_L[k] += (d + 1 - m) * sign(flatL[k]) / abs(flatL[k])
            end
        end
    
        if jacobian
            grad_logP[1] += alpha * grad_logP[1]
            grad_logP[3] += lambda * grad_logP[3]
            grad_logP[5] += (nu - d + 1) * grad_logP[5]
        end
        
        return vcat(grad_logP[1], grad_logP[2], grad_logP[3], grad_L, grad_logP[5])
    end

    function grad_log_Pgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; hyperpriors=true)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        return grad_log_Pgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors)
    end

    function grad_log_Pgenerative(clusters::Vector{Cluster}, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; hyperpriors=true)
        
        d = length(mu)
        N = sum(length.(clusters))

        grad_alpha = length(clusters) / alpha + polygamma(0, alpha) - polygamma(0, alpha + N)

        grad_mu = zeros(d)
        grad_lambda = 0.0
        grad_psi = zeros(d, d)
        grad_nu = 0.0

        invpsi = inv(psi)
        for cluster in clusters
            n = length(cluster)
            mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
            
            invpsi_c = inv(psi_c)
            # invpsic_mm = psi_c \ (cluster.sum_x / n - mu)
            invpsic_mm = invpsi_c * (cluster.sum_x / n - mu)
            grad_mu += nu_c * lambda * n / lambda_c * invpsic_mm
            grad_lambda += d / 2 * (1 / lambda - 1 / lambda_c) - nu_c / 2 * n^2 / lambda_c^2 * (cluster.sum_x / n - mu)' * invpsic_mm
            
            grad_psi += 1/2 * (nu * invpsi - nu_c * invpsi_c) .* (2 .- I(d))

            grad_nu += 1/2 * sum([polygamma(0, nu_c/2 + (1 - j)/2) - polygamma(0, nu/2 + (1 - j)/2) for j in 1:d])
            grad_nu += 1/2 * (logdetpsd(psi) - logdetpsd(psi_c))
        end

        if hyperpriors
            num1 = (polygamma(1, alpha + N) - polygamma(1, alpha)) / alpha
            num2 = (polygamma(0, alpha + N) - polygamma(0, alpha)) / alpha^2
            num3 = polygamma(2, alpha + N) - polygamma(2, alpha)
            denum1 = (polygamma(0, alpha + N) - polygamma(0, alpha)) / alpha
            denum2 = polygamma(1, alpha + N) - polygamma(1, alpha)
            grad_alpha += 1/2 * (num1 - num2 + num3) / (denum1 + denum2)

            grad_lambda -= 1 / lambda

            grad_psi -= d * (2 .- I(d)) .* invpsi

            grad_nu += 1/4 * sum([polygamma(2, nu/2 + (1-j)/2) for j in 1:d])/sum([polygamma(1, nu/2 + (1-j)/2) for j in 1:d])
        end


        return grad_alpha, grad_mu, grad_lambda, grad_psi, grad_nu
    end


    function log_cluster_weight(element::Vector{Float64}, cluster::Cluster, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; N::Union{Int64, Nothing}=nothing)

        @assert !(element in cluster) "$(element)"
        
        d = length(mu)

        if isempty(cluster)
            log_weight = log(alpha) # - log(alpha + Nminus1)
        else
            log_weight = log(length(cluster)) # - log(alpha + Nminus1)
        end

        if !(N === nothing)
            log_weight -= log(alpha + N)
        end

        push!(cluster, element)
        log_weight += log_Zniw(cluster, mu, lambda, psi, nu)
        pop!(cluster, element)
        log_weight -= log_Zniw(cluster, mu, lambda, psi, nu)
        log_weight -= d/2 * log(2pi)
        
        return log_weight

    end 

    # function log_cluster_weights(element::Vector{Float64}, clusters::AbstractVector{Cluster}, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; add_empty=false)
    #     N = sum(length.(clusters))
    #     lcw = Float64[log_cluster_weight(element, cluster, alpha, mu, lambda, psi, nu) for cluster in clusters]
    #     d = length(mu)
    #     if add_empty
    #         push!(lcw, log_cluster_weight(element, Cluster(d), alpha, mu, lambda, psi, nu))
    #     end
    #     return lcw
    # end
    
    function advance_gibbs!(element::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; temperature=1.0)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)
            
        if sum(isempty.(clusters)) < 1
            push!(clusters, Cluster(d))
        end

        log_weights = zeros(length(clusters))
        for (i, cluster) in enumerate(clusters)
            log_weights[i] = log_cluster_weight(element, cluster, alpha, mu, lambda, psi, nu)
        end
        
        if temperature > 0.0
            unnorm_logp = log_weights / temperature
            norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
            probs = Weights(exp.(norm_logp))
            new_assignment = sample(clusters, probs)
        elseif temperature <= 0.0
            _, max_idx = findmax(log_weights)
            new_assignment = clusters[max_idx]
        end
        
        push!(new_assignment, element)

        filter!(!isempty, clusters)

    end

    function advance_gibbs!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; temperature=1.0)

        scheduled_elements = shuffle!(elements(clusters))
    
        for element in scheduled_elements
            
            pop!(clusters, element)
            advance_gibbs!(element, clusters, hyperparams, temperature=temperature)
            
            # pop!(clusters, e)
            
            # if sum(isempty.(clusters)) < 1
            #     push!(clusters, Cluster(d))
            # end

            # log_weights = zeros(length(clusters))
            # for (i, cluster) in enumerate(clusters)
            #     log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
            # end
            
            # if temperature > 0.0
            #     unnorm_logp = log_weights / temperature
            #     norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
            #     probs = Weights(exp.(norm_logp))
            #     new_assignment = sample(clusters, probs)
            # elseif temperature == 0.0
            #     _, max_idx = findmax(log_weights)
            #     new_assignment = clusters[max_idx]
            # end
            
            # push!(new_assignment, e)

        end

        # filter!(!isempty, clusters)

        return clusters
    
    end


    # Sequential splitmerge from Dahl & Newcomb
    function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; t=3, temperature=1.0)

        @assert t >= 1
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)
    
        cluster_indices = Tuple{Int64, Vector{Float64}}[(ce, e) for (ce, cluster) in enumerate(clusters) for e in cluster]

        (ci, ei), (cj, ej) = sample(cluster_indices, 2, replace=false)
    
        if ci == cj

            scheduled_elements = [e for e in clusters[ci] if !(e === ei) && !(e === ej)]
            initial_state = Cluster[clusters[ci]]
            deleteat!(clusters, ci)

        elseif ci != cj

            scheduled_elements = [e for e in flatten((clusters[ci], clusters[cj])) if !(e === ei) && !(e === ej)]
            initial_state = Cluster[clusters[ci], clusters[cj]]
            deleteat!(clusters, sort([ci, cj]))

        end

        shuffle!(scheduled_elements)
    
        proposed_state = Cluster[Cluster([ei]), Cluster([ej])]
        launch_state = Cluster[]

        log_q = 0.0

        for step in flatten((0:t, [:create_proposed_state]))
        
            
            # Keep copy of launch state
            # 
            if step == :create_proposed_state
                if ci == cj
                    # Do a last past to the proposed
                    # split state to accumulate
                    # q(proposed|launch)
                    # remember that
                    # q(launch|proposed) 
                    # = q(merged|some split launch state) = 1
                    launch_state = copy(proposed_state)
                elseif ci != cj
                    # Don't perform last step in a merge,
                    # keep log_q as the transition probability
                    # to the launch state, i.e.
                    # q(launch|proposed) = q(launch|launch-1)
                    launch_state = proposed_state
                    break
                end
            end

            log_q = 0.0

            for e in shuffle!(scheduled_elements)

                delete!(proposed_state, e)

                # Should be true by construction, just
                # making sure the construction is valid
                @assert all([!isempty(c) for c in proposed_state])
                @assert length(proposed_state) == 2

                #############        
            
                log_weights = zeros(length(proposed_state))
                for (i, cluster) in enumerate(proposed_state)
                    log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
                end

                if temperature > 0.0
                    unnorm_logp = log_weights / temperature
                    norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                    probs = Weights(exp.(norm_logp))
                    new_assignment, log_transition = sample(collect(zip(proposed_state, norm_logp)), 
                                                            probs)
                    
                    log_q += log_transition
                elseif temperature <= 0.0
                    _, max_idx = findmax(log_weights)
                    new_assignment = proposed_state[max_idx]
                    log_q += 0.0 # symbolic, transition is certain
                end
                
                push!(new_assignment, e)

            end

        end

        # At this point if we are doing a split state
        # then log_q = q(split*|launch)
        # and if we are doing a merge state
        # then log_q = q(launch|merge)=  q(launch|launch-1)

        if ci != cj
            # Create proposed merge state
            # The previous loop was only to get
            # q(launch|proposed) = q(launch|launch-1)
            # and at this point launch_state = proposed_state
            proposed_state = Cluster[Cluster(Vector{Float64}[e for cluster in initial_state for e in cluster])]
        elseif ci == cj
            # do nothing, we already have the proposed state
        end

        log_acceptance = (log_Pgenerative(proposed_state, hyperparams, hyperpriors=false) 
                        - log_Pgenerative(initial_state, hyperparams, hyperpriors=false))

        log_acceptance /= temperature

        # log_q is plus-minus the log-Hastings factor.
        # log_q already includes the tempering.
        if ci != cj
            log_acceptance += log_q
        elseif ci == cj
            log_acceptance -= log_q
        end

        log_acceptance = min(0.0, log_acceptance)

        if log(rand()) < log_acceptance
            append!(clusters, proposed_state)
            if ci != cj
                hyperparams.diagnostics.accepted_merge += 1
            elseif ci == cj
                hyperparams.diagnostics.accepted_split += 1
            end
        else
            append!(clusters, initial_state)
            if ci != cj
                hyperparams.diagnostics.rejected_merge += 1
            elseif ci == cj
                hyperparams.diagnostics.rejected_split += 1
            end
        end

        return clusters

    end

    function advance_alpha!(
        clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
        step_type=:gaussian, step_size=0.5)
            
        # No Cauchy because it's a very bad idea on a log scale
        if step_type == :gaussian
            step_distrib = Normal(0.0, step_size)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_size/2, step_size/2)
        end

        N = sum([length(c) for c in clusters])
        K = length(clusters)

        alpha = hyperparams.alpha
    
        # 1/x improper hyperprior on alpha
        proposed_logalpha = log(alpha) + rand(step_distrib)
        proposed_alpha = exp(proposed_logalpha)
        
        log_acc = 0.0

        # because we propose moves on the log scale
        # but need them uniform over alpha > 0
        # before feeding them to the hyperprior
        log_hastings = proposed_logalpha - log(alpha)
        log_acc += log_hastings

        log_acc += K * log(proposed_alpha) - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
        log_acc -= K * log(alpha) - loggamma(alpha + N) + loggamma(alpha)

        log_acc += log(jeffreys_alpha(proposed_alpha, N)) - log(jeffreys_alpha(alpha, N))

        log_acc = min(0.0, log_acc)

        if log(rand()) < log_acc
            hyperparams.alpha = proposed_alpha
            hyperparams.diagnostics.accepted_alpha += 1
        else
            hyperparams.diagnostics.rejected_alpha += 1
        end
    
    end

    function advance_mu!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                         random_order=true, step_size=0.25, step_type=:gaussian)
    
        if step_type == :cauchy
            step_distrib = Cauchy(0.0, step_size)
        elseif step_type == :gaussian
            step_distrib = Normal(0.0, step_size)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_size/2, step_size/2)
        end

        lambda, psi, nu = hyperparams.lambda, hyperparams.psi, hyperparams.nu
        
        d = length(hyperparams.mu)
        
        if random_order
            dim_order = randperm(d)
        else
            dim_order = 1:d
        end    
                
        for i in dim_order
            proposed_mu = deepcopy(hyperparams.mu)
            proposed_mu[i] = proposed_mu[i] + rand(step_distrib)

            log_acc = sum(log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(nothing, proposed_mu, lambda, psi, nu)
                        - log_Zniw(c, hyperparams.mu, lambda, psi, nu) + log_Zniw(nothing, hyperparams.mu, lambda, psi, nu) 
                     for c in clusters)
        
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                hyperparams.mu = proposed_mu
                hyperparams.diagnostics.accepted_mu[i] += 1
            else
                hyperparams.diagnostics.rejected_mu[i] += 1
            end
            
        end
            
    end

    function advance_lambda!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                             step_type=:gaussian, step_size=1.5)

        if step_type == :gaussian
            step_distrib = Normal(0.0, step_size)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_size/2, step_size/2)
        end
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
    
        proposed_loglambda = log(lambda) + rand(step_distrib)
        proposed_lambda = exp(proposed_loglambda)
        
        log_acc = sum(log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(nothing, mu, proposed_lambda, psi, nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                     for c in clusters)

        # We leave loghastings = 0.0 because the
        # Jeffreys prior over lambda is the logarithmic 
        # prior and moves are symmetric on the log scale.

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            hyperparams.lambda = proposed_lambda
            hyperparams.diagnostics.accepted_lambda += 1
        else
            hyperparams.diagnostics.rejected_lambda += 1
        end
    
    end

    function advance_psi!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                          random_order=true, step_size=0.1, step_type=:gaussian)
    
        if step_type == :cauchy
            step_distrib = Cauchy(0.0, step_size)
        elseif step_type == :gaussian
            step_distrib = Normal(0.0, step_size)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_size/2, step_size/2)
        end
    
        flatL_d = length(hyperparams.flatL)
        
        if random_order
            dim_order = randperm(flatL_d)
        else
            dim_order = 1:flatL_d
        end
    
        mu, lambda, nu = hyperparams.mu, hyperparams.lambda, hyperparams.nu
    
        d = length(mu)
            
        for k in dim_order
            
            proposed_flatL = deepcopy(hyperparams.flatL)
        
            proposed_flatL[k] = proposed_flatL[k] + rand(step_distrib)
        
            proposed_L = foldflat(proposed_flatL)
            proposed_psi = proposed_L * proposed_L'
        
            log_acc = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                        - log_Zniw(cluster, mu, lambda, hyperparams.psi, nu) + log_Zniw(nothing, mu, lambda, hyperparams.psi, nu) 
                        for cluster in clusters)
                
            # Go from symmetric and uniform in L to uniform in psi
            # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
            # 2^d's cancel in the Hastings ratio
            log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(hyperparams.L)))))
            log_acc += log_hastings

            log_acc += d * (logdetpsd(hyperparams.psi) - logdetpsd(proposed_psi))
            # log_acc += d * (log(det(hyperparams.psi)) - log(det(proposed_psi)))
            
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                flatL!(hyperparams, proposed_flatL)
                hyperparams.diagnostics.accepted_flatL[k] += 1
            else
                hyperparams.diagnostics.rejected_flatL[k] += 1
            end
            
        end
    
    end

    function advance_nu!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                         step_type=:gaussian, step_size=1.0)

        if step_type == :gaussian
            step_distrib = Normal(0.0, step_size)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_size/2, step_size/2)
        end
    
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        # x = nu - (d - 1)
        # we use moves on the log of x
        # so as to always keep nu > d - 1 
        current_logx = log(nu - (d - 1))
        proposed_logx = current_logx + rand(step_distrib)
        proposed_nu = d - 1 + exp(proposed_logx)
    
        log_acc = sum(log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(nothing, mu, lambda, psi, proposed_nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                    for c in clusters)
        
        # Convert back to uniform moves on the positive real line nu > d - 1
        log_hastings = proposed_logx - current_logx
        log_acc += log_hastings

        log_acc += log(jeffreys_nu(proposed_nu, d)) - log(jeffreys_nu(nu, d))

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            hyperparams.nu = proposed_nu
            hyperparams.diagnostics.accepted_nu += 1
        else
            hyperparams.diagnostics.rejected_nu += 1
        end
    
    end

    function optimal_step_scale_local(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; optimize=true, jacobian=false, verbose=false)::Matrix{Float64}
        
        if verbose
            print(" * Finding new step scale...")
        end
        
        if optimize
            hyperparams = optimize_hyperparams(clusters, hyperparams, verbose=verbose, jacobian=jacobian)
        end
        
        if verbose
            println("calculating Hessian...")
        end

        d = length(hyperparams.mu)
        D = 3 + d + div(d * (d + 1), 2)
        theta = unpack(hyperparams, transform=true)
        hessian = finite_difference_hessian(x -> log_Pgenerative(clusters, x; hyperpriors=true, jacobian=jacobian), theta)
        step_scale = -(2.38^2/D) * inv(hessian)
        evals, evec = eigen(step_scale)
        # We only care about the absolute curvature,
        # not its orientation. This allows us to 
        # compute a step_scale away from local optima of
        # log_Pgenerative.
        step_scale = evec * diagm(abs.(evals)) * evec'
        return (step_scale + step_scale') / 2 
    end

    # function optimal_step_scale_adaptive(chain::MNCRPChain; epsilon=1e-4)::Matrix{Float64}
    #     d = length(chain.hyperparams.mu)
    #     D = 3 + d + div(d * (d + 1), 2)
    #     emp_sigma = cov(unpack.(chain.hyperparams_chain))        
    #     return 2.38^2 / D * (emp_sigma + epsilon * I(D))
    # end

    function advance_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, step_scale::Matrix{Float64}; temperature=1.0)
        
        alpha, mu, lambda, flatL, L, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.flatL, hyperparams.L, hyperparams.psi, hyperparams.nu
        
        d = length(mu)
        # thetaD = 3 + d + div(d * (d + 1), 2)

        current_theta = unpack(hyperparams, transform=true)
        
        proposed_theta = rand(MvNormal(current_theta, step_scale))
        proposed_alpha, proposed_mu, proposed_lambda, proposed_flatL, proposed_L, proposed_psi, proposed_nu = pack(proposed_theta, backtransform=true)

        log_metropolis = log_Pgenerative(clusters, proposed_alpha, proposed_mu, proposed_lambda, proposed_psi, proposed_nu, hyperpriors=true)
        log_metropolis -= log_Pgenerative(clusters, alpha, mu, lambda, psi, nu, hyperpriors=true)

        log_hastings = log(proposed_alpha) - log(alpha)
        log_hastings += log(proposed_lambda) - log(lambda)
        log_hastings += sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
        log_hastings += log(proposed_nu - d + 1) - log(nu - d + 1)

        log_acc = min(0.0, log_metropolis / temperature + log_hastings)
        if log(rand()) <= log_acc
            hyperparams.alpha = proposed_alpha
            hyperparams.mu = proposed_mu
            hyperparams.lambda = proposed_lambda
            hyperparams.flatL = proposed_flatL
            hyperparams.L = proposed_L
            hyperparams.psi = proposed_psi
            hyperparams.nu = proposed_nu

            hyperparams.diagnostics.accepted_alpha += 1
            hyperparams.diagnostics.accepted_mu .+= 1
            hyperparams.diagnostics.accepted_lambda += 1
            hyperparams.diagnostics.accepted_flatL .+= 1
            hyperparams.diagnostics.accepted_nu += 1
        else
            hyperparams.diagnostics.rejected_alpha += 1
            hyperparams.diagnostics.rejected_mu .+= 1
            hyperparams.diagnostics.rejected_lambda += 1
            hyperparams.diagnostics.rejected_flatL .+= 1
            hyperparams.diagnostics.rejected_nu += 1
        end
    end

    function advance_hyperparams_algo6!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        
        di = hyperparams.diagnostics
        alpha, mu, lambda, flatL, L, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.flatL, hyperparams.L, hyperparams.psi, hyperparams.nu
        d = length(mu)
        D = 3 + d + div(d * (d + 1), 2)

        old_theta = unpack(hyperparams, transform=true)
        
        step_Lambdasqrt = diagm(exp.(di.step_loglambda/2))
        rescaled_step_sigma = step_Lambdasqrt * di.step_sigma * step_Lambdasqrt
        rescaled_step_sigma = (rescaled_step_sigma + rescaled_step_sigma') / 2

        proposed_Z = rand(MvNormal(zeros(D), rescaled_step_sigma))
        # pack(proposed_theta, backtransform=true)

        function log_acceptance_rate(proposed_theta)
            proposed_alpha, proposed_mu, proposed_lambda, proposed_flatL, proposed_L, proposed_psi, proposed_nu = pack(proposed_theta, backtransform=true)
            
            log_metropolis = log_Pgenerative(clusters, proposed_alpha, proposed_mu, proposed_lambda, proposed_psi, proposed_nu, hyperpriors=true)            
            log_metropolis -= log_Pgenerative(clusters, alpha, mu, lambda, psi, nu, hyperpriors=true)
    
            log_hastings = log(proposed_alpha) - log(alpha)
            log_hastings += log(proposed_lambda) - log(lambda)
            log_hastings += sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
            log_hastings += log(proposed_nu - d + 1) - log(nu - d + 1)
            return min(0.0, log_metropolis + log_hastings)
        end

        if log(rand()) < log_acceptance_rate(old_theta + proposed_Z)
            accepted_alpha, accepted_mu, accepted_lambda, accepted_flatL, accepted_L, accepted_psi, accepted_nu = pack(old_theta + proposed_Z, backtransform=true)
            hyperparams.alpha = accepted_alpha
            hyperparams.mu = accepted_mu
            hyperparams.lambda = accepted_lambda
            hyperparams.flatL = accepted_flatL
            hyperparams.L = accepted_L
            hyperparams.psi = accepted_psi
            hyperparams.nu = accepted_nu

            di.accepted_alpha += 1
            di.accepted_mu .+= 1
            di.accepted_lambda += 1
            di.accepted_flatL .+= 1
            di.accepted_nu += 1
        else
            di.rejected_alpha += 1
            di.rejected_mu .+= 1
            di.rejected_lambda += 1
            di.rejected_flatL .+= 1
            di.rejected_nu += 1
        end

        a = 0.1
        alpha_ss = 0.234
        gamma = (di.step_i + 1)^-a
        for k in 1:D
            acc_rate_k = exp(log_acceptance_rate(old_theta + proposed_Z .* (1:D .== k)))
            di.step_loglambda[k] = di.step_loglambda[k] + gamma * (acc_rate_k - alpha_ss)
        end
        thetammu = unpack(hyperparams, transform=true) - di.step_mu
        di.step_mu, di.step_sigma = (
        di.step_mu + gamma * thetammu,
        di.step_sigma + gamma * (thetammu * thetammu' - di.step_sigma)
        )
        evals, evecs = eigen(di.step_sigma)
        di.step_sigma = evecs * diagm(abs.(evals) + fill(1e-15, D)) * evecs'
        di.step_sigma = (di.step_sigma + di.step_sigma') / 2 
        di.step_i += 1

        return hyperparams
    end

    function advance_hyperparams_slice_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, scales::Vector{Float64}; temperature=1.0)
        
        d = length(hyperparams.mu)
        D = 3 + d + div(d * (d + 1), 2)

        log_q() = begin
            log_p = log_Pgenerative(clusters, hyperparams, hyperpriors=true, temperature=temperature)
            log_p += sum((d:-1:1) .* log.(abs.(diag(hyperparams.L))))
            return log_p
        end

        flatL_lower = fill(-Inf, div(d * (d + 1), 2))
        # flatL_lower[div.((1:d) .* (2:d+1), 2)] .= 0.0
        lower_bounds = vcat(0.0, fill(-Inf, d), 0.0, flatL_lower, d - 1.0)

        upper_bounds = vcat(Inf, fill(Inf, d), Inf, fill(Inf, div(d * (d + 1), 2)), Inf)

        for i in shuffle(1:D)

            log_w = log_q() + log(rand())
            current_theta = get_theta(hyperparams, i, transform=false)
            
            s = hyperparams.diagnostics.slice_s[i]
            l = rand() * s + (current_theta - s / 2)
            s = 2 * abs(l - current_theta) + rand(Exponential(scales[i]))
            hyperparams.diagnostics.slice_s[i] = s
                
            lower_box = max(lower_bounds[i], l - s / 2)
            upper_box = min(upper_bounds[i], l + s / 2)
    
            new_theta = rand() * (upper_box - lower_box) + lower_box
            set_theta!(hyperparams, new_theta, i, backtransform=false)
            while log_q() < log_w
                if new_theta < current_theta
                    lower_box = max(lower_box, new_theta)
                else
                    upper_box = min(upper_box, new_theta)
                end
                
                new_theta = rand() * (upper_box - lower_box) + lower_box
                set_theta!(hyperparams, new_theta, i, backtransform=false)
            end

        end
        
        return hyperparams

    end

    function advance_hyperparams_slice!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; scale=1.0, mask=nothing, use_transformed_variables=false)

        d = length(hyperparams.mu)

        function log_q(theta)
            log_p = log_Pgenerative(clusters, theta, hyperpriors=true, backtransform=use_transformed_variables, jacobian=use_transformed_variables)
            # We are always sampling over L regardless,
            # so the Jacobian for Psi=LL' should always
            # be included
            if !use_transformed_variables
                diag_L = theta[2 + d .+ div.((1:d) .* (2:d+1), 2)]
                log_p += sum((d:-1:1) .* log.(abs.(diag_L)))
            end
            return log_p
        end

        s0 = hyperparams.diagnostics.slice_s
        theta0 = unpack(hyperparams, transform=use_transformed_variables)
        D = length(theta0)
            
        if mask === nothing
            mask = fill(true, D)
            active_D = D
        elseif mask isa Vector{Int64}
            mask = sort!(unique(mask))
            @assert minimum(mask) >= 1 && maximum(mask) <= D
            active_D = length(mask)
        elseif mask isa Vector{Bool}
            @assert length(mask) == D
            active_D = sum(mask)
        end
        
        if scale isa Float64
            scale = fill(scale, active_D)
        elseif length(scale) == D
            scale = scale[mask]
        elseif length(scale) == active_D
            scale = scale
        end

        log_w = log_q(theta0) + log(rand())
        l = rand(active_D) .* s0[mask] .+ (theta0 .- s0 ./ 2)[mask]
        s = 2 * abs.(l .- theta0[mask]) .+ rand.(Exponential.(scale))
        hyperparams.diagnostics.slice_s[mask] = s
        
        flatL_lower = fill(-Inf, div(d * (d + 1), 2))
        flatL_lower[div.((1:d) .* (2:d+1), 2)] .= 0.0
        if use_transformed_variables
            lower_bounds = vcat(-Inf, fill(-Inf, d), -Inf, flatL_lower, -Inf)
        else
            lower_bounds = vcat(0.0, fill(-Inf, d), 0.0, flatL_lower, d - 1)
        end
        lower_bounds = lower_bounds[mask]
        
        upper_bounds = vcat(Inf, fill(Inf, d), Inf, fill(Inf, div(d * (d + 1), 2)), Inf)
        upper_bounds = upper_bounds[mask]

        lower_box = max.(lower_bounds, l .- s / 2)
        upper_box = min.(upper_bounds, l .+ s / 2)

        new_theta = copy(theta0)
        new_active_theta = rand(active_D) .* (upper_box .- lower_box) .+ lower_box
        new_theta[mask] = new_active_theta
        while log_q(new_theta) < log_w
            for i in 1:active_D
                if new_active_theta[i] < theta0[mask][i]
                    lower_box[i] = max(lower_box[i], new_active_theta[i])
                else
                    upper_box[i] = min(upper_box[i], new_active_theta[i])
                end
            end
            
            new_active_theta = rand(active_D) .* (upper_box .- lower_box) .+ lower_box
            new_theta[mask] = new_active_theta
        end

        hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.flatL, hyperparams.L, hyperparams.psi, hyperparams.nu = pack(new_theta, backtransform=use_transformed_variables)

        return hyperparams
    end

    # function sample_alpha(N::Int64, alpha::Float64, s::Float64, scale::Float64)

    #     log_q(theta) = log(jeffreys_alpha(theta, N))

    #     log_w = log_q(alpha) + log(rand())
    #     l = rand() * s + (alpha - s / 2)
    #     s = 2 * abs(l - alpha) .+ rand(Exponential(scale))
        
    #     lower_box = max(0.0, l - s / 2)
    #     upper_box = l + s / 2

    #     new_alpha = rand() * (upper_box - lower_box) + lower_box
    #     while log_q(new_alpha) < log_w
    #         if new_alpha < alpha
    #             lower_box = max(lower_box, new_alpha)
    #         else
    #             upper_box = min(upper_box, new_alpha)
    #         end
            
    #         new_alpha = rand() * (upper_box - lower_box) + lower_box
    #     end

    #     return new_alpha, s
    # end

    function MNCRPChain(filename::AbstractString)
        return JLD2.load(filename)["chain"]
    end

    function MNCRPChain(dataset::MNCRPDataset; chain_samples=200, strategy=:hot)
        chain = MNCRPChain(dataset.data, chain_samples=chain_samples, standardize=false, strategy=strategy)
        chain.data_zero = dataset.data_zero[:]
        chain.data_scale = dataset.data_scale[:]
        return chain
    end

    function MNCRPChain(data::Vector{Vector{Float64}}; standardize=true, chain_samples=100, strategy=:sequential, optimize=false)
        
        d = length(first(data))

        @assert all(length.(data) .== d)

        hyperparams = MNCRPHyperparams(d)
        hyperparams.alpha = 100.0 / log(length(data))

        clusters_samples = CircularBuffer{Vector{Cluster}}(chain_samples)
        hyperparams_samples = CircularBuffer{MNCRPHyperparams}(chain_samples)

        
        # Keep unique observations only in case we standardize
        data = Set{Vector{Float64}}(deepcopy(data))
        println("    Loaded $(length(data)) unique data points into chain")
        data = collect(data)

        data_zero = zeros(d)
        data_scale = ones(d)
        if standardize
            data_zero = mean(data)
            data_scale = std(data)
            data = Vector{Float64}[(x .- data_zero) ./ data_scale for x in data]
        end

        chain = MNCRPChain([], hyperparams, 
        data_zero, data_scale, [], [], [], [],
        clusters_samples, hyperparams_samples,
        [], deepcopy(hyperparams), -Inf, 1)


        # chain.original_data = Dict{Vector{Float64}, Vector{<:Real}}(k => v for (k, v) in zip(data, original_data))

        println("    Initializing clusters...")
        if strategy == :hot
            ##### 1st initialization method: fullseq
            chain.clusters = [Cluster(data)]
            for i in 1:10
                advance_gibbs!(chain.clusters, chain.hyperparams, temperature=1.2)
            end
        elseif strategy == :N
            chain.clusters = [Cluster([datum]) for datum in data]
        elseif strategy == :1
            chain.clusters = [Cluster(data)]
        elseif strategy == :sequential
            for element in data
                advance_gibbs!(element, chain.clusters, chain.hyperparams)
            end
        end
        if optimize
            println("    Initializing hyperparameters...")
            optimize_hyperparams!(chain.clusters, chain.hyperparams, verbose=true)
        end
        chain.hyperparams.diagnostics.step_scale = optimal_step_scale_local(chain.clusters, chain.hyperparams, optimize=false, verbose=true)

        chain.nbclusters_chain = [length(chain.clusters)]
        chain.largestcluster_chain = [maximum(length.(chain.clusters))]
        chain.hyperparams_chain = [deepcopy(hyperparams)]

        chain.map_clusters = deepcopy(chain.clusters)
        lp = log_Pgenerative(chain.clusters, chain.hyperparams)
        chain.map_logprob = lp
        chain.logprob_chain = [lp]
        # map_hyperparams=hyperparams and map_idx=1 have already been 
        # specified when calling MNCRPChain, but let's be explicit
        chain.map_hyperparams = deepcopy(chain.hyperparams)
        chain.map_idx = 1

        chain.logprob_chain = [chain.map_logprob]

        return chain

    end

    
    function advance_chain!(chain::MNCRPChain, nb_steps=100;
        nb_splitmerge=30, splitmerge_t=3, splitmerge_temp=1.0,
        nb_gibbs=1, gibbs_temp=1.0,
        nb_mhhyperparams=10, mh_stepscale=1.0, mh_temp=1.0, slice_sampler=false,
        temperature_schedule=nothing,
        sample_every=10,
        checkpoint_every=-1, checkpoint_prefix="chain",
        attempt_map=true, pretty_progress=true)
        
        checkpoint_every == -1 || typeof(checkpoint_prefix) == String || throw("Must specify a checkpoint prefix string")

        # Used for printing stats #
        hp = chain.hyperparams

        last_accepted_split = hp.diagnostics.accepted_split
        last_rejected_split = hp.diagnostics.rejected_split
        split_total = 0

        last_accepted_merge = hp.diagnostics.accepted_merge
        last_rejected_merge = hp.diagnostics.rejected_merge
        merge_total = 0
        
        # nb_fullseq_moves = 0
        
        nb_map_attemps = 0
        nb_map_successes = 0

        last_checkpoint = -1
        last_map_idx = chain.map_idx

        ###########################

        if typeof(mh_stepscale) <: Real
            mh_stepscale = mh_stepscale * ones(5)
        end

        progbar = Progress(nb_steps; showspeed=true)

        d = length(chain.hyperparams.mu)
        D = 3 + d + div(d * (d + 1), 2)
        alpha_mask = [1]
        mu_mask = collect(2:d+1)
        lambda_mask = [2 + d]
        L_mask = collect(2 + d .+ (1:div(d*(d+1), 2)))
        nu_mask = [D]

        for step in 1:nb_steps

            if temperature_schedule !== nothing
                if temperature_schedule isa Float64
                    splitmerge_temp = gibbs_temp = mh_temp = temperature_schedule
                else
                    if step > length(temperature_schedule)
                        splitmerge_temp = gibbs_temp = mh_temp = last(temperature_schedule)
                    else    
                        splitmerge_temp = gibbs_temp = mh_temp = temperature_schedule[step]
                    end
                end
            end

            # Metropolis-Hastings moves over each parameter
            # step_size is adjusted to roughly hit
            # an #accepted:#rejected ratio of 1
            # It was tuned with standardized data
            # namelyby subtracting the mean and dividing
            # by the standard deviation along each dimension.

            # for i in 1:nb_mhhyperparams
            #     advance_alpha!(chain.clusters, chain.hyperparams, 
            #                     step_type=:gaussian, step_size=0.5 * mh_stepscale[1])
                
            #     advance_mu!(chain.clusters, chain.hyperparams, 
            #                 step_type=:gaussian, step_size=0.3 * mh_stepscale[2])
                
            #     advance_lambda!(chain.clusters, chain.hyperparams, 
            #                     step_type=:gaussian, step_size=0.5 * mh_stepscale[3])

            #     advance_psi!(chain.clusters, chain.hyperparams,
            #                 step_type=:gaussian, step_size=0.1 * mh_stepscale[4])

            #     advance_nu!(chain.clusters, chain.hyperparams, 
            #                 step_type=:gaussian, step_size=0.3 * mh_stepscale[5])
            # end


            # if length(chain.logprob_chain) > 20
            #     chain.hyperparams.diagnostics.step_scale = optimal_step_scale_adaptive(chain, epsilon=0.0)
            # end
                        
            for i in 1:nb_mhhyperparams
                if slice_sampler
                    # Sampling them all at the same time doesn't work as well
                    # as sampling them in blocks
                    # advance_hyperparams_slice!(chain.clusters, chain.hyperparams, mask=alpha_mask, scale=sss)
                    # advance_hyperparams_slice!(chain.clusters, chain.hyperparams, mask=mu_mask, scale=sss)
                    # advance_hyperparams_slice!(chain.clusters, chain.hyperparams, mask=lambda_mask, scale=sss)
                    # advance_hyperparams_slice!(chain.clusters, chain.hyperparams, mask=L_mask, scale=sss)
                    # advance_hyperparams_slice!(chain.clusters, chain.hyperparams, mask=nu_mask, scale=sss)
                    advance_hyperparams_slice_seq!(chain.clusters, chain.hyperparams, chain.hyperparams.diagnostics.slice_sampler_scales, temperature=mh_temp)
                else
                    advance_hyperparams!(chain.clusters, chain.hyperparams, chain.hyperparams.diagnostics.step_scale, temperature=mh_temp)
                end
                # advance_hyperparams_algo6!(chain.clusters, chain.hyperparams)
            end

            push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

            # Sequential split-merge            
            if isfinite(nb_splitmerge)
                for i in 1:nb_splitmerge
                    advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
                end
            else
                old_nb_ssuccess = chain.hyperparams.diagnostics.accepted_split
                old_nb_msuccess = chain.hyperparams.diagnostics.accepted_merge
                while (chain.hyperparams.diagnostics.accepted_split == old_nb_ssuccess) && (chain.hyperparams.diagnostics.accepted_merge == old_nb_msuccess)
                    advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
                end
            end

            # Gibbs sweep
            if nb_gibbs isa Int64
                for i in 1:nb_gibbs
                    advance_gibbs!(chain.clusters, chain.hyperparams, temperature=gibbs_temp)
                end
            elseif 0.0 < nb_gibbs < 1.0
                rand() < nb_gibbs ? advance_gibbs!(chain.clusters, chain.hyperparams, temperature=gibbs_temp) : nothing
            end

            push!(chain.nbclusters_chain, length(chain.clusters))
            push!(chain.largestcluster_chain, maximum(length.(chain.clusters)))

            # Stats #
            split_ratio = "$(hp.diagnostics.accepted_split - last_accepted_split)/$(hp.diagnostics.rejected_split - last_rejected_split)"
            merge_ratio = "$(hp.diagnostics.accepted_merge - last_accepted_merge)/$(hp.diagnostics.rejected_merge - last_rejected_merge)"
            split_total += hp.diagnostics.accepted_split - last_accepted_split
            merge_total += hp.diagnostics.accepted_merge - last_accepted_merge
            split_per_step = round(split_total/step, digits=2)
            merge_per_step = round(merge_total/step, digits=2)
            last_accepted_split = hp.diagnostics.accepted_split
            last_rejected_split = hp.diagnostics.rejected_split
            last_accepted_merge = hp.diagnostics.accepted_merge
            last_rejected_merge = hp.diagnostics.rejected_merge


            ########################
    
            # logprob
            logprob = log_Pgenerative(chain.clusters, chain.hyperparams)
            push!(chain.logprob_chain, logprob)

            # MAP
            history_length = 500
            short_logprob_chain = chain.logprob_chain[max(1, end - history_length):end]
            
            logp_quantile95 = quantile(short_logprob_chain, 0.95)

            map_success = false # try block has its own scope

            if logprob > logp_quantile95 && attempt_map
                    # Summit attempt
                    nb_map_attemps += 1
                    try
                        map_success = attempt_map!(chain, max_nb_pushes=15, verbose=true)
                    catch e
                        map_success = false
                    end                    
                    
                    nb_map_successes += map_success
                    last_map_idx = chain.map_idx

            end

            if (map_success && length(chain.logprob_chain) > 50) || length(chain.logprob_chain) == 50
                if slice_sampler
                    chain.hyperparams.diagnostics.slice_sampler_scales = 3*iqr.(eachrow(reduce(hcat, unpack.(chain.hyperparams_chain, transform=false))))
                    # slice_sampling_scales = 2*abs.(unpack(chain.hyperparams, transform=false))
                else
                    # map_success || attempt_map!(chain, max_nb_pushes=15)
                    # chain.hyperparams.diagnostics.step_scale = optimal_step_scale_local(chain.map_clusters, chain.map_hyperparams, optimize=false, verbose=true)
                end
            end

            # Recalculate step_scale when chain length hits powers of 2.
            # The bias vanishes as length(chain) -> infinity
            if length(chain.logprob_chain) >= 16 && floor(log(2, length(chain.logprob_chain))) == log(2, length(chain.logprob_chain))
                chain.hyperparams.diagnostics.step_scale = optimal_step_scale_local(
                    chain.clusters, chain.hyperparams, 
                    optimize=true, jacobian=false, verbose=true)
            end

            if sample_every !== nothing && sample_every >= 1
                if mod(length(chain.logprob_chain), sample_every) == 0
                    push!(chain.clusters_samples, deepcopy(chain.clusters))
                    push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
                end
            end


            if checkpoint_every > 0 && 
                (mod(length(chain.logprob_chain), checkpoint_every) == 0
                || last_checkpoint == -1)

                last_checkpoint = length(chain.logprob_chain)
                mkpath(dirname("$(checkpoint_prefix)"))
                filename = "$(checkpoint_prefix)_pid$(getpid())_iter$(last_checkpoint).jld2"
                jldsave(filename; chain)
            end

            if pretty_progress
                ProgressMeter.next!(progbar;
                showvalues=[
                (:"step (hp, sm, gb temperatures)", "$(step)/$(nb_steps) ($(round(mh_temp, digits=2)), $(round(splitmerge_temp, digits=2)), $(round(gibbs_temp, digits=2)))"),
                (:"chain length (#chain samples)", "$(length(chain.logprob_chain)) ($(length(chain.clusters_samples))/$(length(chain.clusters_samples.buffer)))"),
                (:"logprob (max, q95)", "$(round(chain.logprob_chain[end], digits=1)) ($(round(maximum(chain.logprob_chain), digits=1)), $(round(logp_quantile95, digits=1)))"),
                (:"clusters (>1, median, mean, max)", "$(length(chain.clusters)) ($(length(filter(c -> length(c) > 1, chain.clusters))), $(round(median([length(c) for c in chain.clusters]), digits=0)), $(round(mean([length(c) for c in chain.clusters]), digits=0)), $(maximum([length(c) for c in chain.clusters])))"),
                (:"split ratio, merge ratio", split_ratio * ", " * merge_ratio),
                (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
                (:"MAP att/succ", "$(nb_map_attemps)/$(nb_map_successes)" * (attempt_map ? "" : " (off)")),
                (:"MAP clusters (>1, median, mean, max)", "$(length(chain.map_clusters)) ($(length(filter(c -> length(c) > 1, chain.map_clusters))), $(round(median([length(c) for c in chain.map_clusters]), digits=0)), $(round(mean([length(c) for c in chain.map_clusters]), digits=0)), $(maximum([length(c) for c in chain.map_clusters])))"),
                (:"last MAP logprob", round(chain.map_logprob, digits=1)),
                (:"last MAP at", last_map_idx),
                (:"last checkpoint at", last_checkpoint)
                ])
            else
                print("\r$(step)/$(nb_steps)")
                print(" (t:$(length(chain.logprob_chain)))")
                print("   lp: $(round(chain.logprob_chain[end], digits=1))")
                print("   sr:" * split_ratio * " mr:" * merge_ratio)
                print("   sps:$(split_per_step), mps:$(merge_per_step)")
                print("   #cl:$(length(chain.clusters)), #cl>1:$(length(filter(x -> length(x) > 1, chain.clusters)))")
                print("   mapattsuc:$(nb_map_attemps)/$(nb_map_successes)")
                print("   lastmap@$(last_map_idx)")
                print("   lchpt@$(last_checkpoint)")
                print("      ")
                if map_success
                    print("!      #clmap:$(length(chain.map_clusters))")
                    println("   $(round(chain.map_logprob, digits=1))")
                end
                flush(stdout)
            end

        end
    end


    function attempt_map!(chain::MNCRPChain; max_nb_pushes=15, optimize_hyperparams=true, verbose=true, force=false)
            
        map_clusters_attempt = deepcopy(chain.clusters)
        map_hyperparams = deepcopy(chain.hyperparams)

        map_mll = log_Pgenerative(map_clusters_attempt, chain.map_hyperparams)

        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = copy(map_clusters_attempt)
            advance_gibbs!(test_attempt, map_hyperparams; temperature=0.0)
            test_mll = log_Pgenerative(test_attempt, map_hyperparams)
            if test_mll <= map_mll
                # We've regressed, so stop and leave
                # the previous state before this test
                # attempt as the approx. map state
                break
            else
                # the test state led to a better state
                # than the previous one, keep going
                map_clusters_attempt = test_attempt
                map_mll = test_mll
            end
        end
        
        if optimize_hyperparams
            optimize_hyperparams!(map_clusters_attempt, map_hyperparams, verbose=verbose)
        end

        attempt_logprob = log_Pgenerative(map_clusters_attempt, map_hyperparams)
        if attempt_logprob > chain.map_logprob || force
            chain.map_logprob = attempt_logprob
            chain.map_clusters = map_clusters_attempt
            chain.map_hyperparams = map_hyperparams
            chain.map_idx = lastindex(chain.logprob_chain)
            return true
        else
            return false
        end
    end

    function plot(clusters::Vector{Cluster}; dims::Vector{Int64}=[1, 2], rev=false, nb_clusters=nothing, plot_kw...)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        p = plot(; legend_position=:topleft, grid=:no, showaxis=:yes, ticks=:true, plot_kw...)

        clusters = project_clusters(sort(clusters, by=length, rev=rev), dims)


        if nb_clusters === nothing || nb_clusters < 0
            nb_clusters = length(clusters)
        end

        for (cluster, color, _) in zip(clusters, cycle(tableau_20), 1:nb_clusters)
            scatter!(collect(Tuple.(cluster)), label="$(length(cluster))", 
            color=color, markerstrokewidth=0)
        end
        
        # display(p)
        return p

    end

    # function plot(chain::MNCRPChain, cluster::Cluster, hyperparams::MNCRPHyperparams; eigdirs::Vector{Float64}=[1, 2], burn=0)

    #     _, evecs = eigen_mode(cluster, hyperparams)

    #     # proj = (evecs[:, end + 1 - eigdirs[1]], evecs[:, end + 1 - eigdirs[2]])
    #     proj = Matrix{Float64}(evecs[:, [end + 1 - eigdirs[1], end + 1 - eigdirs[2]]]')

    #     return plot(chain, proj, burn=burn)

    # end
    
    function plot(chain::MNCRPChain; dims::Vector{Int64}=[1, 2], burn=0, rev=false, nb_clusters=nothing)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        d = length(chain.hyperparams.mu)
        
        proj = dims_to_proj(dims, d)

        return plot(chain, proj, burn=burn, rev=rev, nb_clusters=nb_clusters)
    end

    function plot(chain::MNCRPChain, proj::Matrix{Float64}; burn=0, rev=false, nb_clusters=nothing)
        
        @assert size(proj, 1) == 2 "The projection matrix should have 2 rows"
        
        if burn < 0
            burn = length(chain.logprob_chain) + burn
        end

        map_marginals = project_clusters(chain.map_clusters, proj)
        current_marginals = project_clusters(chain.clusters, proj)
        
        p_map = plot(map_marginals; rev=rev, nb_clusters=nb_clusters, title="MAP state ($(length(chain.map_clusters)) clusters)")
        p_current = plot(current_marginals; rev=rev, nb_clusters=nb_clusters, title="Current state ($(length(chain.clusters)) clusters)", legend=false)

        lpc = chain.logprob_chain
        p_logprob = plot(burn+1:length(lpc), lpc[burn+1:end], grid=:no, label=nothing, title="log probability chain")
        hline!(p_logprob, [chain.map_logprob], label=nothing, color=:green)
        hline!(p_logprob, [maximum(chain.logprob_chain)], label=nothing, color=:black)
        vline!(p_logprob, [chain.map_idx], label=nothing, color=:black)

        ac = alpha_chain(chain)
        p_alpha = plot(burn+1:length(ac), ac[burn+1:end], grid=:no, label=nothing, title="α chain")
        vline!(p_alpha, [chain.map_idx], label=nothing, color=:black)

        muc = reduce(hcat, mu_chain(chain))'
        p_mu = plot(burn+1:size(muc, 1), muc[burn+1:end, :], grid=:no, label=nothing, title="μ₀ chain")
        vline!(p_mu, [chain.map_idx], label=nothing, color=:black)

        lc = lambda_chain(chain)
        p_lambda = plot(burn+1:length(lc), lc[burn+1:end], grid=:no, label=nothing, title="λ₀ chain")
        vline!(p_lambda, [chain.map_idx], label=nothing, color=:black)
        
        pc = flatten.(LowerTriangular.(psi_chain(chain)))
        pc = reduce(hcat, pc)'
        p_psi = plot(burn+1:size(pc, 1), pc[burn+1:end, :], grid=:no, label=nothing, title="Ψ₀ chain")
        vline!(p_psi, [chain.map_idx], label=nothing, color=:black)

        nc = nu_chain(chain)
        p_nu = plot(burn+1:length(nc), nc[burn+1:end], grid=:no, label=nothing, title="ν₀ chain")
        vline!(p_nu, [chain.map_idx], label=nothing, color=:black)

        
        # N = sum(length.(chain.clusters))

        nbc = chain.nbclusters_chain
        # p_nbc = plot(burn+1:length(nbc), log(N) * ac[burn+1:end], grid=:no, label="Asymptotic mean", title="#cluster chain", legend=true)
        p_nbc = plot(burn+1:length(nbc), nbc[burn+1:end], grid=:no,label=nothing, title="Number of clusters chain")
        vline!(p_nbc, [chain.map_idx], label=nothing, color=:black)

        lcc = chain.largestcluster_chain
        p_lcc = plot(burn+1:length(lcc), lcc[burn+1:end], grid=:no,label=nothing, title="Largest cluster chain")
        vline!(p_lcc, [chain.map_idx], label=nothing, color=:black)

        # empty_plot = plot(legend=false, grid=false, foreground_color_subplot=:white)

        lo = @layout [a{0.4h} b; c d; e f; g h; i j]
        p = plot(
        p_map, p_current, 
        p_logprob, p_lcc, 
        p_alpha, p_nbc,
        p_mu, p_lambda, 
        p_psi, p_nu,
        size=(1500, 1500), layout=lo)

        return p
    end

    function eigen_mode(cluster::Cluster, hyperparams::MNCRPHyperparams)
        mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
        d = length(hyperparams.mu)
        sigma_mode = Symmetric((lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c)
        return eigen(sigma_mode)
    end

    """
    For each eigenvalue/eigenvector pair (sorted from largest to smallest eigenvalue) of the (predictive) covariance matrix of the cluster,
    return the layer loadings (elements of the eigenvector) from the largest in absolute value (most important) to the smallest.
    """
    function importances(cluster::Cluster, hyperparams::MNCRPHyperparams)
        evals, evecs = eigen_mode(cluster, hyperparams)

        imps = []
        for (eva, eve) in reverse(collect(zip(evals, eachcol(evecs))))
            imp = sort(collect(zip(eve, 1:length(eve))), by=x->abs(x[1]), rev=true)
            push!(imps, imp)
        end

        return imps
    end

    function local_geometry(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(hyperparams.mu)
        
        local_scale, _ = local_covprec(coordinate, clusters, hyperparams)

        evals, evecs = eigen(local_scale)
        
        imps = []
        for (eva, eve) in reverse(collect(zip(evals, eachcol(evecs))))
            imp = sort(collect(zip(eve, 1:length(eve))), by=x->abs(x[1]), rev=true)
            push!(imps, (eval=eva, sorted_evecs=imp))
        end

        normalized_evals = [imp.eval for imp in imps]
        normalized_evals = normalized_evals ./ sum(normalized_evals)
        entropy = - sum(normalized_evals .* log.(normalized_evals))

        return (importances=imps, entropy=entropy)

    end

    function covellipses!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; dims::Vector{Int64}=[1, 2], n_std=2, scalematrix=nothing, offset=nothing, type=:predictive, lowest_weight=nothing, plot_kw...)

        mu0, lambda0, psi0, nu0 = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu0, 1)

        p = nothing

        for c in clusters
            if lowest_weight === nothing || length(c) >= lowest_weight
                mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(c, mu0, lambda0, psi0, nu0)
                
                if mode == :mode
                    # Sigma mode of the posterior
                    sigma_c = psi_c / (nu_c + d + 1)
                elseif mode == :mean
                    # Average sigma of the posterior
                    sigma_c = psi_c / (nu_c - d - 1)
                elseif mode == :predictive
                    sigma_c = (lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c
                end

                if !(scalematrix === nothing)
                    mu_c = inv(scalematrix) * mu_c
                    sigma_c = inv(scalematrix) * sigma_c * inv(scalematrix)'
                end

                if !(offset === nothing)
                    mu_c += offset
                end

                mu_c = project_vec(mu_c, dims)
                sigma_c = project_mat(sigma_c, dims)

                covellipse!(mu_c, sigma_c; n_std=n_std, legend=nothing, plot_kw...)
            end
        end

    end

    function local_covprec(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        N = sum(length(c) for c in clusters)
        
        clusters = vcat(clusters, [Cluster(d)])        
        log_parts = []
        cluster_covs = []
        cluster_precs = []
        
        coordinate = coordinate .* (1.0 .+ 1e-7 * rand(d))

        for cluster in clusters
            
            push!(log_parts, log_cluster_weight(coordinate, cluster, alpha, mu, lambda, psi, nu))            
            _, _, mvstudent_sigma = updated_mvstudent_params(cluster, mu, lambda, psi, nu)
            push!(cluster_covs, mvstudent_sigma)
            push!(cluster_precs, inv(mvstudent_sigma))
        end
            
        log_parts = log_parts .- logsumexp(log_parts)
        
        covariance = sum(cluster_covs .* exp.(log_parts))
        precision = sum(cluster_precs .* exp.(log_parts))

        return (covariance=covariance, precision=precision)

    end

    function local_covariance(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        return local_covprec(coordinate, clusters, hyperparams).covariance
    end
    
    function local_precision(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        return local_covprec(coordinate, clusters, hyperparams).precision
    end

    function local_covariance_summary(coordinate::Vector{Float64}, clusters_samples::CircularBuffer{Vector{Cluster}},  hyperparams_samples::CircularBuffer{MNCRPHyperparams})
        return local_covariance_summary(coordinates, collect(clusters_samples), collect(hyperparams_samples))
    end
    
    function local_covariance_summary(coordinate::Vector{Float64}, clusters_samples::Vector{Vector{Cluster}},  hyperparams_samples::Vector{MNCRPHyperparams})

        cov_samples = Matrix{Float64}[local_covariance(coordinate, clusters, hyperparams) for (clusters, hyperparams) in zip(clusters_samples, hyperparams_samples)]

        d = size(first(cov_samples), 1)

        cov_median = zeros(d, d)
        cov_iqr = zeros(d, d)
        for i in 1:d
            for j in 1:i
                matrix_element_samples = Float64[m[i, j] for m in cov_samples]
                cov_median[i, j] = median(matrix_element_samples)
                cov_iqr[i, j] = iqr(matrix_element_samples)
                
                cov_median[j, i] = cov_median[i, j]
                cov_iqr[j, i] = cov_iqr[j, i]
            end
        end

        return (median=cov_median, iqr=cov_iqr)

    end

    function local_covariance_summary(coordinate::Vector{Float64}, chain::MNCRPChain)
        return local_covariance_summary(coordinate, chain.clusters_samples, chain.hyperparams_samples)
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::CircularBuffer{Vector{Cluster}},  hyperparams_samples::CircularBuffer{MNCRPHyperparams})
        return local_covariance_summary(coordinates, collect(clusters_samples), collect(hyperparams_samples))    
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::Vector{Vector{Cluster}},  hyperparams_samples::Vector{MNCRPHyperparams})
        
        meds_iqrs = NamedTuple{(:median, :iqr), Tuple{Matrix{Float64}, Matrix{Float64}}}[local_covariance_summary(coordinate, clusters_samples, hyperparams_samples) for coordinate in coordinates]

        medians = Matrix{Float64}[med_iqr.median for med_iqr in meds_iqrs]
        iqrs = Matrix{Float64}[med_iqr.iqr for med_iqr in meds_iqrs]

        return (median=medians, iqr=iqrs)
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, chain::MNCRPChain)
        return local_covariance_summary(coordinates, chain.clusters_samples, chain.hyperparams_samples)
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

    
    function crp_distance(ci::Int64, cj::Int64, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        cluster1, cluster2 = clusters[ci], clusters[cj]

        if cluster1 === cluster2
            return 0.0
        end

        return -(
                -log(alpha)
                
                + loggamma(length(cluster1) + length(cluster2))
                - loggamma(length(cluster1)) 
                - loggamma(length(cluster2))

                + log_Zniw(union(cluster1, cluster2), mu, lambda, psi, nu)
                - log_Zniw(cluster1, mu, lambda, psi, nu)
                - log_Zniw(cluster2, mu, lambda, psi, nu)

                + log_Zniw(nothing, mu, lambda, psi, nu)
                )
    end
    
    function crp_distance_matrix(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
    
        dist = zeros(length(clusters), length(clusters))
        
        for i in 1:length(clusters)
            for j in 1:i
                dist[i, j] = crp_distance(i, j, clusters, hyperparams)
                dist[j, i] = dist[i, j]
            end
        end

        return dist
    end


    function minimum_size(clusters::Vector{Cluster}, proportion=0.05)

        N = sum(length.(clusters))
        include_up_to = N * (1 - proportion)

        sorted_sizes = sort(length.(clusters), rev=true)
        cumul_sizes = cumsum(sorted_sizes)

        last_idx = findlast(x -> x <= include_up_to, cumul_sizes)

        # min_cumul_idx = findfirst(x -> x > propN, cumul_sizes)
        # min_size_idx = findfirst(x -> x > sorted_sizes[min_cumul_idx], sorted_sizes)

        minsize = sorted_sizes[last_idx]

        return max(minsize, 0)

    end

    function optimize_hyperparams(clusters::Vector{Cluster}, hyperparams0::MNCRPHyperparams; jacobian=false, verbose=false)

        objfun(x) = -log_Pgenerative(clusters, x, jacobian=jacobian)

        x0 = unpack(hyperparams0)

        if verbose
            function callback(x)
                print(" * Iter $(x.iteration),   objfun $(-round(x.value, digits=2)),   g_norm $(round(x.g_norm, digits=8))\r")
                return false
            end
        else
            callback =  nothing
        end
        opt_options = Options(iterations=50000,
                              x_tol=1e-8,
                              f_tol=1e-6,
                              g_tol=2e-2,
                              callback=callback)

        optres = optimize(objfun, x0, NelderMead(), opt_options)

        if verbose
            println()
        end

        opt_hp = pack(minimizer(optres))

        return MNCRPHyperparams(opt_hp..., deepcopy(hyperparams0.diagnostics))

    end

    function optimize_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; jacobian=false, verbose=false)
        
        opt_res = optimize_hyperparams(clusters, hyperparams, jacobian=jacobian, verbose=verbose)

        hyperparams.alpha = opt_res.alpha
        hyperparams.mu = opt_res.mu
        hyperparams.lambda = opt_res.lambda
        hyperparams.flatL = opt_res.flatL
        hyperparams.L = opt_res.L
        hyperparams.psi = opt_res.psi
        hyperparams.nu = opt_res.nu

        return hyperparams
    
    end

    function updated_mvstudent_params(
        cluster::Nothing, 
        mu::AbstractVector{Float64}, 
        lambda::Float64, 
        psi::AbstractMatrix{Float64}, 
        nu::Float64
        )::Tuple{Float64, Vector{Float64}, Matrix{Float64}}

        d = length(mu)

        return (nu - d + 1, mu, (lambda + 1)/lambda/(nu - d + 1) * psi)

    end

    function updated_mvstudent_params(
        cluster::Cluster, 
        mu::AbstractVector{Float64}, 
        lambda::Float64, 
        psi::AbstractMatrix{Float64},
        nu::Float64
        )::Tuple{Float64, Vector{Float64}, Matrix{Float64}}

        d = length(mu)
        mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

        return (nu_c - d + 1, mu_c, (lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c)

    end

    function updated_mvstudent_params(
        clusters::Vector{Cluster}, 
        mu::AbstractVector{Float64}, 
        lambda::Float64, 
        psi::AbstractMatrix{Float64}, 
        nu::Float64; 
        add_empty=true
        )::Vector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}}
        
        d = length(mu)
        updated_mvstudent_degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]
        if add_empty
            push!(updated_mvstudent_degs_mus_sigs, updated_mvstudent_params(nothing, mu, lambda, psi, nu))
        end

        return updated_mvstudent_degs_mus_sigs
    end

    function predictive_logpdf(
        clusters::Vector{Cluster}, 
        hyperparams::MNCRPHyperparams;
        ignore_weights=false
        )::Function
        
        alpha, mu, lambda, psi, nu = collect(hyperparams)
        d = length(mu)

        if ignore_weights
            weights = ones(length(clusters) + 1)
            weights ./= sum(weights)
        else
            weights = 1.0 .* length.(clusters)
            push!(weights, alpha)
            weights ./= sum(weights)
        end
        
        updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)

        return predictive_logpdf(weights, updated_mvstudent_degs_mus_sigs)
    end

    function predictive_logpdf(
        component_weights::AbstractVector{Float64}, 
        mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}}
        )::Function

        @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

        d = length(first(mvstudent_degs_mus_sigs)[2])

        function logpdf(x::AbstractVector{Float64})::Float64
            return logsumexp(log(weight) + loggamma((deg + d)/2) - loggamma(deg/2)
                            - d/2 * log(deg) - d/2 * log(pi) - 1/2 * logdetpsd(sig)
                            - (deg + d)/2 * log(1 + 1/deg * (x - mu)' * (sig \ (x - mu)))
                            for (weight, (deg, mu, sig)) in zip(component_weights, mvstudent_degs_mus_sigs))
        end
        
        return logpdf

    end

    
    function tail_probability(
        component_weights::AbstractVector{Float64}, 
        mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}};
        rejection_samples=10000
        )::Function
        
        @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

        logpdf = predictive_logpdf(component_weights, mvstudent_degs_mus_sigs)
        
        sample_logpdfs = zeros(rejection_samples)
        for i in 1:rejection_samples
            sample_deg, sample_mu, sample_sig = sample(mvstudent_degs_mus_sigs, Weights(component_weights))
            sample_draw = rand(MvTDist(sample_deg, sample_mu, sample_sig))
            sample_logpdfs[i] = logpdf(sample_draw)
        end

        function tailprob_func(coordinate::AbstractVector{Float64})::Float64
            log_isocontour_val = logpdf(coordinate)
            tail = sum(sample_logpdfs .<= log_isocontour_val) / rejection_samples
            return tail
        end
        
        return tailprob_func
        
    end

    function clustered_tail_probability(clusters::AbstractVector{Cluster}, hyperparams::MNCRPHyperparams; rejection_samples=1000)::Function

        alpha, mu, lambda, psi, nu = collect(hyperparams)
        d = length(mu)
        N = sum(length.(clusters))

        mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)

        clusters_logpdf_samples = Vector{Float64}[]
        clusters_logpdfs = Function[]
        for (deg, mu, sig) in mvstudent_degs_mus_sigs
            cluster_logpdf = predictive_logpdf([1.0], [(deg, mu, sig)])
            push!(clusters_logpdfs, cluster_logpdf)
            push!(clusters_logpdf_samples, cluster_logpdf.(eachcol(rand(MvTDist(deg, mu, sig), rejection_samples))))
        end

        function tailprob_func(coordinate::AbstractVector{Float64})::Float64
            coordinate = coordinate + 1e-8 * (rand(length(coordinate)) .- 0.5)
            log_cluster_weights = [log_cluster_weight(coordinate, cluster, alpha, mu, lambda, psi, nu, N=N) for cluster in clusters]
            push!(log_cluster_weights, log_cluster_weight(coordinate, Cluster(d), alpha, mu, lambda, psi, nu, N=N))
            log_cluster_probs  = exp.(log_cluster_weights .- logsumexp(log_cluster_weights))

            log_isocontours_vals = [cluster_logpdf(coordinate) for cluster_logpdf in clusters_logpdfs]
            cluster_tail_probs= [sum(cluster_logpdf_samples .<= log_isocontour_val) / rejection_samples for (log_isocontour_val, cluster_logpdf_samples) in zip(log_isocontours_vals, clusters_logpdf_samples)]
            return sum(log_cluster_prob * cluster_tail_prob for (log_cluster_prob, cluster_tail_prob) in zip(log_cluster_probs, cluster_tail_probs))
        end

        return tailprob_func

    end


    function tail_probability(
        clusters::Vector{Cluster}, 
        hyperparams::MNCRPHyperparams; 
        rejection_samples=10000, 
        ignore_weights=false)
        
        alpha, mu, lambda, psi, nu = collect(hyperparams)
        
        if ignore_weights
            weights = ones(length(clusters) + 1)
            weights ./= sum(weights)
        else
            weights = 1.0 * length.(clusters)
            push!(weights, alpha)
            weights ./= sum(weights)
        end

        updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)
        
        return tail_probability(weights, updated_mvstudent_degs_mus_sigs, rejection_samples=rejection_samples)

    end
    
    function tail_probability(coordinates::AbstractVector{Vector{Float64}}, clusters::AbstractVector{Cluster}, hyperparams::MNCRPHyperparams; rejection_samples=10000)
        tprob = tail_probability(clusters, hyperparams, rejection_samples=rejection_samples) 
        return tprob.(coordinates)
    end

    
    function tail_probability(coordinates::AbstractVector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}}, hyperparams_samples::AbstractVector{MNCRPHyperparams}; rejection_samples=10000, nb_samples=nothing)
        
        @assert length(clusters_samples) == length(hyperparams_samples)
        @assert length(clusters_samples) > 1
        
        if nb_samples === nothing
            first_idx = 1
            nb_samples = length(clusters_samples)
        else
            first_idx = max(1, length(clusters_samples) - nb_samples + 1)
        end

        tail_probs_samples = Vector{Float64}[]
        for (i, (clusters, hyperparams)) in enumerate(zip(clusters_samples[first_idx:end], hyperparams_samples[first_idx:end]))
            print("\rProcessing sample $(i)/$(nb_samples)")
            push!(tail_probs_samples, tail_probability(coordinates, clusters, hyperparams, rejection_samples=rejection_samples))
        end
        println()
        
        # transpose
        tail_probs_distributions = collect.(eachrow(reduce(hcat, tail_probs_samples)))
        
        return tail_probs_distributions
        
    end
    
    function tail_probability_summary(coordinates::AbstractVector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}},  hyperparams_samples::AbstractVector{MNCRPHyperparams}; rejection_samples=10000, nb_samples=nothing)
        
        tail_probs_distributions = tail_probability(coordinates, 
                                                    clusters_samples, hyperparams_samples, 
                                                    rejection_samples=rejection_samples, nb_samples=nb_samples)
        
        emp_summaries = empirical_distribution_summary.(tail_probs_distributions)
        
        emp_summaries_t = (mean=Float64[], std=Float64[], mode=Float64[], median=Float64[], iqr=Float64[], iqr90=Float64[], q5=Float64[], q25=Float64[], q75=Float64[], q95=Float64[])
        for summ in emp_summaries
            push!(emp_summaries_t.mean, summ.mean)
            push!(emp_summaries_t.std, summ.std)
            push!(emp_summaries_t.mode, summ.mode)
            push!(emp_summaries_t.median, summ.median)
            push!(emp_summaries_t.iqr, summ.iqr)
            push!(emp_summaries_t.iqr90, summ.iqr90)
            push!(emp_summaries_t.q5, summ.q5)
            push!(emp_summaries_t.q25, summ.q25)
            push!(emp_summaries_t.q75, summ.q75)
            push!(emp_summaries_t.q95, summ.q95)
        end

        return emp_summaries_t
        
    end
    
    function tail_probability_summary(coordinates::Vector{Vector{Float64}}, chain::MNCRPChain; rejection_samples=10000, nb_samples=nothing)
        return tail_probability_summary(coordinates, chain.clusters_samples, chain.hyperparams_samples, rejection_samples=rejection_samples, nb_samples=nb_samples)
    end

    function presence_probability(presence_clusters::AbstractVector{Cluster}, presence_hyperparams::MNCRPHyperparams, absence_clusters::AbstractVector{Cluster}, absence_hyperparams::MNCRPHyperparams; logprob=false)
        
        log_presence_predictive = predictive_logpdf(presence_clusters, presence_hyperparams)
        log_absence_predictive = predictive_logpdf(absence_clusters, absence_hyperparams)
        
        function func(coordinate::Vector{Float64})
            lpp = log_presence_predictive(coordinate)
            lap = log_absence_predictive(coordinate)
            lp = lpp - logsumexp([lpp, lap])
            if logprob
                return lp
            else
                return exp(lp)
            end
        end
        
        return func
        
    end

    
    function presence_probability(
        coordinates::AbstractVector{Vector{Float64}}, 
        presence_clusters::Vector{Cluster}, presence_hyperparams::MNCRPHyperparams, 
        absence_clusters::Vector{Cluster}, absence_hyperparams::MNCRPHyperparams;
        logprob=false
        )

        pres_pred = presence_probability(presence_clusters, presence_hyperparams, 
                                        absence_clusters, absence_hyperparams, 
                                        logprob=logprob)
        
        return pres_pred.(coordinates)
    end


    function presence_probability(
        coordinates::AbstractVector{Vector{Float64}}, 
        presence_clusters_samples::AbstractVector{Vector{Cluster}}, presence_hyperparams_samples::AbstractVector{MNCRPHyperparams}, 
        absence_clusters_samples::AbstractVector{Vector{Cluster}}, absence_hyperparams_samples::AbstractVector{MNCRPHyperparams};
        nb_samples=nothing, logprob=false
        )

        @assert length(presence_clusters_samples) == length(presence_hyperparams_samples) == length(absence_clusters_samples) == length(absence_hyperparams_samples)
        @assert length(presence_clusters_samples) > 1

        if nb_samples === nothing
            first_idx = 1
            nb_samples = length(presence_clusters_samples)
        else
            first_idx = max(1, length(presence_clusters_samples) - nb_samples + 1)
        end
        
        

        presence_probs_samples = Vector{Float64}[]
        for (i, (presence_clusters, presence_hyperparams, absence_clusters, absence_hyperparams)) in enumerate(zip(presence_clusters_samples[first_idx:end], presence_hyperparams_samples[first_idx:end], absence_clusters_samples[first_idx:end], absence_hyperparams_samples[first_idx:end]))
            print("\rProcessing sample $(i)/$(nb_samples)")
            push!(presence_probs_samples, presence_probability(coordinates, presence_clusters, presence_hyperparams, absence_clusters, absence_hyperparams, logprob=logprob))
        end
        println()

        presence_probabilities_distributions = collect.(eachrow(reduce(hcat, presence_probs_samples)))

        return presence_probabilities_distributions
    end


    function presence_probability_summary(
        coordinates::AbstractVector{Vector{Float64}}, 
        presence_clusters_samples::AbstractVector{Vector{Cluster}}, presence_hyperparams_samples::AbstractVector{MNCRPHyperparams},
        absence_clusters_samples::AbstractVector{Vector{Cluster}}, absence_hyperparams_samples::AbstractVector{MNCRPHyperparams};
        nb_samples=nothing, logprob=false
        )

        presence_probability_distributions = presence_probability(coordinates,
                                            presence_clusters_samples, presence_hyperparams_samples, 
                                            absence_clusters_samples, absence_hyperparams_samples,
                                            nb_samples=nb_samples, logprob=logprob
                                            )

        emp_summaries = empirical_distribution_summary.(presence_probability_distributions)
        
        emp_summaries_t = (mean=Float64[], std=Float64[], mode=Float64[], median=Float64[], iqr=Float64[], iqr90=Float64[], q5=Float64[], q25=Float64[], q75=Float64[], q95=Float64[])
        for summ in emp_summaries
            push!(emp_summaries_t.mean, summ.mean)
            push!(emp_summaries_t.std, summ.std)
            push!(emp_summaries_t.mode, summ.mode)
            push!(emp_summaries_t.median, summ.median)
            push!(emp_summaries_t.iqr, summ.iqr)
            push!(emp_summaries_t.iqr90, summ.iqr90)
            push!(emp_summaries_t.q5, summ.q5)
            push!(emp_summaries_t.q25, summ.q25)
            push!(emp_summaries_t.q75, summ.q75)
            push!(emp_summaries_t.q95, summ.q95)
        end

        return emp_summaries_t

    end


    function presence_probability_summary(
        coordinates::AbstractVector{Vector{Float64}}, 
        presence_chain::MNCRPChain, absence_chain::MNCRPChain;
        nb_samples=nothing, logprob=false
        )

        return presence_probability_summary(coordinates,
                    presence_chain.clusters_samples, presence_chain.hyperparams_samples,
                    absence_chain.clusters_samples, absence_chain.hyperparams_samples,
                    nb_samples=nb_samples, logprob=logprob)
        
    end


    function empirical_distribution_summary(empirical_distribution::Vector{Float64})

        hist = fit(Histogram, empirical_distribution)
        _, max_idx = findmax(hist.weights)
        emp_mode = (first(hist.edges)[max_idx] + first(hist.edges)[max_idx+1]) / 2

        emp_mean = mean(empirical_distribution)
        emp_std = std(empirical_distribution)
        emp_median = median(empirical_distribution)
        emp_iqr = iqr(empirical_distribution)
        emp_quantile5 = quantile(empirical_distribution, 0.05)
        emp_quantile25 = quantile(empirical_distribution, 0.25)
        emp_quantile75 = quantile(empirical_distribution, 0.75)
        emp_quantile95 = quantile(empirical_distribution, 0.95)
        emp_iqr90 = emp_quantile95 - emp_quantile5
        
        return (mean=emp_mean, std=emp_std, mode=emp_mode, median=emp_median, iqr=emp_iqr, iqr90=emp_iqr90, q5=emp_quantile5, q25=emp_quantile25, q75=emp_quantile75, q95=emp_quantile95)

    end

    function map_cluster_assignment_idx(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(hyperparams.mu)
        
        clusters = sort(clusters, by=length, rev=true)
        push!(clusters, Cluster(d))

        cluster_weights = Float64[log_cluster_weight(coordinate, cluster, alpha, mu, lambda, psi, nu) for cluster in clusters]
        map_cluster, map_idx = findmax(cluster_weights)
        
        # map_idx = 0 indicates new cluster assignment
        map_idx = map_idx == length(cluster_weights) ? 0 : map_idx
        
        return map_idx

    end

    function append_tail_probabilities!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; rejection_samples=10000, optimize=true)

        # While elements in clusters are changed in place,
        # each cluster's sum_x and sum_xx remain unchanged 
        # because there is no push!/pop! involved. 
        # Therefore tail_prob functions as if we are still
        # in dimensions d rather than d + 1
        
        
        tail_prob = tail_probability(clusters, hyperparams, rejection_samples=rejection_samples)

        d = length(hyperparams.mu)
        new_clusters = Vector{Cluster}()
        
        for cluster in clusters
            new_cluster = Cluster(d + 1)
            push!(new_clusters, new_cluster)
            for element in cluster
                tp = tail_prob(element)
                tp = tp > 0.0 ? tp : 0.65 * rand() / rejection_samples
                tp = tp < 1.0 ? tp : 1.0 - 0.65 * rand() / rejection_samples
                push!(new_cluster, vcat(element, logit(tp)))
            end
        end

        empty!(clusters)
        append!(clusters, new_clusters)



        # for element in elements(clusters)
        #     push!(element, logit(tail_prob(element)))
        # end

        # for cluster in clusters
        #     cluster.sum_x, cluster.sum_xx = calculate_sums(cluster)
        #     cluster.mu_c_volatile = Array{Float64}(undef, size(cluster.sum_x))
        #     cluster.psi_c_volatile = Array{Float64}(undef, size(cluster.sum_xx))
        # end

        add_one_dimension!(hyperparams)

        if optimize
            optimize_hyperparams!(clusters, hyperparams, verbose=true)
        end

    end
    
    function update_tail_probabilities!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; rejection_samples=10000)
        
        rd = length(hyperparams.mu) - 1
        reduced_clusters = Vector{Cluster}()
        
        for cluster in clusters
            new_cluster = Cluster(rd)
            push!(reduced_clusters, new_cluster)
            for element in cluster
                push!(new_cluster, element[1:rd])
            end
        end

        reduced_hyperparams = deepcopy(hyperparams)
        reduced_hyperparams.mu = reduced_hyperparams.mu[1:rd]
        reduced_hyperparams.psi = reduced_hyperparams.psi[1:rd, 1:rd]
        psi!(reduced_hyperparams, reduced_hyperparams.psi)
        reduced_hyperparams.nu -= 1.0

        tail_prob = tail_probability(reduced_clusters, reduced_hyperparams, rejection_samples=rejection_samples)

        d = length(hyperparams.mu)
        updated_clusters = Vector{Cluster}()
        for cluster in reduced_clusters
            new_cluster = Cluster(d)
            push!(updated_clusters, new_cluster)
            for element in cluster
                tp = tail_prob(element)
                tp = tp > 0.0 ? tp : 0.65 * rand() / rejection_samples
                tp = tp < 1.0 ? tp : 1.0 - 0.65 * rand() / rejection_samples
                push!(new_cluster, vcat(element, logit(tp)))
            end
        end

        empty!(clusters)
        append!(clusters, updated_clusters)

    end

end
