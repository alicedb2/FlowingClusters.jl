module MultivariateNormalCRP
    using Distributions: MvNormal, MvTDist, InverseWishart, Normal, Cauchy, Uniform
    using Random: randperm, shuffle, shuffle!, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, var, Weights, std, cov, percentile, quantile, median, iqr, scattermat, fit, Histogram, autocor, autocov
    using LinearAlgebra: logdet, det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd, I
    using SpecialFunctions: loggamma, polygamma
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

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, empty!
    import Base: iterate, deepcopy, copy, sort, in, first
    import Base: show, split

    export Cluster, elements
    export initiate_chain, advance_chain!, attempt_map!, burn!
    export clear_diagnostics!, diagnostics
    export log_Pgenerative, stats, ess
    # export drawNIW
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain, logprob_chain, nbclusters_chain, largestcluster_chain
    export plot, covellipses!
    export project_clusters, project_cluster, project_hyperparams, project_mu, project_psi
    export local_covprec, local_covariance, local_precision, local_covariance_summary
    export eigen_mode, importances, local_geometry, map_cluster_assignment_idx
    export response_correlation
    export tail_probability, tail_probability_summary, predictive_logpdf
    export presence_probability, presence_probability_summary
    # export wasserstein2_distance, wasserstein1_distance_bound
    export minimum_size

    include("types/diagnostics.jl")
    include("types/hyperparams.jl")
    include("types/cluster.jl")
    include("types/chain.jl")

    include("types/dataset.jl")
    using .Dataset
    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes, standardize_with, split

    include("naivebioclim.jl")
    using .NaiveBIOCLIM
    export bioclim_predictor

    # Quick and dirty and faster logdet
    # for positive-definite matrix
    function logdetpsd(A::Matrix{Float64})
        chol = cholesky(Symmetric(A))
        acc = 0.0
        for i in 1:size(A, 1)
            acc += log(chol.U[i, i])
        end
        return 2 * acc
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
    
    # function drawNIW(
    #     mu::Vector{Float64}, 
    #     lambda::Float64, 
    #     psi::Matrix{Float64}, 
    #     nu::Float64)
    
    #     invWish = InverseWishart(nu, psi)
    #     sigma = rand(invWish)
    
    #     multNorm = MvNormal(mu, sigma/lambda)
    #     mu = rand(multNorm)
    
    #     return mu, sigma
    # end

    # function drawNIW(hyperparams::MNCRPHyperparams)
    #     return drawNIW(hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
    # end

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
        # log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi)
        # log_denominator = d/2 * log(lambda) + nu/2 * log(det(psi))

        return log_numerator - log_denominator
    
    end

    function log_Pgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; hyperpriors=true)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        return log_Pgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors)
    end

    function log_Pgenerative(clusters::Vector{Cluster}, theta::Vector{Float64}; hyperpriors=true)
        alpha, mu, lambda, flatL, L, psi, nu = pack(theta)
        return log_Pgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors)
    end

    # Return the log-likelihood of the model
    function log_Pgenerative(clusters::Vector{Cluster}, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; hyperpriors=true)
    
        @assert all(length(c) > 0 for c in clusters)
        
        N = sum([length(c) for c in clusters])
        K = length(clusters)

        # alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)
    
        # Log-probability associated with the Chinese Restaurant Process
        log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum([loggamma(length(c)) for c in clusters])
        
        # Log-probability associated with the data likelihood
        # and Normal-Inverse-Wishart base distribution of the CRP
        log_niw = 0.0
        for cluster in clusters
            log_niw += log_Zniw(cluster, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - length(cluster) * d/2 * log(2pi) 
        end

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

        return log_crp + log_niw + log_hyperpriors
    
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
        log_weight += log_Zniw(cluster, mu, lambda, psi, nu) # - log_Zniw(nothing, mu, lambda, psi, nu)
        pop!(cluster, element)
        log_weight -= log_Zniw(cluster, mu, lambda, psi, nu) # - log_Zniw(nothing, mu, lambda, psi, nu)
        log_weight -= d/2 * log(2pi)
        
        return log_weight

    end 
    

    function advance_gibbs!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; temperature=1.0)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        scheduled_elements = shuffle!([el for cluster in clusters for el in cluster])
    
        for e in scheduled_elements

            pop!(clusters, e)
            
            if sum(isempty.(clusters)) < 1
                push!(clusters, Cluster(d))
            end

            log_weights = zeros(length(clusters))
            for (i, cluster) in enumerate(clusters)
                log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
            end
            
            if temperature > 0.0
                unnorm_logp = log_weights / temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))
                new_assignment = sample(clusters, probs)
            elseif temperature == 0.0
                _, max_idx = findmax(log_weights)
                new_assignment = clusters[max_idx]
            end
            
            push!(new_assignment, e)

        end

        filter!(!isempty, clusters)

        return clusters
    
    end


    # Sequential splitmerge from Dahl & Newcomb
    function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; t=3)

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

                unnorm_logp = log_weights
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))
                new_assignment, log_transition = sample(collect(zip(proposed_state, norm_logp)), 
                                                         probs)
                
                push!(new_assignment, e)

                log_q += log_transition
                            
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

        if ci != cj
            log_acceptance += log_q
        elseif ci == cj
            # print("$(round(log_acceptance, digits=1)), $(round(-log_q)), ")
            log_acceptance -= log_q
            # println("$(round(log_acceptance, digits=1))")
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

    function log_slicef_flatL(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, flatL::Vector{Float64})
        mu, lambda, nu = hyperparams.mu, hyperparams.lambda, hyperparams.nu
        d = length(mu)
        K = length(clusters)
        
        L = foldflat(flatL)
        psi = L * L'

        ret = (K * nu / 2 - d) * 2 * sum(log.(diag(L)))
        for cluster in clusters
            _, _, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
            ret -= nu_c / 2 * logdetpsd(psi_c)
        end

        return ret
    end

    function advance_psi_slice!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)

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

    function optimal_step_scale_local(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; optimize=true, verbose=false)::Matrix{Float64}
        d = length(hyperparams.mu)
        D = 3 + d + div(d * (d + 1), 2)

        if optimize
            hyperparams = optimize_hyperparams(clusters, hyperparams, verbose=verbose)
        end

        theta = unpack(hyperparams, transform=true)
        if verbose
            println()
            println(" * Calculating Hessian")
        end
        hessian = finite_difference_hessian(x -> log_Pgenerative(clusters, x; hyperpriors=true), theta)
        return -(2.38^2/D) * inv(hessian)
    end

    function optimal_step_scale_adaptive(chain::MNCRPChain; n=1, epsilon=1e-2)::Matrix{Float64}
        d = length(chain.hyperparams.mu)
        D = 3 + d + div(d * (d + 1), 2)
        emp_sigma = cov(unpack.(chain.hyperparams_chain[1:n:end]))        
        return 2.38^2 / D * (emp_sigma + epsilon / D * I(D))
    end

    function advance_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, step_scale::Matrix{Float64})
        alpha, mu, lambda, flatL, L, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.flatL, hyperparams.L, hyperparams.psi, hyperparams.nu
        d = length(mu)
        log_alpha = log(alpha)
        log_lambda = log(lambda)
        log_x = log(nu - d + 1)
        current_theta = unpack(hyperparams, transform=true)
        
        thetaD = 3 + d + div(d * (d + 1), 2)
        delta_theta = rand(MvNormal(zeros(thetaD), step_scale))
        proposed_theta = current_theta + delta_theta
        proposed_alpha, proposed_mu, proposed_lambda, proposed_flatL, proposed_L, proposed_psi, proposed_nu = pack(proposed_theta, backtransform=true)

        log_metropolis = log_Pgenerative(clusters, proposed_alpha, proposed_mu, proposed_lambda, proposed_psi, proposed_nu, hyperpriors=true)
        log_metropolis -= log_Pgenerative(clusters, alpha, mu, lambda, psi, nu, hyperpriors=true)

        log_hastings = log(proposed_alpha) - log(alpha)
        log_hastings += sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(L)))))
        log_hastings += log(proposed_nu - d + 1) - log_x

        log_acc = min(0.0, log_metropolis + log_hastings)
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

    # Very cute!
    function jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

    end

    # Very cute as well!
    function jeffreys_nu(nu::Float64, d::Int64)

        return sqrt(1/4 * sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

    end

    function initiate_chain(filename::AbstractString)
        # Expects a JLD2 file
        return load(filename)["chain"]
    end

    function initiate_chain(dataset::MNCRPDataset; chain_samples=200, strategy=:hot)
        chain = initiate_chain(dataset.data, chain_samples=chain_samples, standardize=false, strategy=strategy)
        chain.data_zero = dataset.data_zero[:]
        chain.data_scale = dataset.data_scale[:]
        return chain
    end

    function initiate_chain(data::Vector{Vector{Float64}}; standardize=true, chain_samples=100, strategy=:hot)

        @assert all(size(e, 1) == size(first(data), 1) for e in data)

        N = length(data)

        d = size(first(data), 1)

        hyperparams = MNCRPHyperparams(d)

        clusters_samples = CircularBuffer{Vector{Cluster}}(chain_samples)
        hyperparams_samples = CircularBuffer{MNCRPHyperparams}(chain_samples)

        
        # Keep unique observations only in case we standardize
        data = Set{Vector{Float64}}(data)
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
        end
        println("    Initializing hyperparameters...")
        optimize_hyperparams!(chain.clusters, chain.hyperparams, verbose=true)
        chain.hyperparams.diagnostics.step_scale = optimal_step_scale_local(chain.clusters, chain.hyperparams, optimize=true)

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
        nb_splitmerge=30, splitmerge_t=3,
        nb_gibbs=1, gibbs_temp=1.0,
        nb_mhhyperparams=10, mh_stepscale=1.0,
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

        for step in 1:nb_steps

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


            
            # if length(chain.logprob_chain) < 20
            #     d = length(chain.hyperparams.mu)
            #     D = 3 + d + div(d * (d + 1), 2)
            #     mh_stepscale = 0.01/D * diagm(ones(D))
            # else
            #     if mod(length(chain.logprob_chain), 10) == 0 || step == 1
            #         mh_stepscale = optimal_step_scale_adaptive(chain)
            #     end
            # end

            for i in 1:nb_mhhyperparams
                advance_hyperparams!(chain.clusters, chain.hyperparams, chain.hyperparams.diagnostics.step_scale)
            end

            push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

            # Sequential split-merge            
            for i in 1:nb_splitmerge
                advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t)
            end

            # Gibbs sweep
            for i in 1:nb_gibbs
                advance_gibbs!(chain.clusters, chain.hyperparams, temperature=gibbs_temp)
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
            
            map_success = false
            logp_quantile95 = quantile(short_logprob_chain, 0.95)

            if (logprob > logp_quantile95 || rand() < 0.05) && attempt_map
                    # Summit attempt
                    nb_map_attemps += 1
                    try
                        map_success = attempt_map!(chain, max_nb_pushes=10)
                    catch e
                        map_success = false
                    end

                    # if map_success
                    #     new_step_scale = optimal_step_scale_local(chain.map_clusters, chain.map_hyperparams)
                    #     if all(eigen(new_step_scale).values .> 0.0)
                    #         chain.hyperparams.diagnostics.step_scale = new_step_scale
                    #     end
                    # end

                    nb_map_successes += map_success
                    last_map_idx = chain.map_idx
            end


            if mod(length(chain.logprob_chain), sample_every) == 0
                push!(chain.clusters_samples, deepcopy(chain.clusters))
                push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
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
                (:"step", "$(step)/$(nb_steps)"),
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
                print("   s:" * split_ratio * " m:" * merge_ratio * " ($(_nb_splitmerge))")
                print("   sr:$(split_per_step), mr:$(merge_per_step)")
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


    function attempt_map!(chain::MNCRPChain; max_nb_pushes=15, optimize_hyperparams=true)
            
        map_clusters_attempt = copy(chain.clusters)
        map_hyperparams = deepcopy(chain.hyperparams)

        map_mll = log_Pgenerative(map_clusters_attempt, chain.map_hyperparams)
        
        # if optimize_hyperparams
        #     optimize_hyperparams!(map_clusters_attempt, map_hyperparams)
        # end

        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = copy(map_clusters_attempt)
            advance_gibbs!(test_attempt, map_hyperparams; temperature=0.0)
            if optimize_hyperparams
                optimize_hyperparams!(map_clusters_attempt, map_hyperparams, verbose=true)
            end
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


        attempt_logprob = log_Pgenerative(map_clusters_attempt, map_hyperparams)
        if attempt_logprob > chain.map_logprob
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

        if nb_clusters === nothing || nb_clusters > length(clusters) || nb_clusters <= 0
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

    function stats(chain::MNCRPChain; burn=0)
        println("MAP state")
        println(" log prob: $(chain.map_logprob)")
        println(" #cluster: $(length(chain.map_clusters))")
        println("    alpha: $(chain.map_hyperparams.alpha)")
        println("       mu: $(chain.map_hyperparams.mu)")
        println("   lambda: $(chain.map_hyperparams.lambda)")
        println("      psi:")
        display(chain.map_hyperparams.psi)
        println("       nu: $(chain.map_hyperparams.nu)")
        println()


        nbc = chain.nbclusters_chain[burn+1:end]
        ac = alpha_chain(chain)[burn+1:end]
        muc = mu_chain(chain)[burn+1:end]
        lc = lambda_chain(chain)[burn+1:end]
        psic = psi_chain(chain)[burn+1:end]
        nc = nu_chain(chain)[burn+1:end]

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

    function optimize_hyperparams(clusters::Vector{Cluster}, hyperparams0::MNCRPHyperparams; verbose=false)

        objfun(x) = -log_Pgenerative(clusters, x)

        x0 = unpack(hyperparams0)

        if verbose
            function callback(x)
                print(" * Iter $(x.iteration),   objfun $(-round(x.value, digits=2)),   g_norm $(round(x.g_norm, digits=8))\r")
                return false
            end
        else
            callback =  nothing
        end
        opt_options = Options(iterations=9999, 
                              f_tol=1e-5,
                              callback=callback)

        optres = optimize(objfun, x0, LBFGS(), opt_options)

        opt_hp = pack(minimizer(optres))

        return MNCRPHyperparams(opt_hp..., deepcopy(hyperparams0.diagnostics))

    end

    function optimize_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; verbose=false)
        
        opt_res = optimize_hyperparams(clusters, hyperparams, verbose=verbose)

        hyperparams.alpha = opt_res.alpha
        hyperparams.mu = opt_res.mu
        hyperparams.lambda = opt_res.lambda
        hyperparams.flatL = opt_res.flatL
        hyperparams.L = opt_res.L
        hyperparams.psi = opt_res.psi
        hyperparams.nu = opt_res.nu

        return hyperparams
    
    end


    function updated_mvstudent_params(cluster::Cluster, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)

        d = length(mu)
        mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

        return (nu_c - d + 1, mu_c, (lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c)

    end

    function updated_mvstudent_params(clusters::Vector{Cluster}, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; add_empty=true)
        d = length(mu)
        updated_mvstudent_degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]
        if add_empty
            push!(updated_mvstudent_degs_mus_sigs, updated_mvstudent_params(Cluster(d), mu, lambda, psi, nu))
        end

        return updated_mvstudent_degs_mus_sigs
    end

    function predictive_logpdf(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        weights = 1.0 .* length.(clusters)
        push!(weights, alpha)
        weights ./= sum(weights)
        
        updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)

        logpdf(x) = logsumexp(log(weight) + loggamma((deg + d)/2) - loggamma(deg/2)
                    - d/2 * log(deg) - d/2 * log(pi) - 1/2 * logdetpsd(sig)
                    # - d/2 * log(deg) - d/2 * log(pi) - 1/2 * logdet(sig)
                    - (deg + d)/2 * log(1 + 1/deg * (x - mu)' * (sig \ (x - mu)))
                    # - (deg + d)/2 * log(1 + 1/deg * (x - mu)' * inv(sig) * (x - mu))
                    for (weight, (deg, mu, sig)) in zip(weights, updated_mvstudent_degs_mus_sigs))
        return logpdf
    end

    
    function tail_probability(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, rejection_samples=10000)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)
        
        weights = 1.0 * length.(clusters)
        push!(weights, alpha)
        weights ./= sum(weights)

        updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)
        
        logpdf = predictive_logpdf(clusters, hyperparams)
        
        sample_logpdfs = zeros(rejection_samples)
        for i in 1:rejection_samples
            sample_deg, sample_mu, sample_sig = sample(updated_mvstudent_degs_mus_sigs, Weights(weights))
            sample_draw = rand(MvTDist(sample_deg, sample_mu, sample_sig))
            sample_logpdfs[i] = logpdf(sample_draw)
        end

        function tailprob_func(coordinate::Vector{Float64})
            log_isocontour_val = logpdf(coordinate)
            tail = sum(sample_logpdfs .<= log_isocontour_val) / rejection_samples
            return tail
        end
        
        return tailprob_func
        
    end
    
    function tail_probability(coordinates::Vector{Vector{Float64}}, clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, rejection_samples=10000)
        tprob = tail_probability(clusters, hyperparams, rejection_samples) 
        return tprob.(coordinates)
    end

    
    function tail_probability(coordinates::Vector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}}, hyperparams_samples::AbstractVector{MNCRPHyperparams}, rejection_samples=10000; nb_samples=nothing)
        
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
            push!(tail_probs_samples, tail_probability(coordinates, clusters, hyperparams, rejection_samples))
        end
        println()
        
        # transpose
        tail_probs_distributions = collect.(eachrow(reduce(hcat, tail_probs_samples)))
        
        return tail_probs_distributions
        
    end
    
    function tail_probability_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}},  hyperparams_samples::AbstractVector{MNCRPHyperparams}, rejection_samples=10000; nb_samples=nothing)
        
        tail_probs_distributions = tail_probability(coordinates, 
                                                    clusters_samples, hyperparams_samples, 
                                                    rejection_samples, nb_samples=nb_samples)
        
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
    
    function tail_probability_summary(coordinates::Vector{Vector{Float64}}, chain::MNCRPChain, rejection_samples=10000; nb_samples=nothing)
        return tail_probability_summary(coordinates, chain.clusters_samples, chain.hyperparams_samples, rejection_samples, nb_samples=nb_samples)
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

end
