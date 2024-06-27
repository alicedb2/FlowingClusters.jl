module MultivariateNormalCRP
    using Distributions: MvNormal, MvTDist, InverseWishart, Normal, Cauchy, Uniform, Exponential, Dirichlet, Multinomial, Beta, MixtureModel, Categorical, Distribution, logpdf
    using Random: randperm, shuffle, shuffle!, seed!, Xoshiro
    using StatsFuns: logsumexp, logmvgamma, logit, logistic
    using StatsBase: sample, mean, var, Weights, std, cov, percentile, quantile, median, iqr, scattermat, fit, Histogram, autocor, autocov
    using LinearAlgebra: logdet, det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd, I, diagind, dot, issuccess
    using PDMats
    using SpecialFunctions: loggamma, polygamma, logbeta
    using Base.Iterators: cycle
    import Base.Iterators: flatten
    # using StatsPlots: covellipse!, covellipse
    using Makie
    import Makie: plot, plot!
    using JLD2, CodecBzip2
    using ProgressMeter
    using FiniteDiff: finite_difference_hessian
    using Optim: optimize, minimizer, LBFGS, NelderMead, Options
    using DataStructures: CircularBuffer
    import MCMCDiagnosticTools: ess_rhat
    
    using DiffEqFlux
    using ComponentArrays: ComponentArray
    using DifferentialEquations
    import Optimization, OptimizationOptimisers

    import Base: pop!, push!, length, isempty, union, delete!, empty!
    import Base: iterate, deepcopy, copy, sort, in, first

    export updated_niw_hyperparams, updated_mvstudent_params
    
    export advance_chain!
    export advance_hyperparams_adaptive!
    export advance_ffjord!

    export attempt_map!, burn!
    export logprobgenerative
    export plot, plot!, covellipses!
    export project_clusters, project_cluster, project_hyperparams, project_mu, project_psi
    export local_covprec, local_covariance, local_precision, local_covariance_summary
    export eigen_mode, importances, local_geometry, map_cluster_assignment_idx
    export tail_probability, tail_probability_summary, predictive_distribution, clustered_tail_probability
    export presence_probability, presence_probability_summary
    export optimize_hyperparams, optimize_hyperparams!
    export logdetpsd

    include("types/diagnostics.jl")
    export Diagnostics
    export clear_diagnostics!, diagnostics, acceptance_rates

    include("types/hyperparams.jl")
    export MNCRPHyperparams, pack, unpack, ij, set_theta!, get_theta, dimension, param_dimension

    include("types/cluster.jl")
    export Cluster
    export elements
    export realspace_cluster, realspace_clusters

    include("types/dataset.jl")
    using .Dataset
    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes, standardize_with, standardize_with!, split, standardize!

    include("types/chain.jl")
    export MNCRPChain
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain, logprob_chain, nbclusters_chain, largestcluster_chain, nn_chain
    export ess_rhat, stats

    include("naivebioclim.jl")
    using .NaiveBIOCLIM
    export bioclim_predictor

    include("helpers.jl")
    export performance_scores, drawNIW
    export sqrtsigmoid, sqrttanh, sqrttanhgrow

    include("MNDPVariational.jl")

    # Quick and dirty but faster logdet
    # for positive-definite matrix
    function logdetpsd(A::AbstractMatrix{Float64})
        chol = cholesky(Symmetric(A), check=false)
        if issuccess(chol)
            # marginally faster than 
            # 2 * sum(log.(diag(chol.U)))
            acc = 0.0
            for i in 1:size(A, 1)
                acc += log(chol.U[i, i])
            end
            return 2 * acc
        else
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

            # d = length(mu)

            # @inbounds for j in 1:d
            #     cluster.mu_c_volatile[j] = mu[j]
            #     for i in 1:j
            #         cluster.psi_c_volatile[i, j] = psi[i, j]
            #         cluster.psi_c_volatile[j, i] = psi[i, j]
            #     end
            # end
            
            # return (cluster.mu_c_volatile, lambda, cluster.psi_c_volatile, nu)
            
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
    
        log_denominator = d/2 * log(lambda) + nu/2 * logdetpsd(psi) # + nu/2 * log(det(psi))

        return log_numerator - log_denominator

    end

    # Very cute!
    function jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

    end

    # Very cute as well!
    function jeffreys_nu(nu::Float64, d::Int64)

        return 1/2 * sqrt(sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

    end

    function logprobgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, base2original::Union{Nothing, Dict{Vector{Float64}, Vector{Float64}}}=nothing; hyperpriors=true, temperature=1.0, ffjord=false)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        logp = logprobgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors, temperature=temperature)
        if ffjord && hyperparams.nn !== nothing
            ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
            origmat = reduce(hcat, [base2original[el] for el in elements(clusters)], init=zeros(Float64, dimension(hyperparams), 0))
            ret, _ = ffjord_model(origmat, hyperparams.nn_params, hyperparams.nn_state)
            logp -= sum(ret.delta_logp)

            logp += logpdf(MvNormal(6^2 * I(size(hyperparams.nn_params, 1))), hyperparams.nn_params)
        end

        return logp
    end

    # Theta is assumed to be a concatenated vector of coordinates
    # i.e. vcat(log(alpha), mu, log(lambda), flatL, log(nu -d + 1))
    function logprobgenerative(clusters::Vector{Cluster}, theta::Vector{Float64}; hyperpriors=true, backtransform=true, jacobian=false, temperature=1.0)
        alpha, mu, lambda, flatL, L, psi, nu = pack(theta, backtransform=backtransform)
        log_p = logprobgenerative(clusters, alpha, mu, lambda, psi, nu; hyperpriors=hyperpriors, temperature=temperature) 
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
    function logprobgenerative(clusters::Vector{Cluster}, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; hyperpriors=true, temperature=1.0)

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
        end

        return clusters
    
    end


    # Sequential splitmerge from Dahl & Newcomb
    function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; t=3, temperature=1.0)

        @assert t >= 0
        
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
                    new_assignment, log_transition = sample(collect(zip(proposed_state, norm_logp)), probs)                    
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

        log_acceptance = (logprobgenerative(proposed_state, hyperparams, hyperpriors=false) 
                        - logprobgenerative(initial_state, hyperparams, hyperpriors=false))

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

    function advance_alpha!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; step_size=0.5)
            
        step_distrib = Normal(0.0, step_size)

        N = sum([length(c) for c in clusters])
        K = length(clusters)

        alpha = hyperparams.alpha
    
        # 1/x improper hyperprior on alpha
        proposed_logalpha = log(alpha) + rand(step_distrib)
        proposed_alpha = exp(proposed_logalpha)
        
        log_acceptance = 0.0

        # because we propose moves on the log scale
        # but need them uniform over alpha > 0
        # before feeding them to the hyperprior

        log_acceptance += K * log(proposed_alpha) - loggamma(proposed_alpha + N) + loggamma(proposed_alpha)
        log_acceptance -= K * log(alpha) - loggamma(alpha + N) + loggamma(alpha)

        log_acceptance += log(jeffreys_alpha(proposed_alpha, N)) - log(jeffreys_alpha(alpha, N))

        log_hastings = proposed_logalpha - log(alpha)
        log_acceptance += log_hastings

        log_acceptance = min(0.0, log_acceptance)

        if log(rand()) < log_acceptance
            hyperparams.alpha = proposed_alpha
            hyperparams.diagnostics.accepted_alpha += 1
        else
            hyperparams.diagnostics.rejected_alpha += 1
        end
    
    end

    function advance_mu!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                         random_order=true, step_size=fill(0.1, dimension(hyperparams)))
    
        d = dimension(hyperparams)

        step_distrib = MvNormal(diagm(step_size.^2))
        steps = rand(step_distrib)

        lambda, psi, nu = hyperparams.lambda, hyperparams.psi, hyperparams.nu
                
        if random_order
            dim_order = randperm(d)
        else
            dim_order = 1:d
        end    
                
        for i in dim_order
            proposed_mu = deepcopy(hyperparams.mu)
            proposed_mu[i] = proposed_mu[i] + steps[i]

            log_acceptance = (
            sum(log_Zniw(c, proposed_mu, lambda, psi, nu) - log_Zniw(nothing, proposed_mu, lambda, psi, nu)
              - log_Zniw(c, hyperparams.mu, lambda, psi, nu) + log_Zniw(nothing, hyperparams.mu, lambda, psi, nu) 
            for c in clusters)
            )
                        
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                hyperparams.mu = proposed_mu
                hyperparams.diagnostics.accepted_mu[i] += 1
            else
                hyperparams.diagnostics.rejected_mu[i] += 1
            end
            
        end
            
    end

    function advance_lambda!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; step_size=0.1)

        step_distrib = Normal(0.0, step_size)
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
    
        proposed_loglambda = log(lambda) + rand(step_distrib)
        proposed_lambda = exp(proposed_loglambda)
        
        log_acceptance = sum(log_Zniw(c, mu, proposed_lambda, psi, nu) - log_Zniw(nothing, mu, proposed_lambda, psi, nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                     for c in clusters)

        # We leave loghastings = 0.0 because the
        # Jeffreys prior over lambda is the logarithmic 
        # prior and moves are symmetric on the log scale.

        log_acceptance = min(0.0, log_acceptance)
        
        if log(rand()) < log_acceptance
            hyperparams.lambda = proposed_lambda
            hyperparams.diagnostics.accepted_lambda += 1
        else
            hyperparams.diagnostics.rejected_lambda += 1
        end
    
    end

    function advance_psi!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams;
                          random_order=true, step_size=fill(0.1, length(hyperparams.flatL)))
        
        flatL_d = length(hyperparams.flatL)
    
        step_distrib = MvNormal(diagm(step_size.^2))
        steps = rand(step_distrib)
            
        if random_order
            dim_order = randperm(flatL_d)
        else
            dim_order = 1:flatL_d
        end
    
        mu, lambda, nu = hyperparams.mu, hyperparams.lambda, hyperparams.nu
    
        d = length(mu)
            
        for k in dim_order
            
            proposed_flatL = deepcopy(hyperparams.flatL)
        
            proposed_flatL[k] = proposed_flatL[k] + steps[k]
        
            proposed_L = foldflat(proposed_flatL)
            proposed_psi = proposed_L * proposed_L'
        
            log_acceptance = sum(log_Zniw(cluster, mu, lambda, proposed_psi, nu) - log_Zniw(nothing, mu, lambda, proposed_psi, nu)
                        - log_Zniw(cluster, mu, lambda, hyperparams.psi, nu) + log_Zniw(nothing, mu, lambda, hyperparams.psi, nu) 
                        for cluster in clusters)
                
            # Go from symmetric and uniform in L to uniform in psi
            # det(del psi/del L) = 2^d |L_11|^d * |L_22|^(d-1) ... |L_nn|
            # 2^d's cancel in the Hastings ratio
            log_hastings = sum((d:-1:1) .* (log.(abs.(diag(proposed_L))) - log.(abs.(diag(hyperparams.L)))))
            log_acceptance += log_hastings

            log_acceptance += d * (logdetpsd(hyperparams.psi) - logdetpsd(proposed_psi))
            # log_acceptance += d * (log(det(hyperparams.psi)) - log(det(proposed_psi)))
            
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                flatL!(hyperparams, proposed_flatL)
                hyperparams.diagnostics.accepted_flatL[k] += 1
            else
                hyperparams.diagnostics.rejected_flatL[k] += 1
            end
            
        end
    
    end

    function advance_nu!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; step_size=1.0)

        step_distrib = Normal(0.0, step_size)    
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        # x = nu - (d - 1)
        # we use moves on the log of x
        # so as to always keep nu > d - 1 
        current_logx = log(nu - (d - 1))
        proposed_logx = current_logx + rand(step_distrib)
        proposed_nu = d - 1 + exp(proposed_logx)
    
        log_acceptance = sum(log_Zniw(c, mu, lambda, psi, proposed_nu) - log_Zniw(nothing, mu, lambda, psi, proposed_nu)
                    - log_Zniw(c, mu, lambda, psi, nu) + log_Zniw(nothing, mu, lambda, psi, nu) 
                    for c in clusters)
        
        # Convert back to uniform moves on the positive real line nu > d - 1
        log_hastings = proposed_logx - current_logx
        log_acceptance += log_hastings

        log_acceptance += log(jeffreys_nu(proposed_nu, d)) - log(jeffreys_nu(nu, d))

        log_acceptance = min(0.0, log_acceptance)
        
        if log(rand()) < log_acceptance
            hyperparams.nu = proposed_nu
            hyperparams.diagnostics.accepted_nu += 1
        else
            hyperparams.diagnostics.rejected_nu += 1
        end
    
    end

    function advance_hyperparams_adaptive!(
        clusters::Vector{Cluster}, 
        hyperparams::MNCRPHyperparams, 
        base2original::Dict{Vector{Float64}, Vector{Float64}}; 
        ffjord_sampler=:am, 
        amwg_batch_size=50, acceptance_target=0.44, M=nothing, 
        am_safety_probability=0.05, am_safety_sigma=0.1,
        hyperparams_chain=nothing, temperature=1.0)
        
        di = hyperparams.diagnostics
        d = dimension(hyperparams)

        slice_mu = 2:1+d
        slice_psi = (2 + d + 1):(2 + d + div(d * (d + 1), 2))

        idx_nu = 3 + d + div(d * (d + 1), 2)

        if hyperparams.nn_params === nothing
            nn_D = 0
        else
            nn_D = size(hyperparams.nn_params, 1)
            orig_clusters = realspace_clusters(Matrix, clusters, base2original)
        end

        clear_diagnostics!(di, clearhyperparams=true, clearsplitmerge=false, keepstepscale=true)

        for i in 1:amwg_batch_size
                advance_alpha!(clusters, hyperparams, step_size=exp(di.amwg_logscales[1]))
                
                advance_mu!(clusters, hyperparams, step_size=exp.(di.amwg_logscales[slice_mu]))
                
                advance_lambda!(clusters, hyperparams, step_size=exp(di.amwg_logscales[2+d]))

                advance_psi!(clusters, hyperparams,step_size=exp.(di.amwg_logscales[slice_psi]))

                advance_nu!(clusters, hyperparams, step_size=exp(di.amwg_logscales[idx_nu]))

                if ffjord_sampler === :amwg && hyperparams.nn_params !== nothing
                    step_distrib = MvNormal(diagm(exp.(di.amwg_logscales[(idx_nu+1):(idx_nu+nn_D)].^2)))
                    advance_ffjord!(clusters, hyperparams, base2original, 
                                    step_type=:seq,
                                    step_distrib=step_distrib,
                                    temperature=temperature)
                end
        end

        di.amwg_nbbatches += 1
        adjust_amwg_logscales!(di, acceptance_target=acceptance_target, ffjord=ffjord_sampler === :amwg)

        if ffjord_sampler === :am && hyperparams.nn_params !== nothing

            if length(hyperparams_chain) <= 2 * nn_D
                step_distrib = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))
            else
                nn_sigma_chain = reduce(hcat, [h.nn_params for h in hyperparams_chain], init=zeros(Float64, nn_D, 0))
                sigma_n = cov(nn_sigma_chain, dims=2)
                safety_component = MvNormal(am_safety_sigma^2 / nn_D * I(nn_D))
                empirical_estimate_component = MvNormal(2.38^2 / nn_D * sigma_n)
                step_distrib = MixtureModel([safety_component, empirical_estimate_component], [am_safety_probability, 1 - am_safety_probability])
            end

            advance_ffjord!(clusters, hyperparams, base2original, 
                            step_type=:batch,
                            step_distrib=step_distrib,
                            temperature=temperature)
        end
        
        return hyperparams

    end

    function adjust_amwg_logscales!(diagnostics::Diagnostics; acceptance_target=0.44, minmax_logscale=Inf, min_delta=0.01, ffjord=false)
    
        di = diagnostics
        
        delta_n = min(min_delta, 1/sqrt(di.amwg_nbbatches))

        d = size(di.accepted_mu, 1)
        idx_nu = 3 + d + div(d * (d + 1), 2)

        if di.accepted_alpha / (di.accepted_alpha + di.rejected_alpha) < acceptance_target
            di.amwg_logscales[1] -= delta_n
        else
            di.amwg_logscales[1] += delta_n
        end

        for i in 1:d
            if di.accepted_mu[i] / (di.accepted_mu[i] + di.rejected_mu[i]) < acceptance_target
                di.amwg_logscales[1+i] -= delta_n
            else
                di.amwg_logscales[1+i] += delta_n
            end
        end

        if di.accepted_lambda / (di.accepted_lambda + di.rejected_lambda) < acceptance_target
            di.amwg_logscales[2+d] -= delta_n
        else
            di.amwg_logscales[2+d] += delta_n
        end

        for i in 1:size(di.accepted_flatL, 1)
            if di.accepted_flatL[i] / (di.accepted_flatL[i] + di.rejected_flatL[i]) < acceptance_target
                di.amwg_logscales[2+d+i] -= delta_n
            else
                di.amwg_logscales[2+d+i] += delta_n
            end
        end

        if di.accepted_nu / (di.accepted_nu + di.rejected_nu) < acceptance_target
            di.amwg_logscales[idx_nu] -= delta_n
        else
            di.amwg_logscales[idx_nu] += delta_n
        end

        if ffjord
            for i in 1:size(di.accepted_nn, 1)
                if di.accepted_nn[i] / (di.accepted_nn[i] + di.rejected_nn[i]) < acceptance_target
                    di.amwg_logscales[idx_nu+i] -= delta_n
                else
                    di.amwg_logscales[idx_nu+i] += delta_n
                end
            end
        end

        di.amwg_logscales[di.amwg_logscales .< -minmax_logscale] .= -minmax_logscale
        di.amwg_logscales[di.amwg_logscales .> minmax_logscale] .= minmax_logscale

        return di

    end

    function advance_ffjord!(
        clusters::Vector{Cluster}, 
        hyperparams::MNCRPHyperparams, 
        base2original::Dict{Vector{Float64}, Vector{Float64}}, 
        original_clusters::Union{Nothing, Vector{Matrix{Float64}}}=nothing; 
        step_type=:batch, step_distrib=nothing,
        temperature=1.0)

        if hyperparams.nn === nothing
            return hyperparams
        end

        nn_D = size(hyperparams.nn_params, 1)
        
        # step_distrib = MvNormal(diagm(step.^2))
        steps = rand(step_distrib)
        
        ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), basedist=nothing, ad=AutoForwardDiff())
        
        if original_clusters === nothing
            original_clusters = realspace_clusters(Matrix, clusters, base2original)
        end

        if step_type === :batch
            proposed_nn_params = hyperparams.nn_params .+ steps
            proposed_baseclusters = Matrix{Float64}[]
            proposed_base2original = Dict{Vector{Float64}, Vector{Float64}}()

            log_acceptance = 0.0

            # We could have left the calculation of deltalogps
            # to logprobgenerative below, but we a proposal comes
            # a new base2original so we do both at once here and
            # call logprobgenerative with ffjord=false
            for orig_cluster in original_clusters
                ret, _ = ffjord_model(orig_cluster, proposed_nn_params, hyperparams.nn_state)
                log_acceptance -= sum(ret.delta_logp)
                push!(proposed_baseclusters, Matrix{Float64}(ret.z))
                merge!(proposed_base2original, Dict(collect.(eachcol(ret.z)) .=> collect.(eachcol(orig_cluster))))
            end

            # We already accounted for the ffjord deltalogps above
            # so call logprobgenerative with ffjord=false on the
            # proposed state.
            log_acceptance += (
                logprobgenerative(Cluster.(proposed_baseclusters), hyperparams, hyperpriors=false, ffjord=false) 
              - logprobgenerative(clusters, hyperparams, base2original, hyperpriors=false, ffjord=true)
            )

            # We called logprobgenerative with ffjord=true on the current state
            # but not on the proposed state, so we need to account for the
            # prior on the neural network for the proposed state
            log_acceptance += logpdf(MvNormal(6^2 * I(nn_D)), proposed_nn_params)

            log_acceptance /= temperature

            log_acceptance = min(0.0, log_acceptance)
            if  log(rand()) < log_acceptance
                hyperparams.nn_params = proposed_nn_params
                empty!(clusters)
                append!(clusters, Cluster.(proposed_baseclusters))
                empty!(base2original)
                merge!(base2original, proposed_base2original)
                hyperparams.diagnostics.accepted_nn .+= 1
            else
                hyperparams.diagnostics.rejected_nn .+= 1
            end

        elseif step_type === :sequential

            dim_order = randperm(nn_D)
            for i in dim_order
                # proposed_nn_params = hyperparams.nn_params .+ rand(step_distrib)
                proposed_nn_params = hyperparams.nn_params .+ steps .* (1:nn_D .== i)
                proposed_baseclusters = Matrix{Float64}[]
                proposed_base2original = Dict{Vector{Float64}, Vector{Float64}}()

                log_acceptance = 0.0

                # We could have left the calculation of deltalogps
                # to logprobgenerative below, but we a proposal comes
                # a new base2original so we do both at once here and
                # call logprobgenerative with ffjord=false
                for orig_cluster in original_clusters
                    ret, _ = ffjord_model(orig_cluster, proposed_nn_params, hyperparams.nn_state)
                    log_acceptance -= sum(ret.delta_logp)
                    push!(proposed_baseclusters, Matrix{Float64}(ret.z))
                    merge!(proposed_base2original, Dict(collect.(eachcol(ret.z)) .=> collect.(eachcol(orig_cluster))))
                end

                # We already accounted for the ffjord deltalogps above
                # so call logprobgenerative with ffjord=false on the
                # proposed state.
                log_acceptance += (
                    logprobgenerative(Cluster.(proposed_baseclusters), hyperparams, hyperpriors=false, ffjord=false) 
                - logprobgenerative(clusters, hyperparams, base2original, hyperpriors=false, ffjord=true)
                )

                # We called logprobgenerative with ffjord=false
                # because we already included the ffjord deltalogps
                log_acceptance += logpdf(Normal(0, 5), proposed_nn_params[i]) - logpdf(Normal(0, 5), hyperparams.nn_params[i])

                log_acceptance = min(0.0, log_acceptance)
                if  log(rand()) < log_acceptance
                    hyperparams.nn_params = proposed_nn_params
                    empty!(clusters)
                    append!(clusters, Cluster.(proposed_baseclusters))
                    empty!(base2original)
                    merge!(base2original, proposed_base2original)
                    hyperparams.diagnostics.accepted_nn[i] += 1
                else
                    hyperparams.diagnostics.rejected_nn[i] += 1
                end
            end
        end
        
        return clusters, hyperparams

    end
    
    function advance_chain!(chain::MNCRPChain, nb_steps=100;
        nb_splitmerge=30, splitmerge_t=3, splitmerge_temp=1.0,
        nb_gibbs=1, gibbs_temp=1.0,
        nb_hyperparams=1, mh_stepscale=1.0, mh_temp=1.0, 
        ffjord_sampler=:am, ffjord_every=nothing,
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

        if pretty_progress
            progbar = Progress(nb_steps; showspeed=true)
        end

        nn_D = hp.nn_params === nothing ? 0 : size(hp.nn_params, 1)

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

            for i in 1:nb_hyperparams
                # So many ifs and elses ._.
                if chain.hyperparams.nn === nothing
                    _ffjord_sampler = nothing
                else
                    if ffjord_sampler === :amwg
                        if (
                            ffjord_every !== nothing
                            && length(chain) >= ffjord_every
                            && length(chain) % ffjord_every == 0
                           )
                            _ffjord_sampler = :amwg
                        else
                            _ffjord_sampler = nothing
                        end
                    elseif ffjord_sampler === :am
                        _ffjord_sampler = :am
                    else
                        _ffjord_sampler = nothing
                    end
                end
                
                advance_hyperparams_adaptive!(
                    chain.clusters, 
                    chain.hyperparams, 
                    chain.base2original, 
                    ffjord_sampler=_ffjord_sampler,
                    hyperparams_chain=_ffjord_sampler !== nothing ? chain.hyperparams_chain : nothing,
                    amwg_batch_size=30, acceptance_target=0.44, M=nothing
                    )
            end

            push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

            # Sequential split-merge            
            if nb_splitmerge isa Int64
                for i in 1:nb_splitmerge
                    advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
                end
            elseif nb_splitmerge isa Float64 || nb_splitmerge === :matchgibbs
                # old_nb_ssuccess = chain.hyperparams.diagnostics.accepted_split
                # old_nb_msuccess = chain.hyperparams.diagnostics.accepted_merge
                # while (chain.hyperparams.diagnostics.accepted_split == old_nb_ssuccess) && (chain.hyperparams.diagnostics.accepted_merge == old_nb_msuccess)
                #     advance_splitmerge_seq!(chain.clusters, chain.hyperparams, t=splitmerge_t, temperature=splitmerge_temp)
                # end
            end

            # Gibbs sweep
            if nb_gibbs isa Int64
                for i in 1:nb_gibbs
                    advance_gibbs!(chain.clusters, chain.hyperparams, temperature=gibbs_temp)
                end
            elseif (0.0 < nb_gibbs < 1.0) && (rand() < nb_gibbs)
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
            logprob = logprobgenerative(chain.clusters, chain.hyperparams, chain.base2original, ffjord=chain.hyperparams.nn !== nothing)
            push!(chain.logprob_chain, logprob)

            # MAP
            history_length = 500
            short_logprob_chain = chain.logprob_chain[max(1, end - history_length):end]
            
            logp_quantile95 = quantile(short_logprob_chain, 0.95)

            map_success = false # try block has its own scope

            if logprob > logp_quantile95 && attempt_map
                    nb_map_attemps += 1
                    try
                        map_success = attempt_map!(chain, max_nb_pushes=15, verbose=false)
                    catch e
                        map_success = false
                    end                    
                    
                    nb_map_successes += map_success
                    last_map_idx = chain.map_idx

            end

            if sample_every !== nothing && sample_every >= 1
                if mod(length(chain.logprob_chain), sample_every) == 0
                    push!(chain.clusters_samples, deepcopy(chain.clusters))
                    push!(chain.hyperparams_samples, deepcopy(chain.hyperparams))
                    # push!(chain.base2original, deepcopy(chain.base2original))
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
                (:"step (mh, sm, gb temps)", "$(step)/$(nb_steps) ($(round(mh_temp, digits=2)), $(round(splitmerge_temp, digits=2)), $(round(gibbs_temp, digits=2)))"),
                (:"chain length (#chain samples)", "$(length(chain.logprob_chain)) ($(length(chain.clusters_samples))/$(length(chain.clusters_samples.buffer)))"),
                (:"logprob (max, q95)", "$(round(chain.logprob_chain[end], digits=1)) ($(round(maximum(chain.logprob_chain), digits=1)), $(round(logp_quantile95, digits=1)))"),
                (:"clusters (>1, median, mean, max)", "$(length(chain.clusters)) ($(length(filter(c -> length(c) > 1, chain.clusters))), $(round(median([length(c) for c in chain.clusters]), digits=0)), $(round(mean([length(c) for c in chain.clusters]), digits=0)), $(maximum([length(c) for c in chain.clusters])))"),
                (:"split #successes/#total, merge #successes/#total", split_ratio * ", " * merge_ratio),
                (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
                (:"MAP #attempts/#successes", "$(nb_map_attemps)/$(nb_map_successes)" * (attempt_map ? "" : " (off)")),
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

        map_mll = logprobgenerative(map_clusters_attempt, chain.map_hyperparams, ffjord=false)

        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = copy(map_clusters_attempt)
            advance_gibbs!(test_attempt, map_hyperparams; temperature=0.0)
            test_mll = logprobgenerative(test_attempt, map_hyperparams, ffjord=false)
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

        attempt_logprob = logprobgenerative(map_clusters_attempt, map_hyperparams, chain.base2original, ffjord=true)
        if attempt_logprob > chain.map_logprob || force
            chain.map_logprob = attempt_logprob
            chain.map_clusters = map_clusters_attempt
            chain.map_hyperparams = map_hyperparams
            chain.map_base2original = deepcopy(chain.base2original)
            chain.map_idx = lastindex(chain.logprob_chain)
            return true
        else
            return false
        end
    end

    function plot(clusters::Vector{Cluster}; dims::Vector{Int64}=[1, 2], rev=false, nb_clusters=nothing, plot_kw...)
        fig = Figure()
        ax = Axis(fig[1, 1])
        plot!(ax, clusters, dims=dims, rev=rev, nb_clusters=nb_clusters, plot_kw...)
        return fig
    end

    function plot!(ax, clusters::Vector{Cluster}; dims::Vector{Int64}=[1, 2], rev=false, nb_clusters=nothing, plot_kw...)

        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        clusters = project_clusters(sort(clusters, by=length, rev=rev), dims)

        if nb_clusters === nothing || nb_clusters < 0
            nb_clusters = length(clusters)
        end

        for (cluster, i) in zip(clusters, 1:nb_clusters)
            x = getindex.(clusters[i], dims[1])
            y = getindex.(clusters[i], dims[2])
            scatter!(ax, x, y, label="$(length(cluster))", color=Cycled(i), plot_kw...)
        end
        axislegend(ax)

        return ax
    end
    
    function plot(chain::MNCRPChain; dims::Vector{Int64}=[1, 2], burn=0, rev=true, nb_clusters=nothing, plot_kw...)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        d = length(chain.hyperparams.mu)
        
        proj = dims_to_proj(dims, d)

        return plot(chain, proj, burn=burn, rev=rev, nb_clusters=nb_clusters; plot_kw...)
    end

    function plot(chain::MNCRPChain, proj::Matrix{Float64}; burn=0, rev=true, nb_clusters=nothing)
        
        @assert size(proj, 1) == 2 "The projection matrix should have 2 rows"
        
        N = length(chain)

        if burn >= N
            @error("Can't burn the whole chain, burn must be smaller than $N")
        end
    
        map_idx = chain.map_idx - burn

        map_marginals = project_clusters(realspace_clusters(Cluster, chain.map_clusters, chain.map_base2original), proj)
        current_marginals = project_clusters(realspace_clusters(Cluster, chain.clusters, chain.base2original), proj)

        basemap_marginals = project_clusters(chain.map_clusters, proj)
        basecurrent_marginals = project_clusters(chain.clusters, proj)

        function deco!(axis)
            hidespines!(axis, :t, :r)
            hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
            return axis
        end

        if chain.hyperparams.nn === nothing
            ysize = 1800
        else
            ysize = 2300
        end

        fig = Figure(size=(1200, ysize))

        p_map_axis = Axis(fig[1:2, 1], title="MAP state ($(length(chain.map_clusters)) clusters)")
        deco!(p_map_axis)
        plot!(p_map_axis, map_marginals; rev=rev, nb_clusters=nb_clusters)
        # axislegend(p_map_axis, framecolor=:white)

        p_current_axis = Axis(fig[1:2, 2], title="Current state ($(length(chain.clusters)) clusters)")
        deco!(p_current_axis)
        plot!(p_current_axis, current_marginals; rev=rev, nb_clusters=nb_clusters)
        # axislegend(p_current_axis, framecolor=:white)

        p_basemap_axis = Axis(fig[3:4, 1], title="Base MAP state ($(length(chain.map_clusters)) clusters)")
        deco!(p_basemap_axis)
        plot!(p_basemap_axis, basemap_marginals; rev=rev, nb_clusters=nb_clusters)
        # axislegend(p_basemap_axis, framecolor=:white)

        p_basecurrent_axis = Axis(fig[3:4, 2], title="Base current state ($(length(chain.clusters)) clusters)")
        deco!(p_basecurrent_axis)
        plot!(p_basecurrent_axis, basecurrent_marginals; rev=rev, nb_clusters=nb_clusters)
        # axislegend(p_basecurrent_axis, framecolor=:white)

        lpc = logprob_chain(chain, burn)
        logprob_axis = Axis(fig[5, 1], title="log probability chain", aspect=3)
        deco!(logprob_axis)
        lines!(logprob_axis, burn+1:N, lpc, label=nothing)
        hlines!(logprob_axis, [chain.map_logprob], label=nothing, color=:green)
        hlines!(logprob_axis, [maximum(chain.logprob_chain)], label=nothing, color=:black)
        if map_idx > 0
            vlines!(logprob_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(logprob_axis, framecolor=:white)

        nbc = nbclusters_chain(chain, burn)
        nbc_axis = Axis(fig[5, 2], title="#cluster chain", aspect=3)
        deco!(nbc_axis)
        lines!(nbc_axis, burn+1:N, nbc, label=nothing)
        if map_idx > 0
            vlines!(nbc_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(nbc_axis, framecolor=:white)
        
        lcc = largestcluster_chain(chain, burn)
        lcc_axis = Axis(fig[6, 1], title="Largest cluster chain", aspect=3)
        deco!(lcc_axis)
        lines!(lcc_axis, burn+1:N, lcc, label=nothing)
        if map_idx > 0
            vlines!(lcc_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(lcc_axis, framecolor=:white)
        
        ac = alpha_chain(chain, burn)
        alpha_axis = Axis(fig[6, 2], title="α chain", aspect=3)
        deco!(alpha_axis)
        lines!(alpha_axis, burn+1:N, ac, label=nothing)
        if map_idx > 0
            vlines!(alpha_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(alpha_axis, framecolor=:white)

        muc = mu_chain(Matrix, chain, burn)
        mu_axis = Axis(fig[7, 1], title="μ chain", aspect=3)
        deco!(mu_axis)
        for mucomponent in eachrow(muc)
            lines!(mu_axis, burn+1:N, mucomponent, label=nothing)
        end
        if map_idx > 0
            vlines!(mu_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(mu_axis, framecolor=:white)

        lc = lambda_chain(chain, burn)
        lambda_axis = Axis(fig[7, 2], title="λ chain", aspect=3)
        deco!(lambda_axis)
        lines!(lambda_axis, burn+1:N, lc, label=nothing)
        if map_idx > 0
            vlines!(lambda_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(lambda_axis, framecolor=:white)

        pc = psi_chain(Matrix, chain, burn)
        psi_axis = Axis(fig[8, 1], title="Ψ chain", aspect=3)
        deco!(psi_axis)
        for psicomponent in eachrow(pc)
            lines!(psi_axis, burn+1:N, psicomponent, label=nothing)
        end
        if map_idx > 0
            vlines!(psi_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(psi_axis, framecolor=:white)
        
        nc = nu_chain(chain, burn)
        nu_axis = Axis(fig[8, 2], title="ν chain", aspect=3)
        deco!(nu_axis)
        lines!(nu_axis, burn+1:N, nc, label=nothing)
        if map_idx > 0
            vlines!(nu_axis, [map_idx], label=nothing, color=:black)
        end
        # axislegend(nu_axis, framecolor=:white)

        if chain.hyperparams.nn !== nothing
            nnc = nn_chain(Matrix, chain, burn)
            nn_axis = Axis(fig[9:10, 1:2], title="FFJORD neural network chain")
            deco!(nn_axis)
            for p in eachrow(nnc)
                lines!(nn_axis, burn+1:N, collect(p), label=nothing)
            end
            if map_idx > 0
                vlines!(nn_axis, [map_idx], label=nothing, color=:black)
            end
            # axislegend(nn_axis, framecolor=:white)
        end

        return fig
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

    function optimize_hyperparams(
        clusters::Vector{Cluster}, 
        hyperparams0::MNCRPHyperparams; 
        jacobian=false, verbose=false
        )

        objfun(x) = -logprobgenerative(clusters, x, jacobian=jacobian)

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

        return MNCRPHyperparams(opt_hp..., deepcopy(hyperparams0.diagnostics), hyperparams0.nn, hyperparams0.nn_params, hyperparams0.nn_state)

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

    function updated_mvstudent_params(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; add_empty=true)
        return updated_mvstudent_params(clusters, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu, add_empty=add_empty)
    end

    function predictive_distribution(
        clusters::Vector{Cluster}, 
        hyperparams::MNCRPHyperparams;
        ignore_weights=false
        )
        
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

        return predictive_distribution(weights, updated_mvstudent_degs_mus_sigs)
    end

    function predictive_distribution(
        component_weights::AbstractVector{Float64}, 
        mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}}
        )

        @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

        d = length(first(mvstudent_degs_mus_sigs)[2])

        # function logpdf(x::AbstractVector{Float64})::Float64
        #     return logsumexp(log(weight) + loggamma((deg + d)/2) - loggamma(deg/2)
        #                     - d/2 * log(deg) - d/2 * log(pi) - 1/2 * logdetpsd(sig)
        #                     - (deg + d)/2 * log(1 + 1/deg * (x - mu)' * (sig \ (x - mu)))
        #                     for (weight, (deg, mu, sig)) in zip(component_weights, mvstudent_degs_mus_sigs))
        # end

        return MixtureModel(
            [MvTDist(deg, d, mu, PDMat((sig + sig')/2)) for (deg, mu, sig) in mvstudent_degs_mus_sigs],
            Categorical(component_weights))

    end

    function tail_probability(
        component_weights::AbstractVector{Float64}, 
        mvstudent_degs_mus_sigs::AbstractVector{Tuple{Float64, Vector{Float64}, Matrix{Float64}}};
        rejection_samples=10000
        )::Function
        
        @assert isapprox(sum(component_weights), 1.0) "sum(component_weights) = $(sum(component_weights)) !~= 1"

        dist = predictive_distribution(component_weights, mvstudent_degs_mus_sigs)
        # logpdf = predictive_logpdf(component_weights, mvstudent_degs_mus_sigs)

        # sample_logpdfs = zeros(rejection_samples)
        # for i in 1:rejection_samples
        #     sample_deg, sample_mu, sample_sig = sample(mvstudent_degs_mus_sigs, Weights(component_weights))
        #     sample_draw = rand(MvTDist(sample_deg, sample_mu, sample_sig))
        #     sample_logpdfs[i] = logpdf(sample_draw)
        # end
        sample_draw = rand(dist, rejection_samples)
        sample_logpdfs = logpdf(dist, sample_draw)

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
