module MultivariateNormalCRP
    using Distributions: MvNormal, MvTDist, InverseWishart, Normal, Cauchy, Uniform
    using Random: randperm, shuffle, shuffle!, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, var, Weights, std, cov, percentile, quantile, median, iqr, scattermat, fit, Histogram, autocor, autocov
    using LinearAlgebra: logdet, det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd, pinv
    using SpecialFunctions: loggamma, polygamma
    using Base.Iterators: cycle
    import Base.Iterators: flatten
    using ColorSchemes: Paired_12, tableau_20
    using Plots: plot, plot!, vline!, hline!, scatter!, @layout, grid, scalefontsizes, mm
    using StatsPlots: covellipse!, covellipse
    using JLD2
    using ProgressMeter
    using Optim: optimize, minimizer, LBFGS, NelderMead
    using DataStructures: CircularBuffer

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, empty!
    import Base: iterate, deepcopy, copy, sort, in
    import Base: show

    export Cluster, elements
    export initiate_chain, advance_chain!, attempt_map!, burn!
    export clear_diagnostics!
    export log_Pgenerative, drawNIW, stats, ess
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain, logprob_chain, nbclusters_chain, largestcluster_chain
    export plot, covellipses!
    export project_clusters, project_cluster, project_hyperparams, project_mu, project_psi
    export local_covprec, local_covariance, local_precision, local_covariance_summary
    export response_correlation
    export tail_probability, tail_probability_summary, predictive_logpdf
    # export wasserstein2_distance, wasserstein1_distance_bound
    export minimum_size

    include("types/diagnostics.jl")
    include("types/hyperparams.jl")
    include("types/cluster.jl")
    include("types/chain.jl")
    include("types/dataset.jl")
    using .Dataset
    export load_dataset, cluster_dataframe, original, longlats

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

            # mean_x = cluster.sum_x ./ n

            # mu_c = (lambda * mu + n * mean_x) / (lambda + n)
            # psi_c = psi + lambda * mu * mu' + cluster.sum_xx - lambda_c * mu_c * mu_c'

            # psi_c_2 = (psi 
            #         + sum((x - mean_x) * (x - mean_x)' for x in cluster) 
            #         + lambda * n / (lambda + n) * (mean_x - mu) * (mean_x - mu)')

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

        empty_cluster = Cluster(length(mu))

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
        
        d = length(mu)

        mu, lambda, psi, nu = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
        
        log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    
        # logdet is almost twice as slow as log(det).
        # It's probably useful in higher dimensions only.
        log_denominator = d/2 * log(lambda) + nu/2 * log(det(psi))

        return log_numerator - log_denominator
    
    end

    # Return the log-likelihood of the model
    function log_Pgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; hyperpriors=true)
    
        @assert all(length(c) > 0 for c in clusters)
        
        N = sum([length(c) for c in clusters])
        K = length(clusters)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
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
            log_hyperpriors += -d * log(det(psi))
            # nu hyperprior
            log_hyperpriors += log(jeffreys_nu(nu, d))
        end

        return log_crp + log_niw + log_hyperpriors
    
    end

    function log_cluster_weight(element::Vector{Float64}, cluster::Cluster, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; N=nothing)

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
    

    function advance_gibbs!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; temperature=1.0)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        scheduled_elements = shuffle!([el for cluster in clusters for el in cluster])
    
        for e in scheduled_elements

            pop!(clusters, e)
            
            if sum(isempty.(clusters)) == 0
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
    function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; t=3)

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
        clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

    function advance_mu!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

    function advance_lambda!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

    function advance_psi!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

            log_acc += d * (log(det(hyperparams.psi)) - log(det(proposed_psi)))
            
            log_acc = min(0.0, log_acc)
        
            if log(rand()) < log_acc
                flatL!(hyperparams, proposed_flatL)
                hyperparams.diagnostics.accepted_flatL[k] += 1
            else
                hyperparams.diagnostics.rejected_flatL[k] += 1
            end
            
        end
    
    end

    function advance_nu!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

    function initiate_chain(data::Vector{Vector{Float64}}; standardize=false, nb_samples=200, strategy=:hot)

        @assert all(size(e, 1) == size(first(data), 1) for e in data)

        N = length(data)

        d = size(first(data), 1)

        hyperparams = MNCRPhyperparams(d)

        clusters_samples = CircularBuffer{Vector{Cluster}}(nb_samples)
        hyperparams_samples = CircularBuffer{MNCRPhyperparams}(nb_samples)

        chain = MNCRPchain([], hyperparams, 
        # Dict{Vector{Float64}, Vector{<:Real}}(), # data point => original datapoint
        zeros(d), diagm(ones(d)), [], [], [], [],
        clusters_samples, hyperparams_samples,
        [], deepcopy(hyperparams), -Inf, 1)
        
        # Keep unique observations only in case we standardize
        data = Set{Vector{Float64}}(data)
        println("Loading $(length(data)) unique data points")
        data = collect(data)
        if standardize
            chain.data_mean = mean(data)
            chain.data_scalematrix = diagm(std(data))
            data = Vector{Float64}[inv(chain.data_scalematrix) * (x .- chain.data_mean) for x in data]
        end

        # chain.original_data = Dict{Vector{Float64}, Vector{<:Real}}(k => v for (k, v) in zip(data, original_data))

        if strategy == :hot
            ##### 1st initialization method: fullseq
            chain.clusters = [Cluster(data)]
            # advance_full_sequential_gibbs!(chain.clusters, chain.hyperparams, nb_samples=N, proposal_temperature=1.0, force_acceptance=true)
            for i in 1:10
                advance_gibbs!(chain.clusters, chain.hyperparams, temperature=1.2)
            end
        elseif strategy == :N
            chain.clusters = [Cluster([datum]) for datum in data]
        elseif strategy == :1
            chain.clusters = [Cluster(data)]
        end

        chain.nbclusters_chain = [length(chain.clusters)]
        chain.largestcluster_chain = [maximum(length.(chain.clusters))]
        chain.hyperparams_chain = [deepcopy(hyperparams)]

        chain.map_clusters = deepcopy(chain.clusters)
        lp = log_Pgenerative(chain.clusters, chain.hyperparams)
        chain.map_logprob = lp
        chain.logprob_chain = [lp]
        # map_hyperparams=hyperparams and map_idx=1 have already been 
        # specified when calling MNCRPchain, but let's be explicit
        chain.map_hyperparams = deepcopy(chain.hyperparams)
        chain.map_idx = 1

        chain.logprob_chain = [chain.map_logprob]

        return chain

    end

    
    function advance_chain!(chain::MNCRPchain, nb_steps=100;
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

            for i in 1:nb_mhhyperparams
                advance_alpha!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_size=0.5 * mh_stepscale[1])
                
                advance_mu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_size=0.3 * mh_stepscale[2])
                
                advance_lambda!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_size=0.5 * mh_stepscale[3])

                advance_psi!(chain.clusters, chain.hyperparams,
                            step_type=:gaussian, step_size=0.1 * mh_stepscale[4])

                advance_nu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_size=0.3 * mh_stepscale[5])
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
                (:"chain length", length(chain.logprob_chain)),
                (:"logprob (max, q95)", "$(round(chain.logprob_chain[end], digits=1)) ($(round(maximum(chain.logprob_chain), digits=1)), $(round(logp_quantile95, digits=1)))"),
                (:"clusters (>1, median, mean, max)", "$(length(chain.clusters)) ($(length(filter(c -> length(c) > 1, chain.clusters))), $(round(median([length(c) for c in chain.clusters]), digits=0)), $(round(mean([length(c) for c in chain.clusters]), digits=0)), $(maximum([length(c) for c in chain.clusters])))"),
                (:"split ratio, merge ratio", split_ratio * ", " * merge_ratio),
                (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
                (:"MAP att/succ", "$(nb_map_attemps)/$(nb_map_successes)" * (attempt_map ? "" : " (off)")),
                (:"clusters in last MAP", length(chain.map_clusters)),
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


    function attempt_map!(chain::MNCRPchain; max_nb_pushes=15, optimize_hyperparams=true)
            
        map_clusters_attempt = copy(chain.clusters)
        map_mll = log_Pgenerative(map_clusters_attempt, chain.hyperparams)
        
        hyperparams = deepcopy(chain.hyperparams)
        
        # if optimize_hyperparams
        #     optimize_hyperparams!(map_clusters_attempt, hyperparams)
        # end

        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = copy(map_clusters_attempt)
            advance_gibbs!(test_attempt, hyperparams; temperature=0.0)
            test_mll = log_Pgenerative(test_attempt, hyperparams)
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
            optimize_hyperparams!(map_clusters_attempt, hyperparams)
        end

        attempt_logprob = log_Pgenerative(map_clusters_attempt, hyperparams)
        if attempt_logprob > chain.map_logprob
            chain.map_logprob = attempt_logprob
            chain.map_clusters = map_clusters_attempt
            chain.map_hyperparams = hyperparams
            chain.map_idx = lastindex(chain.logprob_chain)
            return true
        else
            return false
        end
    end

    function plot(clusters::Vector{Cluster}; dims::Vector{Int64}=[1, 2], rev=false, plot_kw...)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        p = plot(
            legend_position=:topleft, grid=:no, 
            showaxis=:no, ticks=:true;
            plot_kw...)

        clusters = project_clusters(sort(clusters, by=length, rev=rev), dims)

        for (cluster, color) in zip(clusters, cycle(tableau_20))
            scatter!(collect(Tuple.(cluster)), label="$(length(cluster))", 
            color=color, markerstrokewidth=0)
        end
        
        # display(p)
        return p

    end

    # function plot(chain::MNCRPchain, cluster::Cluster, hyperparams::MNCRPhyperparams; eigdirs::Vector{Float64}=[1, 2], burn=0)

    #     _, evecs = eigen_mode(cluster, hyperparams)

    #     # proj = (evecs[:, end + 1 - eigdirs[1]], evecs[:, end + 1 - eigdirs[2]])
    #     proj = Matrix{Float64}(evecs[:, [end + 1 - eigdirs[1], end + 1 - eigdirs[2]]]')

    #     return plot(chain, proj, burn=burn)

    # end
    
    function plot(chain::MNCRPchain; dims::Vector{Int64}=[1, 2], burn=0, rev=false)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        d = length(chain.hyperparams.mu)
        
        proj = dims_to_proj(dims, d)

        return plot(chain, proj, burn=burn, rev=rev)
    end

    function plot(chain::MNCRPchain, proj::Matrix{Float64}; burn=0, rev=false)
        
        @assert size(proj, 1) == 2 "The projection matrix should have 2 rows"
        
        map_marginals = project_clusters(chain.map_clusters, proj)
        current_marginals = project_clusters(chain.clusters, proj)
        
        p_map = plot(map_marginals; rev=rev, title="MAP state ($(length(chain.map_clusters)) clusters)")
        p_current = plot(current_marginals; rev=rev, title="Current state ($(length(chain.clusters)) clusters)", legend=false)

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
        p_nbc = plot(burn+1:length(nbc), nbc[burn+1:end], grid=:no,label=nothing, title="#cluster chain")
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

    function eigen_mode(cluster::Cluster, hyperparams::MNCRPhyperparams)
        _, _, psi_c, nu_c = updated_niw_hyperparams(cluster, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
        d = length(hyperparams.mu)
        sigma_mode = Symmetric(psi_c / (nu_c + d + 1))
        return eigen(sigma_mode)
    end

    function covellipses!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; dims::Vector{Int64}=[1, 2], n_std=2, scalematrix=nothing, offset=nothing, type=:predictive, lowest_weight=nothing, plot_kw...)

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

    function stats(chain::MNCRPchain; burn=0)
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

    function local_covprec(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        N = sum(length(c) for c in clusters)
        
        clusters = vcat(clusters, [Cluster(d)])
        
        coordinate_set = Cluster([coordinate])
        log_parts = []
        cluster_covs = []
        cluster_precs = []
        
        for cluster in clusters
            
            log_crp_weight = (length(cluster) > 0 ? log(length(cluster)) - log(alpha + N) : log(alpha)) - log(alpha + N)

            coordinate = coordinate .* (1.0 .+ 1e-7 * rand(d))
            
            push!(cluster, coordinate)
            log_data_weight = log_Zniw(cluster, mu, lambda, psi, nu)
            _, _, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
            pop!(cluster, coordinate)
            log_data_weight -= log_Zniw(cluster, mu, lambda, psi, nu) 
            log_data_weight -= d/2 * log(2pi)
            
            push!(log_parts, log_crp_weight + log_data_weight)

            push!(cluster_covs, psi_c / (nu_c + d + 1))
            push!(cluster_precs, nu_c * inv(psi_c))

        end
            
        log_parts = log_parts .- logsumexp(log_parts)
        
        covariance = sum(cluster_covs .* exp.(log_parts))
        precision = sum(cluster_precs .* exp.(log_parts))

        return (covariance=covariance, precision=precision)

    end

    function local_covariance(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        return local_covprec(coordinate, clusters, hyperparams).covariance
    end
    
    function local_precision(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        return local_covprec(coordinate, clusters, hyperparams).precision
    end

    function local_covariance_summary(coordinate::Vector{Float64}, clusters_samples::CircularBuffer{Vector{Cluster}},  hyperparams_samples::CircularBuffer{MNCRPhyperparams})
        return local_covariance_summary(coordinates, collect(clusters_samples), collect(hyperparams_samples))
    end
    
    function local_covariance_summary(coordinate::Vector{Float64}, clusters_samples::Vector{Vector{Cluster}},  hyperparams_samples::Vector{MNCRPhyperparams})

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

    function local_covariance_summary(coordinate::Vector{Float64}, chain::MNCRPchain)
        return local_covariance_summary(coordinate, chain.clusters_samples, chain.hyperparams_samples)
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::CircularBuffer{Vector{Cluster}},  hyperparams_samples::CircularBuffer{MNCRPhyperparams})
        return local_covariance_summary(coordinates, collect(clusters_samples), collect(hyperparams_samples))    
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::Vector{Vector{Cluster}},  hyperparams_samples::Vector{MNCRPhyperparams})
        
        meds_iqrs = NamedTuple{(:median, :iqr), Tuple{Matrix{Float64}, Matrix{Float64}}}[local_covariance_summary(coordinate, clusters_samples, hyperparams_samples) for coordinate in coordinates]

        medians = Matrix{Float64}[med_iqr.median for med_iqr in meds_iqrs]
        iqrs = Matrix{Float64}[med_iqr.iqr for med_iqr in meds_iqrs]

        return (median=medians, iqr=iqrs)
    end

    function local_covariance_summary(coordinates::Vector{Vector{Float64}}, chain::MNCRPchain)
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

    
    function crp_distance(ci::Int64, cj::Int64, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
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
    
    function crp_distance_matrix(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
    
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

    function optimize_hyperparams(clusters::Vector{Cluster}, hyperparams0::MNCRPhyperparams)

        objfun(x) = -log_Pgenerative(clusters, opt_pack(x))

        optres = optimize(objfun, opt_unpack(hyperparams0), NelderMead())

        opt_hp = opt_pack(minimizer(optres))

        return opt_hp

    end

    function optimize_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
        opt_res = optimize_hyperparams(clusters, hyperparams)

        hyperparams.alpha = opt_res.alpha
        hyperparams.mu = opt_res.mu
        hyperparams.lambda = opt_res.lambda
        hyperparams.psi = opt_res.psi
        hyperparams.nu = opt_res.nu

        return hyperparams
    
    end


    function updated_mvstudent_params(cluster::Cluster, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)

        d = length(mu)
           
        mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)

        return (nu_c - d + 1, mu_c, (lambda_c + 1)/lambda_c/(nu_c - d + 1) * psi_c)

    end

    function predictive_logpdf(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        weights = 1.0 .* length.(clusters)
        degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]

        push!(weights, alpha)
        push!(degs_mus_sigs, updated_mvstudent_params(Cluster(d), mu, lambda, psi, nu))

        weights ./= sum(weights)

        logpdf(x) = logsumexp(log(w) + loggamma((deg + d)/2) - loggamma(deg/2)
                    - d/2 * log(deg) - d/2 * log(pi) - 1/2 * log(det(sig))
                    - (deg + d)/2 * log(1 + 1/deg * (x - mu)' * inv(sig) * (x - mu))
                    for (w, (deg, mu, sig)) in zip(weights, degs_mus_sigs))
        return logpdf
    end

    function tail_probability(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams, nb_samples=1000)
        return tail_probability(clusters, hyperparams, nb_samples)(coordinate)
    end

    function tail_probability(coordinates::Vector{Vector{Float64}}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams, nb_samples=1000)
        return tail_probability(clusters, hyperparams, nb_samples).(coordinates)
    end

    function tail_probability(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams, nb_samples=1000)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = length(mu)

        weights = 1.0 .* length.(clusters)
        degs_mus_sigs = [updated_mvstudent_params(cluster, mu, lambda, psi, nu) for cluster in clusters]

        push!(weights, alpha)
        push!(degs_mus_sigs, updated_mvstudent_params(Cluster(d), mu, lambda, psi, nu))

        weights ./= sum(weights)

        logpdf = predictive_logpdf(clusters, hyperparams)

        sample_logpdfs = zeros(nb_samples)
        for i in 1:nb_samples
            deg, mu, sig = sample(degs_mus_sigs, Weights(weights))
            sample_draw = rand(MvTDist(deg, mu, sig))
            sample_logpdfs[i] = logpdf(sample_draw)
        end

        function tailprob_func(coordinate::Vector{Float64})
            isocontour_val = logpdf(coordinate)
            return sum(sample_logpdfs .<= isocontour_val) / nb_samples
        end

        return tailprob_func

    end

    function tail_probability(coordinate::Vector{Float64}, chain::MNCRPchain, nb_samples=1000)
        return first(tail_probability(Vector{Float64}[coordinate], chain.clusters_samples, chain.hyperparams_samples, nb_samples))
    end

    function tail_probability(coordinates::Vector{Vector{Float64}}, chain::MNCRPchain, nb_samples=1000)
        return tail_probability(coordinates, chain.clusters_samples, chain.hyperparams_samples, nb_samples)
    end
    
    function tail_probability(coordinates::Vector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}}, hyperparams_samples::AbstractVector{MNCRPhyperparams}, nb_samples=1000)
        
        @assert length(clusters_samples) == length(hyperparams_samples)

        tail_probs_samples = Vector{Float64}[]
        for (i, (clusters, hyperparams)) in enumerate(zip(clusters_samples, hyperparams_samples))
            print("\rProcessing sample $(i)/$(length(clusters_samples))")
            push!(tail_probs_samples, tail_probability(coordinates, clusters, hyperparams, nb_samples))
        end
        println()

        tail_probs_distributions = collect.(eachrow(reduce(hcat, tail_probs_samples)))

        return tail_probs_distributions

    end

    function tail_probability_summary(coordinates::Vector{Vector{Float64}}, chain::MNCRPchain, nb_samples=1000)
        return tail_probability_summary(coordinates, chain.clusters_samples, chain.hyperparams_samples, nb_samples)
    end


    function tail_probability_summary(coordinates::Vector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}},  hyperparams_samples::AbstractVector{MNCRPhyperparams}, nb_samples=1000)
        
        tail_probs_distributions = tail_probability(coordinates, clusters_samples, hyperparams_samples, nb_samples)
        
        # hists = [fit(Histogram, dist; nbins=ceil(log2(length(dist))) + 1) for dist in tail_probs_distributions]
        # modes = zeros(length(tail_probs_distributions))
        # for i in 1:length(modes)
        #     hist = hists[i]
        #     maxweight, idx = findmax(hist.weights)
        #     midbin = (first(hist.edges)[idx] + first(hist.edges)[idx+1]) / 2
        #     modes[i] = midbin
        # end
    
        means = mean.(tail_probs_distributions)
        stds = std.(tail_probs_distributions)
        medians = median.(tail_probs_distributions)
        iqrs = iqr.(tail_probs_distributions)
        quantiles5 = quantile.(tail_probs_distributions, 0.05)
        quantiles25 = quantile.(tail_probs_distributions, 0.25)
        quantiles75 = quantile.(tail_probs_distributions, 0.75)
        quantiles95 = quantile.(tail_probs_distributions, 0.95)
        
        return (mean=means, std=stds, median=medians, iqr=iqrs, q5=quantiles5, q25=quantiles25, q75=quantiles75, q95=quantiles95)

    end

    function response_correlation(coordinates::Vector{Vector{Float64}}, clusters_samples::AbstractVector{Vector{Cluster}},  hyperparams_samples::AbstractVector{MNCRPhyperparams}, nb_samples=1000; kernel_covs=:auto)

        d = size(first(coordinates), 1)

        if kernel_covs == :auto
            println("Calculating local covariances at coordinates");flush(stdout)
            kernel_covs = local_covariance_summary(coordinates, clusters_samples, hyperparams_samples).median
        elseif typeof(kernel_covs) == Float64
            kernel_covs = Matrix{Float64}[diagm(kernel_covs^2 * ones(d)) for _ in coordinates]
        elseif typeof(kernel_covs) == Vector{Float64}
            @assert length(kernel_covs) == d
            kernel_covs = Matrix{Float64}[diagm(kernel_covs.^2) for _ in coordinates]
        elseif typeof(kernel_covs) == Matrix{Float64}
            @assert size(kernel_covs) == (d, d)
            kernel_covs = Matrix{Float64}[kernel_covs for _ in coordinates]
        end

        elements = Vector{Float64}[x for cluster in first(clusters_samples) for x in cluster]
        
        logodd(p) = log(p) - log(1 - p)
        println("Calculating tail probabilitiessc"); flush(stdout)
        elements_tail_probs = tail_probability_summary(elements, clusters_samples, hyperparams_samples, nb_samples).median
        elements_tail_logodds = logodd.(elements_tail_probs)
        elements = Vector{Float64}[element for (element, lo) in zip(elements, elements_tail_logodds) if isfinite(lo)]
        augmented_elements = Vector{Float64}[vcat(element, lo) for (element, lo) in zip(elements, elements_tail_logodds) if isfinite(lo)]
        augmented_elements_matrix = reduce(hcat, augmented_elements)    
        # coordinates = coordinates .* (1.0 .+ 1e-7 * rand(length(coordinates)))
        # coordinates_tail_probs = tail_probability_summary(coordinates, clusters_samples, hyperparams_samples, nb_samples).median
        # coordinates_tail_logodds = logodd.(coordinates_tail_probs)
        # augmented_coordinates = Vector{Float64}[vcat(coordinate, lo) for (coordinate, lo) in zip(coordinates, coordinates_tail_logodds)]
        # mask = isfinite.(coordinates_tail_logodds)

        correlations = Matrix{Float64}[]
        partial_correlations = Matrix{Float64}[]

        for (i, (coordinate, kernel_cov)) in enumerate(zip(coordinates, kernel_covs))
            if mod(i, 100) == 0
                print("\rCalculating correlations...$(i)"); flush(stdout);
            end
            weights = zeros(size(augmented_elements))
            for k in 1:length(weights)
                weights[k] = exp(-1/2 * (elements[k] .- coordinate)' * inv(kernel_cov) * (elements[k] - coordinate))
            end
            weights ./= sqrt(det(2 * pi * kernel_cov))
            
            m = mean(augmented_elements_matrix, Weights(weights), 2)
            sm = scattermat(augmented_elements_matrix .- hcat(m), Weights(weights), mean=0, dims=2)
            covariance = sm ./ sum(weights)
            # covariance = cov(augmented_elements_matrix, Weights(weights), 2)
            precision = inv(covariance)
            
            correlation = zeros(d + 1, d + 1)
            partial_correlation = zeros(d+1, d+1)
            for m in 1:d+1
                for n in 1:m
                    correlation[m, n] = covariance[m, n] / sqrt(covariance[m, m] * covariance[n, n])
                    correlation[n, m] = correlation[m, n]

                    partial_correlation[m, n] = -precision[m, n] / sqrt(precision[m, m] * precision[n, n])
                    partial_correlation[n, m] = partial_correlation[m, n]
                end
            end
            push!(correlations, correlation)
            push!(partial_correlations, partial_correlation)
        end

        return (correlation=correlations, partial_correlation=partial_correlations)
    end


    function response_correlation(coordinates::Vector{Vector{Float64}}, chain::MNCRPchain, nb_samples=1000; kernel_covs=kernel_covs)
        return response_correlation(coordinates, chain.clusters_samples, chain.hyperparams_samples, nb_samples; kernel_covs=kernel_covs)
    end
end
