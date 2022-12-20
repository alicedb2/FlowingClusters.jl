module MultivariateNormalCRP
    using Distributions: MvNormal, InverseWishart, Normal, Cauchy, Uniform, Multinomial
    using Random: randperm, shuffle, shuffle!, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, Weights, std, percentile, quantile, median
    using LinearAlgebra: det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd
    using SpecialFunctions: loggamma, polygamma
    using Base.Iterators: cycle
    import Base.Iterators: flatten
    using ColorSchemes: Paired_12, tableau_20
    using Plots: plot, plot!, vline!, hline!, scatter!, @layout, grid, scalefontsizes, mm
    using StatsPlots: covellipse!
    using JLD2
    using ProgressMeter
    using Optim: optimize, minimizer, LBFGS

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, empty!, iterate, deepcopy, copy, sort, in

    export Cluster
    export initiate_chain, advance_chain!, attempt_map!, burn!
    export clear_diagnostics!
    export log_Pgenerative, drawNIW, stats
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export plot, covellipses!, project_clusters, project_hyperparams
    export local_average_covprec, crp_distance, crp_distance_matrix
    export presence_probability, average_tail_probability
    export wasserstein2_distance, wasserstein1_distance_bound

    include("types/diagnostics.jl")
    include("types/hyperparams.jl")
    include("types/cluster.jl")
    include("types/chain.jl")

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

        # @assert size(mu, 1) == size(psi, 1) == size(psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
  
        if isempty(cluster)

            return (mu, lambda, psi, nu)
            
        else
            
            d = size(mu, 1)
            n = length(cluster)

            lambda_c = lambda + n
            nu_c = nu + n

            mean_x = cluster.sum_x / n

            # mu_c = (lambda * mu + n * mean_x) / (lambda + n)
            # psi_c = psi + lambda * mu * mu' + cluster.sum_xx - lambda_c * mu_c * mu_c'

            # psi_c_2 = (psi 
            #         + sum((x - mean_x) * (x - mean_x)' for x in cluster) 
            #         + lambda * n / (lambda + n) * (mean_x - mu) * (mean_x - mu)')

            mu_c = Array{Float64}(undef, d)
            for i in 1:d
                mu_c[i] = (lambda * mu[i] + n * mean_x[i]) / (lambda + n)
            end

            psi_c = Array{Float64}(undef, d, d)
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

        empty_cluster = Cluster(size(mu, 1))

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
        
        log_numerator = d/2 * log(2pi) + nu * d/2 * log(2) + logmvgamma(d, nu/2)
    
        log_denominator = d/2 * log(lambda) + nu/2 * log(det(psi))

        return log_numerator - log_denominator
    
    end

    # Return the log-likelihood of the model
    function log_Pgenerative(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; hyperpriors=true)
    
        @assert all(length(c) > 0 for c in clusters)
        
        N = sum([length(c) for c in clusters])
        K = length(clusters)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)
    
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

    function log_cluster_weight(element::Vector{Float64}, cluster::Cluster, alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)

        @assert !(element in cluster)
        
        
        # d = size(mu, 1)

        if isempty(cluster)
            log_weight = log(alpha) # - log(alpha + Nminus1)
        else
            log_weight = log(length(cluster)) # - log(alpha + Nminus1)
        end
                    
        push!(cluster, element)
        log_weight += log_Zniw(cluster, mu, lambda, psi, nu) # - log_Zniw(nothing, mu, lambda, psi, nu)
        pop!(cluster, element)
        log_weight -= log_Zniw(cluster, mu, lambda, psi, nu) # - log_Zniw(nothing, mu, lambda, psi, nu)
        # log_weight -= d/2 * log(2pi)
        
        return log_weight

    end 
    

    function advance_gibbs!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; temperature=1.0)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)

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
        d = size(hyperparams.mu, 1)
    
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
        step_type=:gaussian, step_scale=5.5)
            
        # No Cauchy because it's a very bad idea on a log scale
        if step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_scale/2, step_scale/2)
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
                         random_order=true, step_scale=0.25, step_type=:gaussian)
    
        if step_type == :cauchy
            step_distrib = Cauchy(0.0, step_scale)
        elseif step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
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
                             step_type=:gaussian, step_scale=1.5)

        if step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_scale/2, step_scale/2)
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
                          random_order=true, step_scale=0.1, step_type=:gaussian)
    
        if step_type == :cauchy
            step_distrib = Cauchy(0.0, step_scale)
        elseif step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
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
                         step_type=:gaussian, step_scale=1.0)

        if step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end
    
        d = size(hyperparams.mu, 1)
        
        mu, lambda, psi, nu = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

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

    function initiate_chain(data::Vector{Vector{Float64}}; standardize=false, strategy=:hot)

        @assert all(size(e, 1) == size(first(data), 1) for e in data)

        N = length(data)

        d = size(first(data), 1)

        hyperparams = MNCRPhyperparams(d)

        chain = MNCRPchain([], hyperparams, zeros(d), zeros(d), [], [], [], [], [], [], hyperparams, -Inf, 1)

        chain.hyperparams_chain = [hyperparams]
        
        if standardize
            offset = mean(dataset)
            scale = std(dataset)
            data = map(x -> x .- offset, dataset)
            data = map(x -> x ./ scale, dataset)
            chain.data_offset = offset
            chain.data_scale = scalee
        end

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

    
    function advance_chain!(chain::MNCRPchain; nb_steps=100,
        nb_splitmerge=0, splitmerge_t=2,
        nb_gibbs=1, gibbs_temp=1.0,
        nb_hyperparamsmh=10,
        observation_points=nothing, observe_every=10,
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

        progbar = Progress(nb_steps; showspeed=true)

        for step in 1:nb_steps



            # Metropolis-Hastings moves over each parameter
            # step_scale is adjusted to roughly hit
            # an #accepted:#rejected ratio of 1
            # It was tuned with standardized data
            # namelyby subtracting the mean and dividing
            # by the standard deviation along each dimension.
            for i in 1:nb_hyperparamsmh
                advance_alpha!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_scale=0.5)
                
                advance_mu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_scale=0.3)
                
                advance_lambda!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_scale=0.5)

                advance_psi!(chain.clusters, chain.hyperparams,
                            step_type=:gaussian, step_scale=0.1)

                advance_nu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_scale=0.3)
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
                    map_success = attempt_map!(chain, max_nb_pushes=10)
                    nb_map_successes += map_success
                    last_map_idx = chain.map_idx
            end


            if observation_points !== nothing && mod(step, observe_every) == 0

                observations = Float64[presence_probability(x, chain.clusters, chain.hyperparams, minsize=0.05) for x in observation_points]
                # observations = Float64[average_tail_probability(x, chain.clusters, chain.hyperparams) for x in observation_points]
                push!(chain.observation_chain, observations)

            end


            if checkpoint_every > 0 && 
                (mod(step, checkpoint_every) == 0
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

    function burn!(chain::MNCRPchain, nb_burn::Int64)

        if nb_burn >= length(chain.logprob_chain)
            @error("Can't burn the whole chain, nb_burn must be smaller than $(length(chain.logprob_chain))")
        end

        if nb_burn <= 0
            @error("nb_burn must be at least 1")
        end

        chain.logprob_chain = chain.logprob_chain[nb_burn+1:end]
        chain.hyperparams_chain = chain.hyperparams_chain[nb_burn+1:end]
        chain.nbclusters_chain = chain.nbclusters_chain[nb_burn+1:end]

        if chain.map_idx <= nb_burn        
            chain.map_clusters = deepcopy(chain.clusters)
            chain.map_hyperparams = deepcopy(chain.hyperparams)
            chain.map_logprob = log_Pgenerative(chain.clusters, chain.hyperparams)
            chain.map_idx = length(chain.logprob_chain)
        else
            chain.map_idx -= nb_burn
        end
        
        return chain

    end

    function attempt_map!(chain::MNCRPchain; max_nb_pushes=15, optimize_hyperparams=true)
            
        map_clusters_attempt = copy(chain.clusters)
        map_mll = log_Pgenerative(map_clusters_attempt, chain.hyperparams)
        
        hyperparams = deepcopy(chain.hyperparams)
        
        if optimize_hyperparams
            optimize_hyperparams!(map_clusters_attempt, hyperparams)
        end

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
    
    function plot(chain::MNCRPchain; dims::Vector{Int64}=[1, 2], burn=0)
        
        @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

        d = size(chain.hyperparams.mu, 1)
        
        proj = dims_to_proj(dims, d)

        return plot(chain, proj, burn=burn)
    end

    function plot(chain::MNCRPchain, proj::Matrix{Float64}; burn=0)
        @assert size(proj, 1) == 2 "For plotting the projection matrix should have 2 rows"
        map_marginals = project_clusters(chain.map_clusters, proj)
        current_marginals = project_clusters(chain.clusters, proj)
        
        p_map = plot(map_marginals, title="MAP state ($(length(chain.map_clusters)) clusters)")
        p_current = plot(current_marginals, title="Current state ($(length(chain.clusters)) clusters)", legend=false)

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
        d = size(hyperparams.mu, 1)
        sigma_mode = Symmetric(psi_c / (nu_c + d + 1))
        return eigen(sigma_mode)
    end

    function covellipses!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; dims::Vector{Int64}=[1, 2], n_std=2, scalematrix=nothing, offset=nothing, mode=false, lowest_weight=nothing, plot_kw...)

        mu0, lambda0, psi0, nu0 = hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu0, 1)

        clusters = proj_clusters(clusters, dims)

        for c in clusters
            if lowest_weight === nothing || length(c) >= lowest_weight
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

    function local_average_covprec(coordinate::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)

        coordinate_set = Cluster([coordinate])

        N = sum(length(c) for c in clusters)

        log_parts = []
        cluster_covs = []
        cluster_precs = []


        clusters = vcat(clusters, [Cluster(d)])

        for c in clusters

            c_union_coordinate = union(coordinate_set, c)

            log_crp_weight = (length(c) > 0 ? log(length(c)) - log(alpha + N) : log(alpha)) - log(alpha + N)

            log_data_weight = (log_Zniw(c_union_coordinate, mu, lambda, psi, nu) 
                               - log_Zniw(c, mu, lambda, psi, nu) 
                               - d/2 * log(2pi))

            _, _, psi_c, nu_c = updated_niw_hyperparams(c_union_coordinate, mu, lambda, psi, nu)
            
            push!(log_parts, log_crp_weight + log_data_weight)

            push!(cluster_covs, psi_c / (nu_c - d - 1))
            push!(cluster_precs, nu_c * inv(psi_c))
        end

        log_parts = log_parts .- logsumexp(log_parts)
        
        average_covariance = sum(cluster_covs .* exp.(log_parts))
        average_precision = sum(cluster_precs .* exp.(log_parts))

        return (average_covariance=average_covariance, 
                average_precision=average_precision)
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

    function presence_probability(x::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; minsize=0)
        

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        
        d = size(mu, 1)
        N = sum([length(cl) for cl in clusters])

        @assert all(!isempty, clusters)

        push!(clusters, Cluster(d))

        if typeof(minsize) == Float64
            sorted_sizes = sort(length.(clusters))
            minsize = sorted_sizes[findfirst(x -> x[2] >= minsize * N, 
                                            collect(zip(sorted_sizes, cumsum(sorted_sizes))))
                                        ]
        end
        
        minsize = max(minsize, 1)

        under_tresh_mask = [length(cl) < minsize for cl in clusters]

        log_assignments = zeros(length(clusters))
        for (i, cluster) in enumerate(clusters)
            log_assignments[i] = log_cluster_weight(x, cluster, alpha, mu, lambda, psi, nu)
        end

        # remove empty cluster
        pop!(clusters)

        log_assignments .-= logsumexp(log_assignments)
        
        return 1.0 - exp(logsumexp(log_assignments[under_tresh_mask]))

    end

    # Only valid in 2-D I believe
    # function average_tail_probability(x::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        

    #     alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        
    #     d = size(mu, 1)
    #     N = sum([length(cl) for cl in clusters])

    #     @assert all(!isempty, clusters)

    #     push!(clusters, Cluster(d))

    #     log_weights = zeros(length(clusters))
    #     tail_probabilities = zeros(length(clusters))
    #     for (i, cluster) in enumerate(clusters)
    #         log_weights[i] = log_cluster_weight(x, cluster, alpha, mu, lambda, psi, nu)

    #         push!(cluster, x)
    #         mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
    #         tail_probabilities[i] = exp(-1/2 * (x - mu_c)' * inv(psi_c / (nu_c + d + 1)) * (x - mu_c))
    #         pop!(cluster, x)

    #     end

    #     pop!(clusters)

    #     log_weights .-= logsumexp(log_weights)

    #     component_probabilities = exp.(log_weights)

    #     average_tail_probability = sum(tail_probabilities .* component_probabilities)

    #     return average_tail_probability

    # end

    function optimize_hyperparams(clusters::Vector{Cluster}, hyperparams0::MNCRPhyperparams)

        objfun(x) = -log_Pgenerative(clusters, opt_pack(x))

        optres = optimize(objfun, opt_unpack(hyperparams0), LBFGS())

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
end
