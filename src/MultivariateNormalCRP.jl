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
    # using Serialization: serialize
    using JLD2
    using ProgressMeter

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, empty!, iterate, deepcopy, copy, sort, in

    export Cluster
    export initiate_chain, advance_chain!, attempt_map!, burn!
    export clear_diagnostics!
    export log_Pgenerative, drawNIW, stats
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export plot, covellipses!, project_clusters, project_hyperparams
    export local_average_covprec, crp_distance, crp_distance_matrix, presence_probability
    export wasserstein2_distance, wasserstein1_distance_bound

    include("types/diagnostics.jl")
    include("types/hyperparams.jl")
    include("types/cluster.jl")
    include("types/chain.jl")

    function dims_to_proj(dims::Vector{Int64}, d::Int64)
        return diagm(ones(d))[dims, :]
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


    # Split-Merge Metropolis-Hastings with restricted Gibbs sampling from Jain & Neal 2004
    # (nothing wrong with a little ravioli code, 
    #  might refactor later, might not)
    function advance_JNsplitmerge!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; t=3, proposal_temperature=1.0)
            
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

        d = size(mu, 1)
    
        # just to make explicit what's going on
        elements = Tuple{Int64, Vector{Float64}}[(ce, e) for (ce, c) in enumerate(clusters) for e in c]

        (ci, ei), (cj, ej) = sample(elements, 2, replace=false)
    
        if ci == cj

            Sminusij = Vector{Float64}[e for e in clusters[ci] 
                                         if !(e === ei) && !(e === ej)]

            initial_S = clusters[ci]
            deleteat!(clusters, ci)

        elseif ci != cj

            Sminusij = collect(flatten((
                Vector{Float64}[e for e in clusters[ci] if !(e == ei)], 
                Vector{Float64}[e for e in clusters[cj] if !(e == ej)]
                )))

            initial_Si = clusters[ci]
            initial_Sj = clusters[cj]
            deleteat!(clusters, sort([ci, cj]))

        end
    
        launch_Si = Cluster([ei])
        launch_Sj = Cluster([ej])

        # Perform t restricted Gibbs sweeps.
        # The first sweep is sequential
        # i.e. we grow launch_Si/launch_Sj
        # until all elements in Sminusij
        # are either in launch_Si or
        # launch_Sj

        for n in 1:t
        
            # for e in flatten((Sminusij, [ei, ej]))
            for e in shuffle!(Sminusij)
                
                # It's in one of them.
                # Make sure it's in neither.
                delete!(launch_Si, e)
                delete!(launch_Sj, e)
  
                log_weights = zeros(2)
                for (i, launch_cluster) in enumerate(Cluster[launch_Si, launch_Sj])
                    # log_weights[i] = log_cluster_weight(e, launch_cluster, hyperparams)                 
                    log_weights[i] = log_cluster_weight(e, launch_cluster, alpha, mu, lambda, psi, nu)
                end

                ######## Still in restricted gibbs ########

                unnorm_logp = log_weights #/ proposal_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))
                new_assignment = sample(Cluster[launch_Si, launch_Sj], probs)
                push!(new_assignment, e)
            end
        end
    
        ##############
        ### Split! ###
        ##############
        if ci == cj
            # Perform last restricted Gibbs scan to form
            # proposed split state and keep track of assignment
            # probabilities to form the Hastings ratio

            log_q_proposed_given_launch = 0.0

            # We rename launch_Si and launch_Sj just
            # to make it explicit that the last t+1 restricted
            # Gibbs scan produces the proposed split state
            proposed_Si = launch_Si
            proposed_Sj = launch_Sj
        
            for e in shuffle!(Sminusij)
            # for e in flatten((Sminusij, [ei, ej]))

                delete!(proposed_Si, e)
                delete!(proposed_Sj, e)

                #############

                log_weights = zeros(2)
                for (i, proposed_cluster) in enumerate(Cluster[proposed_Si, proposed_Sj])
                    log_weights[i] = log_cluster_weight(e, proposed_cluster, alpha, mu, lambda, psi, nu)
                end

                ######## Still in split! ########

                unnorm_logp = log_weights / proposal_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp) 
                probs = Weights(exp.(norm_logp))

                new_assignment, log_qe = sample(Tuple{Cluster, Float64}[
                                                 (proposed_Si, norm_logp[1]), 
                                                 (proposed_Sj, norm_logp[2])], 
                                                probs)
                
                push!(new_assignment, e)

                log_q_proposed_given_launch += log_qe

            end

            # if ei in proposed_Sj
                # proposed_Si, proposed_Sj = proposed_Sj, proposed_Si
            # end
            # delete!(proposed_Si, ej)
            # push!(proposed_Sj, ej)
            
            # The Hastings ratio is 
            # q(initial | proposed) / q(proposed | initial)
            # Here q(initial | proposed) = 1 
            # because the initial state is a merge state with its ci = cj
            # and the proposed state is a split state, so the proposal process
            # would be a merge proposal and there is only one way to do so.
            
            # Note also that q(proposed | initial) = q(proposed | launch)
            # because every initial states gets shuffled and forgotten
            # when forming a random launch state by restricted Gibbs moves

            n_proposed_Si = length(proposed_Si)
            n_proposed_Sj = length(proposed_Sj)
            n_initial_S = length(initial_S)        
        
            log_crp_ratio = log(alpha) + loggamma(n_proposed_Si) + loggamma(n_proposed_Sj) - loggamma(n_initial_S)

            log_likelihood_ratio = log_Zniw(proposed_Si, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Si * d/2 * log(2pi)        
            log_likelihood_ratio += log_Zniw(proposed_Sj, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Sj * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_S, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_S * d/2 * log(2pi)
            
            ######## Still in split! ########

            log_acceptance = -log_q_proposed_given_launch + log_crp_ratio + log_likelihood_ratio
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                push!(clusters, proposed_Si, proposed_Sj)
                hyperparams.diagnostics.accepted_split += 1
            else
                push!(clusters, initial_S)
                hyperparams.diagnostics.rejected_split += 1
            end
        
            
            @assert all([!isempty(c) for c in clusters])

            return clusters

        ##############
        ### Merge! ###
        ##############
        elseif ci != cj
            
            # We are calculating
            # q(initial | merge) = q(initial | launch)
            # and we remember that 
            # q(merge | initial) = q(merge | launch) = 1

            # We do that by performing an hypothetical
            # restricted gibbs sweep that transforms
            # the current launch state into the
            # initial state.

            log_q_initial_given_launch = 0.0

            for e in Sminusij

                delete!(launch_Si, e)
                delete!(launch_Sj, e)

                #############

                log_weights = zeros(2)
                for (i, cluster) in enumerate(Cluster[launch_Si, launch_Sj])
                    log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
                end

                ######## Still in merge! ########

                unnorm_logp = log_weights / proposal_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

                ### We are progressively forcing the transformation
                ### of the launch state into the initial state
                ### and accumulating the transition probabilities

                if e in initial_Si
                    push!(launch_Si, e)
                    log_q_initial_given_launch += norm_logp[1]
                elseif e in initial_Sj
                    push!(launch_Sj, e)
                    log_q_initial_given_launch += norm_logp[2]
                end
            end

            ######## Still in merge! ########
            
            proposed_S = union(launch_Si, launch_Sj)

            n_proposed_S = length(proposed_S)
            n_initial_Si = length(initial_Si)
            n_initial_Sj = length(initial_Sj)
        
            log_crp_ratio = -log(alpha) + loggamma(n_proposed_S) - loggamma(n_initial_Si) - loggamma(n_initial_Sj)
        
            log_likelihood_ratio = log_Zniw(proposed_S, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_S * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_Si, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_Si * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_Sj, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_Sj * d/2 * log(2pi)
        
            log_acceptance = log_q_initial_given_launch + log_crp_ratio + log_likelihood_ratio
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                push!(clusters, proposed_S)
                hyperparams.diagnostics.accepted_merge += 1
            else
                push!(clusters, initial_Si, initial_Sj)
                hyperparams.diagnostics.rejected_merge += 1
            end

            @assert all([!isempty(c) for c in clusters])
            
            return clusters

        end
    
    end


    function hypothetical_2restricted_logtransition(initial_state::Vector{Cluster}, final_state::Vector{Cluster}, hyperparams::MNCRPhyperparams; scheduled_elements=nothing, proposal_temperature=1.0)

        @assert length(initial_state) == length(final_state) == 2

        initial_state = copy(initial_state)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

        if scheduled_elements === nothing
            scheduled_elements = Vector{Float64}[e for cluster in initial_state for e in cluster]
        end

        log_q_final_given_initial = 0.0

        for e in shuffle!(scheduled_elements)

            pop!(initial_state, e; delete_empty=false)

            log_weights = zeros(2)

            for (i, cluster) in enumerate(initial_state)
                log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
            end

            unnorm_logp = log_weights / proposal_temperature
            norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

            if e in final_state[1]
                log_q_final_given_initial += norm_logp[1]
                push!(initial_state[1], e)
            elseif e in final_state[2]
                log_q_final_given_initial += norm_logp[2]
                push!(initial_state[2], e)
            else
                @error "Something went wrong"
            end

        end   

        return log_q_final_given_initial

    end

    # Wish it were faster but it is much superior to
    # Jain & Neal's split-merge move.
    function advance_splitmerge_seq!(clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; t=3, proposal_temperature=1.0)

        @assert t >= 1
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(hyperparams.mu, 1)
    
        cluster_indices = Int64[ce for (ce, cluster) in enumerate(clusters) for e in cluster]

        ci, cj = sample(cluster_indices, 2, replace=false)
    
        if ci == cj

            scheduled_elements = collect(clusters[ci])

            initial_state = Cluster[clusters[ci]]
            deleteat!(clusters, ci)

        elseif ci != cj

            scheduled_elements = collect(flatten((clusters[ci], clusters[cj])))

            initial_state = Cluster[clusters[ci], clusters[cj]]
            deleteat!(clusters, sort([ci, cj]))

        end

        shuffle!(scheduled_elements)
    
        launch_state = Cluster[]
        proposed_state = Cluster[]

        log_q_proposed_given_launch = 0.0

        for step in flatten((1:t, [:create_proposed_state]))
        
            # The launch state is the one before
            # the last restricted Gibbs sweep.
            # Save if for next for when we calculate
            # the q(launch|proposed) transition probability
            if step == :create_proposed_state
                launch_state = copy(proposed_state)                
            end

            log_q_proposed_given_launch = 0.0

            for e in shuffle!(scheduled_elements)

                # Elements are not in the
                # state during the first
                # sequential Gibbs sweep step=1
                # so we don't use pop!
                # We keep empty clusters because
                # the left/right ordering, including
                # of empty clusters, is important when
                # calculating q(launch|proposed)
                delete!(proposed_state, e; delete_empty=false)

                if length(proposed_state) < 2
                    push!(proposed_state, Cluster(d))
                end

                @assert sum(isempty.(proposed_state)) <= 1

                #############        
            
                log_weights = zeros(length(proposed_state))
                for (i, cluster) in enumerate(proposed_state)
                    log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
                end

                unnorm_logp = log_weights / (step == :create_proposed_state ? proposal_temperature : 1.0)
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))
                new_assignment, log_transition = sample(collect(zip(proposed_state, norm_logp)), probs)
                
                push!(new_assignment, e)

                log_q_proposed_given_launch += log_transition
                            
            end

        end

        @assert length(proposed_state) == 2
        @assert length(launch_state) == 2

        log_q_proposed_given_launch += hypothetical_2restricted_logtransition(
                                        launch_state, 
                                        Cluster[proposed_state[2], proposed_state[1]],
                                        hyperparams; 
                                        scheduled_elements=shuffle!(scheduled_elements))


        # launch_state_2 = Cluster[]

        # for step in 1:t

        #     for e in shuffle!(scheduled_elements)

        #         delete!(launch_state_2, e; delete_empty=false)

        #         if length(launch_state_2) < 2
        #             push!(launch_state_2, Cluster(d))
        #         end

        #         @assert sum(isempty.(launch_state_2)) <= 1

        #         #############        
            
        #         log_weights = zeros(length(launch_state_2))
        #         for (i, cluster) in enumerate(launch_state_2)
        #             log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
        #         end

        #         unnorm_logp = log_weights
        #         norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
        #         probs = Weights(exp.(norm_logp))
        #         new_assignment = sample(launch_state_2, probs)
                
        #         push!(new_assignment, e)
                            
        #     end

        # end

        # @assert length(launch_state_2) == 2        

        # The left/right ordering of clusters
        # in both the launch state and proposed
        # state is important!

        # ...therefore we only filter the proposed
        # state after copying it. This copy will
        # be used next to calculate the proposed
        # to launch transition probability.
        # proposed_state_copy = copy(proposed_state)
        # filter!(!isempty, proposed_state)
        # The proposed state is left untouched
        # from here as it will be the one that
        # will make it back into the global state
        # if it is accepted and we always impose
        # the the global state must not contain
        # any empty cluster.

        # log_q_launch_given_proposed = 0.0

        # for e in shuffle!(scheduled_elements)

        #     pop!(proposed_state_copy, e; delete_empty=false)

        #     log_weights = zeros(2)

        #     for (i, cluster) in enumerate(proposed_state_copy)
        #         log_weights[i] = log_cluster_weight(e, cluster, alpha, mu, lambda, psi, nu)
        #     end

        #     unnorm_logp = log_weights / proposal_temperature
        #     norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

        #     if e in launch_state[1]
        #         log_q_launch_given_proposed += norm_logp[1]
        #         push!(proposed_state_copy[1], e)
        #     elseif e in launch_state[2]
        #         log_q_launch_given_proposed += norm_logp[2]
        #         push!(proposed_state_copy[2], e)
        #     else
        #         @error "Something went wrong"
        #     end

        # end

        log_q_launch_given_proposed = hypothetical_2restricted_logtransition(
                                        proposed_state, 
                                        launch_state,
                                        # launch_state_2,
                                        hyperparams;
                                        scheduled_elements=shuffle!(scheduled_elements), proposal_temperature=proposal_temperature)
        
        log_q_launch_given_proposed += hypothetical_2restricted_logtransition(
                                        proposed_state, 
                                        Cluster[launch_state[2], 
                                        launch_state[1]], 
                                        # Cluster[launch_state_2[2], 
                                        # launch_state_2[1]], 
                                        hyperparams; 
                                        scheduled_elements=shuffle!(scheduled_elements), proposal_temperature=proposal_temperature)
        
        filter!(!isempty, proposed_state)
        # filter!(!isempty, launch_state)

        log_acceptance = (log_Pgenerative(proposed_state, hyperparams, hyperpriors=false) 
                        - log_Pgenerative(initial_state, hyperparams, hyperpriors=false))

        log_acceptance += log_q_launch_given_proposed - log_q_proposed_given_launch

        log_acceptance = min(0.0, log_acceptance)
        
        if log(rand()) < log_acceptance
            append!(clusters, proposed_state)
            hyperparams.diagnostics.accepted_fullseq += 1
        else
            append!(clusters, initial_state)
            hyperparams.diagnostics.rejected_fullseq += 1
        end

        @assert all([!isempty(c) for c in clusters])

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

    function jeffreys_alpha(alpha::Float64, n::Int64)

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

    end

    # Cute result but unfortunately lead to 
    # improper a-posteriori probability in nu and psi
    function jeffreys_nu(nu::Float64, d::Int64)

        # return sqrt(1/4 * sum(polygamma.(1, nu/2 .+ 1/2 .- 1/2 * (1:d))))
        return sqrt(1/4 * sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

    end

    function initiate_chain(filename::AbstractString)
        # Expects a JLD2 file
        return load(filename)["chain"]
    end

    function initiate_chain(data::Vector{Vector{Float64}}; strategy=:hot)

        @assert all(size(e, 1) == size(first(data), 1) for e in data)

        N = length(data)

        d = size(first(data), 1)

        hyperparams = MNCRPhyperparams(d)

        chain = MNCRPchain([], hyperparams, [], [], [], [], [], hyperparams, -Inf, 1)

        chain.hyperparams_chain = [hyperparams]
        
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
        nb_splitmerge=0, splitmerge_t=2, sm_prop_temp=1.0,
        nb_gibbs=1, gibbs_temp=1.0,
        nb_hyperparamsmh=10,
        nb_fullseq=0, fullseq_t=2, fullseq_prop_temp=1.0,
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

        last_accepted_fullseq = hp.diagnostics.accepted_fullseq
        last_rejected_fullseq = hp.diagnostics.rejected_fullseq
        fullseq_total = 0
        
        # nb_fullseq_moves = 0
        
        nb_map_attemps = 0
        nb_map_successes = 0

        last_checkpoint = -1
        last_map_idx = chain.map_idx

        ###########################

        progbar = Progress(nb_steps; showspeed=true)

        for step in 1:nb_steps

            ## Large moves ##

            # Split-merge
            for i in 1:nb_splitmerge
                advance_JNsplitmerge!(chain.clusters, chain.hyperparams, 
                t=splitmerge_t, 
                proposal_temperature=sm_prop_temp)
            end            
            
            #################

            # Gibbs sweep
            for i in 1:nb_gibbs
                advance_gibbs!(chain.clusters, chain.hyperparams, temperature=gibbs_temp)
            end

            for i in 1:nb_fullseq
                advance_splitmerge_seq!(chain.clusters, chain.hyperparams, 
                t=fullseq_t,
                proposal_temperature=fullseq_prop_temp)              
            end


            push!(chain.nbclusters_chain, length(chain.clusters))

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

            fullseq_ratio = "$(hp.diagnostics.accepted_fullseq - last_accepted_fullseq)/$(hp.diagnostics.rejected_fullseq - last_rejected_fullseq)"
            fullseq_total += hp.diagnostics.accepted_fullseq - last_accepted_fullseq
            fullseq_per_step = round(fullseq_total/step, digits=2)
            last_accepted_fullseq = hp.diagnostics.accepted_fullseq
            last_rejected_fullseq = hp.diagnostics.rejected_fullseq

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
                (:"fullseq ratio, fullseq/step, T", fullseq_ratio * ", $(fullseq_per_step), T=$(round(fullseq_prop_temp, digits=1))"),
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
                print("   f:$(nb_fullseq_moves)")
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

    function attempt_map!(chain::MNCRPchain; max_nb_pushes=15)
            
        map_clusters_attempt = copy(chain.clusters)
        map_mll = log_Pgenerative(map_clusters_attempt, chain.hyperparams)
        
        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = copy(map_clusters_attempt)
            advance_gibbs!(test_attempt, chain.hyperparams; temperature=0.0)
            test_mll = log_Pgenerative(test_attempt, chain.hyperparams)
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

        attempt_logprob = log_Pgenerative(map_clusters_attempt, chain.hyperparams)
        if attempt_logprob > chain.map_logprob
            chain.map_logprob = attempt_logprob
            chain.map_clusters = map_clusters_attempt
            chain.map_hyperparams = deepcopy(chain.hyperparams)
            chain.map_idx = lastindex(chain.logprob_chain)
            return true
        else
            return false
        end
    end

    function plot(clusters::Vector{Cluster}; rev=false, plot_kw...)
        
        p = plot(
            legend_position=:topleft, grid=:no, 
            showaxis=:no, ticks=:true;
            plot_kw...)

        clusters = sort(clusters, by=length, rev=rev)
        for (cluster, color) in zip(clusters, cycle(tableau_20))
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

    function plot(chain::MNCRPchain; dims::Tuple{Int64, Int64}=(1, 2), burn=0)
        d = size(chain.hyperparams.mu, 1)
        
        proj = dims_to_proj([dims[1], dims[2]], d)

        return plot(chain, proj, burn=burn)
    end

    function eigen_mode(cluster::Cluster, hyperparams::MNCRPhyperparams)
        _, _, psi_c, nu_c = updated_niw_hyperparams(cluster, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu)
        d = size(hyperparams.mu, 1)
        sigma_mode = Symmetric(psi_c / (nu_c + d + 1))
        return eigen(sigma_mode)
    end

    function plot(chain::MNCRPchain, cluster::Cluster, hyperparams::MNCRPhyperparams; eigdirs::Tuple{Int64, Int64}=(1, 2), burn=0)

        _, evecs = eigen_mode(cluster, hyperparams)

        # proj = (evecs[:, end + 1 - eigdirs[1]], evecs[:, end + 1 - eigdirs[2]])
        proj = Matrix{Float64}(evecs[:, [end + 1 - eigdirs[1], end + 1 - eigdirs[2]]]')

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

    function average_tail_probability(x::Vector{Float64}, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        
        d = size(mu, 1)
        N = sum([length(cl) for cl in clusters])

        @assert all(!isempty, clusters)

        push!(clusters, Cluster(d))

        log_weights = zeros(length(clusters))
        tail_probabilities = zeros(length(clusters))
        for (i, cluster) in enumerate(clusters)
            log_weights[i] = log_cluster_weight(x, cluster, alpha, mu, lambda, psi, nu)

            push!(cluster, x)
            mu_c, lambda_c, psi_c, nu_c = updated_niw_hyperparams(cluster, mu, lambda, psi, nu)
            tail_probabilities[i] = exp(-1/2 * (x - mu_c)' * inv(psi_c / (nu_c - d - 1)) * (x - mu_c))
            pop!(cluster, x)

        end

        pop!(clusters)

        log_weights .-= logsumexp(log_weights)

        component_probabilities = exp.(log_weights)

        average_tail_probability = sum(tail_probabilities .* component_probabilities)

        return average_tail_probability

    end


end
