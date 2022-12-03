module MultivariateNormalCRP
    using Distributions: MvNormal, InverseWishart, Normal, Cauchy, Uniform, Multinomial
    using Random: randperm, shuffle, seed!
    using StatsFuns: logsumexp, logmvgamma
    using StatsBase: sample, mean, Weights, std, percentile, quantile
    using LinearAlgebra: det, LowerTriangular, Symmetric, cholesky, diag, tr, diagm, inv, norm, eigen, svd
    using SpecialFunctions: loggamma, polygamma
    using Base.Iterators: cycle
    using ColorSchemes: Paired_12, tableau_20
    using Plots: plot, vline!, hline!, scatter!, @layout, grid, scalefontsizes, mm
    using StatsPlots: covellipse!
    # using Serialization: serialize
    using JLD2
    using ProgressMeter

    import RecipesBase: plot
    import Base: pop!, push!, length, isempty, union, delete!, iterate, deepcopy, sort, in

    export initiate_chain, advance_chain!, attempt_map!, burn!
    export log_Pgenerative, drawNIW, stats
    export alpha_chain, mu_chain, lambda_chain, psi_chain, nu_chain, flatL_chain
    export plot, covellipses!, project_clusters, project_hyperparams
    export local_average_covprec, crp_distance, crp_distance_matrix, presence_probability
    export wasserstein2_distance, wasserstein1_distance_bound


    mutable struct Cluster
        elements::Set{Vector{Float64}}
        sum_x::Vector{Float64}
        sum_xx::Matrix{Float64}
    end

    function Cluster(d::Int64)
        return Cluster(Set{Vector{Float64}}(), zeros(Float64, d), zeros(Float64, d, d))
    end

    function Cluster(elements::Vector{Vector{Float64}})
        
        # We need at least an element
        # to know the dimension so that
        # we can initialize sum_x and sum_xx
        @assert length(elements) > 0 
        
        # d = size(first(elements), 1)
        # @assert all([size(element, 1) == d for element in elements])
        
        # sum_x = sum(elements)
        # sum_xx = sum([x * x' for x in elements])

        d = size(first(elements), 1)
        sum_x = zeros(Float64, d)
        @inbounds for i in 1:d
            for x in elements
                sum_x[i] += x[i]
            end
        end
        sum_xx = zeros(Float64, d, d)
        @inbounds for j in 1:d
            @inbounds for i in 1:j
                for x in elements
                    sum_xx[i, j] += x[i] * x[j]
                end
                sum_xx[j, i] = sum_xx[i, j]
            end
        end
        elements = Set{Vector{Float64}}(elements)    
        return Cluster(elements, sum_x, sum_xx)
    end

    function pop!(cluster::Cluster, x::Vector{Float64})
        el = pop!(cluster.elements, x)
        # cluster.sum_xx -= x * x'
        # cluster.sum_x -= x
        d = size(x, 1)
        @inbounds for i in 1:d
            cluster.sum_x[i] -= x[i]
        end
        @inbounds for j in 1:d
            @inbounds for i in 1:j
                cluster.sum_xx[i, j] -= x[i] * x[j]
                cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
            end
        end
        return el
    end

    function pop!(cluster::Cluster)
        x = pop!(cluster.elements)
        # cluster.sum_xx -= x * x'
        # cluster.sum_x -= x
        d = size(x, 1)
        @inbounds for i in 1:d
            cluster.sum_x[i] -= x[i]
        end
        @inbounds for j in 1:d
            @inbounds for i in 1:j
                cluster.sum_xx[i, j] -= x[i] * x[j]
                cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
            end
        end
        return x
    end

    function pop!(list_of_clusters::Vector{Cluster}, x::Vector{Float64})
        for (ci, cluster) in enumerate(list_of_clusters)
            if x in cluster
                rel = pop!(cluster, x)
                if isempty(cluster)
                    deleteat!(list_of_clusters, ci)
                end
                return rel
            end
        end
        error(KeyError, ": key $x not found")
    end

    function push!(cluster::Cluster, x::Vector{Float64})
        if !(x in cluster.elements)
            # cluster.sum_xx += x * x'
            # cluster.sum_x += x
            d = size(x, 1)
            @inbounds for i in 1:d
                cluster.sum_x[i] += x[i]
            end
            @inbounds for j in 1:d
                @inbounds for i in 1:j
                    cluster.sum_xx[i, j] += x[i] * x[j]
                    cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
                end
            end
        end
        push!(cluster.elements, x)
        return cluster
    end

    function delete!(cluster::Cluster, x::Vector{Float64})
        if x in cluster.elements
            pop!(cluster, x)
        end
        return cluster
    end

    function union(cluster1::Cluster, cluster2::Cluster)
        @assert size(cluster1.sum_x, 1) == size(cluster2.sum_x, 1)
        d = size(cluster1.sum_x, 1)
        new_elements = union(cluster1.elements, cluster2.elements)

        # we don't know if they have elements in common
        # so we need to recompute sum_x and sum_xx
        # maybe we'll accelerate this by
        # tracking the intersection

        # new_sum_xx = sum([x * x' for x in elements])
        # new_sum_x = sum(elements)

        new_sum_x = zeros(Float64, d)
        for i in 1:d
            for x in new_elements
                new_sum_x[i] += x[i]
            end
        end
        new_sum_xx = zeros(Float64, d, d)
        for j in 1:d
            for i in 1:j
                for x in new_elements
                    new_sum_xx[i, j] += x[i] * x[j]
                end
                new_sum_xx[j, i] = new_sum_xx[i, j]
            end
        end
        
        return Cluster(new_elements, new_sum_x, new_sum_xx)
    end

    function isempty(cluster::Cluster)
        return isempty(cluster.elements)
    end

    function length(cluster::Cluster)
        return length(cluster.elements)
    end

    function in(element::Vector{Float64}, cluster::Cluster)
        return in(element, cluster.elements)
    end

    function find(element::Vector{Float64}, clusters::Vector{Cluster})
        for (i, cl) in enumerate(clusters)
            if in(element, cl)
                return (cl, i)
            end
        end
        error(KeyError, ": key $element not found")
    end

    function iterate(cluster::Cluster)
        return iterate(cluster.elements)
    end

    function iterate(cluster::Cluster, state)
        return iterate(cluster.elements, state)
    end

    function deepcopy(cluster::Cluster)
        return Cluster(deepcopy(cluster.elements), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx))
    end

    mutable struct MNCRPhyperparams
        alpha::Float64
        mu::Vector{Float64}
        lambda::Float64
        flatL::Vector{Float64}
        L::LowerTriangular{Float64, Matrix{Float64}}
        psi::Matrix{Float64}
        nu::Float64

        # idealy this should be wrapped in
        # a diagnostic struct

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

        accepted_fullseq::Int64
        rejected_fullseq::Int64

    end

    function MNCRPhyperparams(alpha, mu, lambda, flatL, L, psi, nu)
        d = size(mu, 1)
        sizeflatL = Int64(d * (d + 1) / 2)

        return MNCRPhyperparams(
        alpha, mu, lambda, flatL, L, psi, nu, 
        0, 0, # acc/rej alpha 
        zeros(Int64, d), zeros(Int64, d), #acc/rej mu
        0, 0, # acc/rej lambda
        zeros(Int64, sizeflatL), zeros(Int64, sizeflatL), # acc/rej psi
        0, 0, # acc/rej nu
        0, 0, # acc/rej split
        0, 0, # acc/rej merge
        0, 0) # acc/rej fullseq
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
        return MNCRPhyperparams(1.0, zeros(d), 1.0, LowerTriangular(diagm(fill(sqrt(0.1), d))), 1.0 * d)
    end

    function clear_diagnostics!(hyperparams::MNCRPhyperparams)
        d = size(hyperparams.mu, 1)
        sizeflatL = Int64(d * (d + 1) / 2)

        hyperparams.accepted_alpha = 0
        hyperparams.rejected_alpha = 0

        hyperparams.accepted_mu = zeros(Int64, d)
        hyperparams.rejected_mu = zeros(Int64, d)

        hyperparams.accepted_lambda = 0
        hyperparams.rejected_lambda = 0
        
        hyperparams.accepted_flatL = zeros(Int64, sizeflatL)
        hyperparams.rejected_flatL = zeros(Int64, sizeflatL)

        hyperparams.accepted_nu = 0
        hyperparams.rejected_nu = 0

        hyperparams.accepted_split = 0
        hyperparams.rejected_split = 0

        hyperparams.accepted_merge = 0
        hyperparams.rejected_merge = 0

        hyperparams.accepted_fullseq = 0
        hyperparams.rejected_fullseq = 0

        return hyperparams
        
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
    flatL_chain(chain::MNCRPchain) = [p.flatL for p in chain.hyperparams_chain]
    flatL_chain(chain::MNCRPchain, i) = [p.flatL[i] for p in chain.hyperparams_chain]
    nu_chain(chain::MNCRPchain) = [p.nu for p in chain.hyperparams_chain]
    
    function ij(flat_k::Int64)
        i = Int64(ceil(1/2 * (sqrt(1 + 8 * flat_k) - 1)))
        j = Int64(flat_k - i * (i - 1)/2)
        return (i, j)
    end

    function flatten(L::LowerTriangular{Float64})
        
        d = size(L, 1)
    
        flatL = zeros(Int64(d * (d + 1) / 2))
    
        # Fortran flashbacks
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
        hyperparams.flatL = deepcopy(value) # Just making sure?
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

    function project_cluster(cluster::Cluster, proj::Matrix{Float64})
        return Cluster([proj * x for x in cluster])
    end

    function project_clusters(clusters::Vector{Cluster}, proj::Matrix{Float64})
        return Cluster[project_cluster(cluster, proj) for cluster in clusters]
    end

    function project_clusters(clusters::Vector{Cluster}, dims::Vector{Int64})
        el = pop!(clusters[1])
        d = size(el, 1)
        push!(clusters[1], el)
        return project_clusters(clusters, dims_to_proj(dims, d))
    end

    function project_hyperparams(hyperparams::MNCRPhyperparams, proj::Matrix{Float64})
        proj_hp = deepcopy(hyperparams)
        proj_hp.mu = proj * proj_hp.mu
        proj_hp.psi = proj * proj_hp.psi * proj'
        return proj_hp
    end

    function project_hyperparams(hyperparams::MNCRPhyperparams, dims::Vector{Int64})
        d = size(hyperparams.mu, 1)
        return project_hyperparams(hyperparams, dims_to_proj(dims, d))
    end

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
    function log_Pgenerative(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; hyperpriors=true)
    
        @assert all(length(c) > 0 for c in list_of_clusters)
        
        N = sum(length(c) for c in list_of_clusters)
        K = length(list_of_clusters)

        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)
    
        # Log-probability associated with the Chinese Restaurant Process
        log_crp = K * log(alpha) - loggamma(alpha + N) + loggamma(alpha) + sum([loggamma(length(c)) for c in list_of_clusters])
        
        # Log-probability associated with the data likelihood
        # and Normal-Inverse-Wishart base distribution of the CRP
        log_niw = 0.0
        for cluster in list_of_clusters
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

    function add_to_state_gibbs!(element::Vector{Float64}, list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; temperature=1.0)


        # @assert all(!isempty(c) for c in list_of_clusters)
        filter!(c -> !isempty(c), list_of_clusters)

        # element is not in any cluster in list_of_clusters
        # so the total number of elements across clusters
        # is now Nminus1
        Nminus1 = sum([length(c) for c in list_of_clusters])
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)

        empty_cluster = Cluster(d)
        push!(list_of_clusters, empty_cluster)

        log_weights = Array{Float64}(undef, length(list_of_clusters))
        for (i, cluster) in enumerate(list_of_clusters)
            if isempty(cluster)
                log_weights[i] = log(alpha) - log(alpha + Nminus1)
            else
                log_weights[i] = log(length(cluster)) - log(alpha + Nminus1)
            end
            push!(cluster, element)
            log_weights[i] += log_Zniw(cluster, mu, lambda, psi, nu)
            pop!(cluster, element)
            log_weights[i] -= log_Zniw(cluster, mu, lambda, psi, nu)
            log_weights[i] -= d/2 * log(2pi)
            # We omit log(Z0) and d/2 log(2pi) because they cancel
        end

        unnorm_logp = log_weights / (temperature > 0.0 ? temperature : 1.0)
        norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

        if temperature > 0.0
            # Gibbs
            probs = Weights(exp.(norm_logp))
            selected_cluster = sample(list_of_clusters, probs)
        else
            # greedy Gibbs
            _, max_idx = findmax(norm_logp)
            selected_cluster = list_of_clusters[max_idx]
        end
        
        push!(selected_cluster, element)

        # Remove empty cluster(s) 
        filter!(c -> !isempty(c), list_of_clusters)

    end

    function advance_gibbs!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; temperature=1.0)
    
        elements = shuffle([el for cluster in list_of_clusters for el in cluster])
    
        for e in elements
            pop!(list_of_clusters, e)
            add_to_state_gibbs!(e, list_of_clusters, hyperparams; temperature=temperature)
        end
    
    end


    # Split-Merge Metropolis-Hastings with restricted Gibbs sampling from Jain & Neal 2004
    # (nothing wrong with a little ravioli code, 
    #  might refactor later, might not)
    function advance_clusters_JNrestrictedsplitmerge!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; t=3, restricted_temperature=1.0)
            
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu

        d = size(mu, 1)
    
        # just to make explicit what's going on
        elements = [(ce, e) for (ce, c) in enumerate(list_of_clusters) for e in c]

        (ci, ei), (cj, ej) = sample(elements, 2, replace=false)
    
        # S(minus)ij will only be used for sampling without replacement
        # and shouldn't change during execution
        Sminusij = union(list_of_clusters[ci], list_of_clusters[cj])
        pop!(Sminusij, ei)
        pop!(Sminusij, ej)
        Sminusij = collect(Sminusij)
    
        # For merge moves we need the initial state
        # for the "hypothetical" restricted gibbs state
        # that would bring the launch state into the initial state.
        # Namely we need it to form q(initial | merge) = q(initial | launch)
        if ci == cj

            # Keep initial state in case
            # the proposal is rejected
            initial_S = deepcopy(list_of_clusters[cj])
        
            launch_Si = Cluster([ei])

            # Avoid one more copy
            launch_Sj = list_of_clusters[cj]
            pop!(launch_Sj, ei)
        
            # Remove cluster from the active state
            # filter!(x -> !(x === launch_Sj), list_of_clusters)
            deleteat!(list_of_clusters, cj)

        elseif ci != cj

            # These will serve to calculate The
            # Hastings factor of the merge proposal
            initial_Si = deepcopy(list_of_clusters[ci])
            initial_Sj = deepcopy(list_of_clusters[cj])

            # Avoid two more copies
            launch_Si = list_of_clusters[ci]
            launch_Sj = list_of_clusters[cj]
        
            filter!(x -> !(x === launch_Si || x === launch_Sj), list_of_clusters)
        
        end
    

        # Shuffle elements (except ei and ej) with equal probability
        # of being reassigned to either launch cluster
        for e in Sminusij
            
            # Make sure it's in neither
            delete!(launch_Si, e)
            delete!(launch_Sj, e)

            assignment = sample([launch_Si, launch_Sj])
            push!(assignment, e)
        end
    
    
        # if t >= 1 perform t restricted Gibbs scans
        # where each element (except ei and ej) is reassigned
        # to either the cluster with ei or the cluster with ej
        for n in 1:t-1
        
            for e in sample(Sminusij, length(Sminusij), replace=false)
                
                # It's in one of them.
                # Make sure it's in neither.
                delete!(launch_Si, e)
                delete!(launch_Sj, e)
                        
                #############

                # CRP contribution
                log_weight_Si = log(length(launch_Si))

                push!(launch_Si, e)
                log_weight_Si += log_Zniw(launch_Si, mu, lambda, psi, nu)
                log_weight_Si -= log_Zniw(nothing, mu, lambda, psi, nu)
                log_weight_Si -= length(launch_Si) * d/2 * log(2pi)
                pop!(launch_Si, e)

                #############

                # CRP contribution
                log_weight_Sj = log(length(launch_Sj))

                push!(launch_Sj, e)
                log_weight_Sj += log_Zniw(launch_Sj, mu, lambda, psi, nu)
                log_weight_Sj -= log_Zniw(nothing, mu, lambda, psi, nu)
                log_weight_Sj -= length(launch_Sj) * d/2 * log(2pi)             
                pop!(launch_Sj, e)

                #############

                ######## Still in restricted gibbs ########

                unnorm_logp = [log_weight_Si, log_weight_Sj] / restricted_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))

                assignment = sample([launch_Si, launch_Sj], probs)
            
                push!(assignment, e)
            
            end

            # @assert !isempty(launch_Si) && !isempty(launch_Sj)
        end

        scheduled_elements = sample(Sminusij, length(Sminusij), replace=false)
    
        ##############
        ### Split! ###
        ##############
        if ci == cj
            # Perform last restricted Gibbs scan to form
            # proposed split state and keep track of assignment
            # probabilities to form the Hastings ratio

            log_q_proposed_given_launch = 0.0

            # We rename launch_Si and launch_Sj just
            # to make it explicit that the last t-1 restricted
            # Gibbs scan produces the proposed split state
            proposed_Si = launch_Si
            proposed_Sj = launch_Sj
        
            for e in scheduled_elements

                delete!(proposed_Si, e)
                delete!(proposed_Sj, e)

                #############

                # restricted CRP contribution
                log_weight_Si = log(length(proposed_Si))

                push!(proposed_Si, e)
                log_weight_Si += log_Zniw(proposed_Si, mu, lambda, psi, nu)
                log_weight_Si -= log_Zniw(nothing, mu, lambda, psi, nu)
                log_weight_Si -= length(proposed_Si) * d/2 * log(2pi)
                pop!(proposed_Si, e)
            
                #############

                # restricted CRP contribution
                log_weight_Sj = log(length(proposed_Sj))

                push!(proposed_Sj, e)
                log_weight_Sj += log_Zniw(proposed_Sj, mu, lambda, psi, nu) 
                log_weight_Sj -= log_Zniw(nothing, mu, lambda, psi, nu)
                log_weight_Sj -= length(proposed_Sj) * d/2 * log(2pi)
                pop!(proposed_Sj, e)

                #############

                ######## Still in split! ########

                unnorm_logp = [log_weight_Si, log_weight_Sj] / restricted_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
                probs = Weights(exp.(norm_logp))

                assignment = sample([proposed_Si, proposed_Sj], probs)
                push!(assignment, e)

                # Accumulate the log transition probability 
                # of the restricted assignment
                if assignment === proposed_Si
                    log_q_proposed_given_launch += norm_logp[1]
                elseif assignment === proposed_Sj
                    log_q_proposed_given_launch += norm_logp[2]
                else
                    @error("Something went wrong during split")
                end
            end
        

            n_proposed_Si = length(proposed_Si)
            n_proposed_Sj = length(proposed_Sj)
            n_initial_S = length(initial_S)        
        
            # The Hastings ratio is 
            # q(initial | proposed) / q(proposed | initial)
            # Here q(initial | proposed) = 1 
            # because the initial state is a merge state with its ci = cj
            # and the proposed state is a split state, so the proposal process
            # would be a merge proposal and there is only one way to do so.
        
            # Note also that q(proposed | initial) = q(proposed | launch)
            # because every initial states gets shuffled and forgotten
            # when forming a random launch state by restricted Gibbs moves
            log_hastings_ratio = -log_q_proposed_given_launch
        
            log_crp_ratio = log(alpha) + loggamma(n_proposed_Si) + loggamma(n_proposed_Sj) - loggamma(n_initial_S)

            log_likelihood_ratio = log_Zniw(proposed_Si, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Si * d/2 * log(2pi)        
            log_likelihood_ratio += log_Zniw(proposed_Sj, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_proposed_Sj * d/2 * log(2pi)
            log_likelihood_ratio -= log_Zniw(initial_S, mu, lambda, psi, nu) - log_Zniw(nothing, mu, lambda, psi, nu) - n_initial_S * d/2 * log(2pi)
            
            ######## Still in split! ########

            log_acceptance = log_hastings_ratio + log_crp_ratio + log_likelihood_ratio
            log_acceptance = min(0.0, log_acceptance)
        
            if log(rand()) < log_acceptance
                push!(list_of_clusters, proposed_Si, proposed_Sj)
                hyperparams.accepted_split += 1
            else
                push!(list_of_clusters, initial_S)
                hyperparams.rejected_split += 1
            end
        
            filter!(c -> !isempty(c), list_of_clusters)

        ##############
        ### Merge! ###
        ##############
        elseif ci != cj
            
            # We are calculating
            # q(initial | merge) = q(initial | launch)
            # and we remember that 
            # q(merge | initial) = q(merge | launch) = 1
            log_q_initial_given_launch = 0.0 
        
            proposed_S = union(launch_Si, launch_Sj)
        
            for e in reverse(scheduled_elements)

                delete!(launch_Si, e)
                delete!(launch_Sj, e)

                #############

                log_weight_Si = log(length(launch_Si))

                push!(launch_Si, e)
                log_weight_Si += (log_Zniw(launch_Si, mu, lambda, psi, nu) 
                                - log_Zniw(nothing, mu, lambda, psi, nu)
                                - length(launch_Si) * d/2 * log(2pi))
                pop!(launch_Si, e)

                #############
                
                log_weight_Sj = log(length(launch_Sj))

                push!(launch_Sj, e)
                log_weight_Sj += (log_Zniw(launch_Sj, mu, lambda, psi, nu)
                                - log_Zniw(nothing, mu, lambda, psi, nu)
                                - length(launch_Sj) * d/2 * log(2pi))
                pop!(launch_Sj, e)                

                #############

                ######## Still in merge! ########

                unnorm_logp = [log_weight_Si, log_weight_Sj] / restricted_temperature
                norm_logp = unnorm_logp .- logsumexp(unnorm_logp)

                if e in initial_Si
                    log_q_initial_given_launch += norm_logp[1]
                    push!(launch_Si, e)
                elseif e in initial_Sj
                    log_q_initial_given_launch += norm_logp[2]
                    push!(launch_Sj, e)
                else
                    error("Something went wrong during merge")
                end
            end
        
            log_hastings_ratio = log_q_initial_given_launch
        
            n_proposed_S = length(proposed_S)
            n_initial_Si = length(initial_Si)
            n_initial_Sj = length(initial_Sj)
        
            ######## Still in merge! ########

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

            filter!(c -> !isempty(c), list_of_clusters)

        end
    
    end

    function advance_alpha!(
        list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
        step_type=:gaussian, step_scale=0.4)
            
        # No Cauchy because it's a very bad idea on a log scale
        if step_type == :gaussian
            step_distrib = Normal(0.0, step_scale)
        elseif step_type == :uniform
            step_distrib = Uniform(-step_scale/2, step_scale/2)
        end

        N = sum([length(c) for c in list_of_clusters])
        K = length(list_of_clusters)

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
            hyperparams.accepted_alpha += 1
        else
            hyperparams.rejected_alpha += 1
        end
    
    end

    function advance_mu!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                         random_order=true, step_scale=0.3, step_type=:gaussian)
    
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
                             step_type=:gaussian, step_scale=0.4)

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
                     for c in list_of_clusters)

        # We leave loghastings = 0.0 because the
        # Jeffreys prior over lambda is the logarithmic 
        # prior and moves are symmetric on the log scale.

        log_acc = min(0.0, log_acc)
        
        if log(rand()) < log_acc
            hyperparams.lambda = proposed_lambda
            hyperparams.accepted_lambda += 1
        else
            hyperparams.rejected_lambda += 1
        end
    
    end

    function advance_psi!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
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

    function advance_nu!(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams;
                         step_type=:gaussian, step_scale=0.2)

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

        return sqrt((polygamma(0, alpha + n) - polygamma(0, alpha))/alpha + polygamma(1, alpha + n) - polygamma(1, alpha))

    end

    # Cute result but unfortunately lead to 
    # improper a-posteriori probability in nu and psi
    function jeffreys_nu(nu::Float64, d::Int64)

        # return sqrt(1/4 * sum(polygamma.(1, nu/2 .+ 1/2 .- 1/2 * (1:d))))
        return sqrt(1/4 * sum(polygamma(1, nu/2 + (1 - i)/2) for i in 1:d))

    end


    function advance_full_sequential_gibbs!(current_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; nb_elements=:all, proposal_temperature=1.0)

        @assert proposal_temperature > 0.0 "Can't calculate Hastings ratio when temperature=0.0. Use very small number instead."
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)

        if nb_elements == :all
            nb_elements = sum([length(c) for c in current_clusters])
        end
        
        proposed_state = deepcopy(current_clusters)
        
        scheduled_elements = shuffle([el for cluster in proposed_state for el in cluster])

        for element in scheduled_elements[1:nb_elements]
            pop!(proposed_state, element)
        end

        ##### Forward/proposed transition #####
        #####  wish I could use add_to_state_gibbs
        #####  but I need the cumulative lognorm_p
        #####  to calculate the transition probability
        #####  for the Hastings ratio.

        log_gxxp, log_gxpx = 0.0, 0.0

        for element in scheduled_elements

            Nminus1 = sum([length(c) for c in proposed_state])

            if all(!isempty(c) for c in proposed_state)
                empty_cluster = Cluster(d)
                push!(proposed_state, empty_cluster)
            end
            
            log_weights = Array{Float64}(undef, length(proposed_state))
            for (i, cluster) in enumerate(proposed_state)
                log_weights[i] = (isempty(cluster) ? log(alpha) : log(length(cluster))) - log(alpha + Nminus1)
                push!(cluster, element)
                log_weights[i] += log_Zniw(cluster, mu, lambda, psi, nu)
                pop!(cluster, element)
                log_weights[i] -= log_Zniw(cluster, mu, lambda, psi, nu)
                log_weights[i] -= d/2 * log(2pi)
            end


            unnorm_logp = log_weights / proposal_temperature
            norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
            
            selected_cluster_idx = sample(1:length(proposed_state), Weights(exp.(norm_logp)))

            log_gxpx += norm_logp[selected_cluster_idx]
            
            push!(proposed_state[selected_cluster_idx], element)
            
            filter!(c -> !isempty(c), proposed_state)
        end

        ##### Backward transition from proposed to initial #####
        initial_state = deepcopy(current_clusters)

        for element in reverse(scheduled_elements)

            container, container_idx = find(element, initial_state)
            pop!(container, element)

            # Make sure there's at least one empty
            # cluster so that we calculate the
            # correct transition probabilities.
            # Otherwise keep the empty cluster where it is
            # so that container_idx remains valid.
            if !isempty(container)
                empty_cluster = Cluster(d)
                push!(initial_state, empty_cluster)
            end
            
            Nminus1 = sum([length(c) for c in initial_state])

            log_weights = Array{Float64}(undef, length(initial_state))
            for (i, cluster) in enumerate(initial_state)

                log_weights[i] = (isempty(cluster) ? log(alpha) : log(length(cluster))) - log(alpha + Nminus1)
                
                push!(cluster, element)
                log_weights[i] += log_Zniw(cluster, mu, lambda, psi, nu)
                pop!(cluster, element)
                log_weights[i] -= log_Zniw(cluster, mu, lambda, psi, nu)
                log_weights[i] -= d/2 * log(2pi)

            end

            unnorm_logp = log_weights / proposal_temperature
            norm_logp = unnorm_logp .- logsumexp(unnorm_logp)
            log_gxxp += norm_logp[container_idx]
            filter!(c -> !isempty(c), initial_state)
        end

        log_acceptance = log_Pgenerative(proposed_state, hyperparams) - log_Pgenerative(current_clusters, hyperparams)
        println("log_metropolis  $log_acceptance")
        println("      log_gxxp  $log_gxxp")
        println("      log_gxpx  $log_gxpx")
        println("  log_hastings  $(log_gxxp - log_gxpx)")

        # Hastings ratio
        log_acceptance += log_gxxp - log_gxpx
        log_acceptance = min(0.0, log_acceptance)
        
        println("log_acceptance  $log_acceptance")
        println()

        if log(rand()) < log_acceptance
            empty!(current_clusters)
            append!(current_clusters, proposed_state)
            hyperparams.accepted_fullseq += 1
            println("accepted")
        else
            hyperparams.rejected_fullseq += 1
            println("rejected")
        end

        return current_clusters

    end

    function initiate_chain(filename::AbstractString)
        # Expects a JLD2 file
        return load(filename)
    end

    function initiate_chain(data::Vector{Vector{Float64}}; strategy=:fullseq)

        @assert all(size(e, 1) == size(first(data), 1) for e in data)

        d = size(first(data), 1)

        hyperparams = MNCRPhyperparams(d)

        chain = MNCRPchain([], hyperparams, [], [], [], [], hyperparams, -Inf, 1)

        chain.hyperparams_chain = [hyperparams]
        
        if strategy == :fullseq
            ##### 1st initialization method: fullseq
            chain.clusters = [Cluster(collect.(data))]
            for i in 1:10
                advance_full_sequential_gibbs!(chain.clusters, chain.hyperparams)
            end

        ####### 2nd initialization method: marriage problem
        # best_yet = -Inf
        # nb_test_suitors = 10
        # for i in Iterators.countfrom(1, 1)
        #     print("\rSuitor $(i)")
        #     advance_full_sequential_gibbs!(chain.clusters, chain.hyperparams)
        #     logprob = log_Pgenerative(chain.clusters, chain.hyperparams)
        #     if logprob > best_yet
        #         best_yet = logprob
        #         if i > nb_test_suitors
        #             println("!")
        #             break
        #         end
        #     end
        # end
        
        ####### 3rd initialization method: N clusters
        elseif strategy == :N
            chain.clusters = [Cluster([datum]) for datum in data]
        # @assert all(!isempty(c) for c in chain.clusters)

        elseif strategy == :1
        ####### 4rd initialization method: 1 cluster
            chain.clusters = [Cluster(data)]
        
        ####### 5th initialization method: K clusters
        # K = 3
        # chain.clusters = [Set{Vector{Float64}}(p)
        #                         for p in Iterators.partition(data, ceil(length(data)/K))]
        ##########################


        ######## 6th initialization method: Random CRP partition
        elseif strategy == :crp
            alpha = chain.hyperparams.alpha
            chain.clusters = Cluster[]
            N = 0
            for datum in data
                push!(chain.clusters, Cluster(d))
                log_crp = [isempty(cl) ? log(alpha) - log(N + alpha) : log(length(cl)) - log(N + alpha)
                                    for cl in chain.clusters]
                probs = Weights(exp.(log_crp))
                selected_cluster = sample(chain.clusters, probs)
                push!(selected_cluster, datum)
                N += 1
                filter!(x -> !isempty(x), chain.clusters)
            end
        ##########################
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
        nb_splitmerge=5, splitmerge_t=2, restricted_temperature=1.0,
        nb_gibbs=1, temperature=1.0,
        nb_hyperparamsmh=10, step_mult=1.0,
        fullseq_prob=0.0, fullseq_prop_temp=0.01,
        checkpoint_every=-1, checkpoint_prefix="chain",
        pretty_progress=true)
        
        # _nb_splitmerge = div(nb_splitmerge, 2)
        _nb_splitmerge = nb_splitmerge
        # Used for printing stats #
        hp = chain.hyperparams

        last_accepted_split = hp.accepted_split
        last_rejected_split = hp.rejected_split
        split_total = 0

        last_accepted_merge = hp.accepted_merge
        last_rejected_merge = hp.rejected_merge
        merge_total = 0
        
        nb_fullseq_moves = 0
        
        nb_map_attemps = 0
        nb_map_successes = 0

        last_checkpoint = -1
        last_map_idx = chain.map_idx
        ###########################

        progbar = Progress(nb_steps; showspeed=true)

        for step in 1:nb_steps
            ## Large moves ##

            # This one might be biasing away
            # from the stationary distribution
            if rand() < fullseq_prob
                advance_full_sequential_gibbs!(chain.clusters, chain.hyperparams, proposal_temperature=fullseq_prop_temp)
                nb_fullseq_moves += 1
            end

            # Split-merge
            # random walk between 0 and nb_splitmerge
            # _nb_splitmerge += rand([-3, -1, 0, 1, 3])
            _nb_splitmerge = max(0, min(nb_splitmerge, _nb_splitmerge))
            for i in 1:_nb_splitmerge
                advance_clusters_JNrestrictedsplitmerge!(chain.clusters, chain.hyperparams, t=splitmerge_t, restricted_temperature=restricted_temperature)
            end            
            
            #################
        
            # Gibbs sweep
            for i in 1:nb_gibbs
                # print(step, " ", i)
                # print(" ", length(chain.clusters))
                # print([length(c) for c in chain.clusters])
                advance_gibbs!(chain.clusters, chain.hyperparams, temperature=temperature)
                # print([length(c) for c in chain.clusters])
                # println(" ", length(chain.clusters))
            end

            push!(chain.nbclusters_chain, length(chain.clusters))

            # print("  #cl:$(length(chain.clusters))"); flush(stdout)


            # Metropolis-Hastings moves over each parameter
            # step_scale is adjusted to roughly hit
            # an #accepted:#rejected ratio of 1
            # It was tuned with standardized data
            # namelyby subtracting the mean and dividing
            # by the standard deviation along each dimension.
            for i in 1:nb_hyperparamsmh
                advance_alpha!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_scale=0.2 * step_mult)
                
                advance_mu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_scale=0.3 * step_mult)
                
                advance_lambda!(chain.clusters, chain.hyperparams, 
                                step_type=:gaussian, step_scale=0.3 * step_mult)

                advance_psi!(chain.clusters, chain.hyperparams,
                            step_type=:gaussian, step_scale=0.1 * step_mult)

                advance_nu!(chain.clusters, chain.hyperparams, 
                            step_type=:gaussian, step_scale=0.1 * step_mult)
            end

            push!(chain.hyperparams_chain, deepcopy(chain.hyperparams))

            
            # Stats #
            split_ratio = "$(hp.accepted_split - last_accepted_split)/$(hp.rejected_split - last_rejected_split)"
            merge_ratio = "$(hp.accepted_merge - last_accepted_merge)/$(hp.rejected_merge - last_rejected_merge)"
            split_total += hp.accepted_split - last_accepted_split
            merge_total += hp.accepted_merge - last_accepted_merge
            split_per_step = round(split_total/step, digits=2)
            merge_per_step = round(merge_total/step, digits=2)
            last_accepted_split = hp.accepted_split
            last_rejected_split = hp.rejected_split
            last_accepted_merge = hp.accepted_merge
            last_rejected_merge = hp.rejected_merge
            ########################
    
            # logprob
            logprob = log_Pgenerative(chain.clusters, chain.hyperparams)
            push!(chain.logprob_chain, logprob)

            # MAP
            history_length = 500
            short_logprob_chain = chain.logprob_chain[max(1, end - history_length):end]
            
            map_success = false
            logp_quantile95 = quantile(short_logprob_chain, 0.95)
            if logprob > logp_quantile95
                # Summit attempt
                nb_map_attemps += 1
                map_success = attempt_map!(chain, max_nb_pushes=10)
                nb_map_successes += map_success
                last_map_idx = chain.map_idx
            end
            
            if checkpoint_every > 0 && 
                (mod(step, checkpoint_every) == 0
                || last_checkpoint == -1)

                last_checkpoint = length(chain.logprob_chain)
                filename = "$(checkpoint_prefix)_pid$(getpid())_iter$(last_checkpoint).jld2"
                jldsave(filename, chain)
            end

            if pretty_progress
                ProgressMeter.next!(progbar;
                showvalues=[
                (:"step", "$(step)/$(nb_steps)"),
                (:"logprob", "$(round(chain.logprob_chain[end], digits=1))"),
                (:"chain length", length(chain.logprob_chain)),
                (:"split ratio, merge ratio", split_ratio * ", " * merge_ratio * " ($(_nb_splitmerge))"),
                (:"split/step, merge/step", "$(split_per_step), $(merge_per_step)"),
                (:"clusters (>1)", "$(length(chain.clusters)) ($(length(filter(c -> length(c) > 1, chain.clusters))))"),
                (:"fullseq moves acc/rej (%)", "$(hp.accepted_fullseq)/$(hp.rejected_fullseq) ($(round(hp.accepted_fullseq/(hp.accepted_fullseq + hp.rejected_fullseq), digits=2)))"),
                (:"MAP att/succ", "$(nb_map_attemps)/$(nb_map_successes)"),
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
            
        map_clusters_attempt = deepcopy(chain.clusters)
        map_mll = log_Pgenerative(map_clusters_attempt, chain.hyperparams)
        
        # Greedy Gibbs!
        for p in 1:max_nb_pushes
            test_attempt = deepcopy(map_clusters_attempt)
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
        vline!(p_logprob, [chain.map_idx], label=nothing, color=:black)
        hline!(p_logprob, [chain.map_logprob], label=nothing, color=:black)

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

        nbc = chain.nbclusters_chain
        p_nbc = plot(burn+1:length(nbc), nbc[burn+1:end], grid=:no, label=nothing, title="#cluster chain")
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

    
    function crp_distance(ci::Int64, cj::Int64, list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        cluster1, cluster2 = list_of_clusters[ci], list_of_clusters[cj]

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
    
    function crp_distance_matrix(list_of_clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams)
    
        dist = zeros(length(list_of_clusters), length(list_of_clusters))
        
        for i in 1:length(list_of_clusters)
            for j in 1:i
                dist[i, j] = crp_distance(i, j, list_of_clusters, hyperparams)
                dist[j, i] = dist[i, j]
            end
        end

        return dist
    end

    function presence_probability(x, clusters::Vector{Cluster}, hyperparams::MNCRPhyperparams; min=0)
        
        alpha, mu, lambda, psi, nu = hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu
        d = size(mu, 1)
        N = sum([length(cl) for cl in clusters])

        clusters = filter(x -> length(x) >= min, clusters)
        push!(clusters, Cluster(d))
        log_assignments = zeros(length(clusters))
        for (i, cluster) in enumerate(clusters)

            log_assignments[i] += (isempty(cluster) ? log(alpha) : log(length(cluster))) - log(alpha + N)

            push!(cluster, x)
            log_assignments[i] += log_Zniw(cluster, mu, lambda, psi, nu)
            pop!(cluster, x)
            log_assignments[i] -= log_Zniw(cluster, mu, lambda, psi, nu) + d/2 * log(2pi)

        end
        pop!(clusters)

        log_assignments .-= logsumexp(log_assignments)
        
        return 1.0 - exp(log_assignments[end])

    end


end
