# Must be mutable because of MAP state :(
mutable struct FCChain{T, D, E, C, H, DG}
    clusters::Vector{C}
    hyperparams::H
    diagnostics::DG

    # We could keep track of the cluster chain
    # but then the chain would blow up in size

    # Some chains of interests
    hyperparams_chain::Vector{H}
    logprob_chain::Vector{T}
    nbclusters_chain::Vector{Int}
    largestcluster_chain::Vector{Int}

    clusters_samples::CircularBuffer{Vector{C}}
    hyperparams_samples::CircularBuffer{H}
    samples_idx::CircularBuffer{Int}

    # Maximum a-posteriori state and location
    map_clusters::Vector{C}
    map_hyperparams::H
    map_logprob::T
    map_idx::Int

    rng::AbstractRNG

end

function Base.show(io::IO, chain::FCChain)
    println(io, typeof(chain))
    println(io)
    println(io, "                      #elements: $(sum(length.(chain.clusters)))")
    println(io, "                data dimensions: $(datadimension(chain.hyperparams))")
    println(io, "               model dimensions: $(modeldimension(chain.hyperparams))")
    println(io, "                   chain length: $(length(chain))")
    conv = length(chain) > 30 ? ess_rhat(largestcluster_chain(chain)[div(end, 3):end]) : (ess=0.0, rhat=0.0)
    println(io, " convergence largest (burn 50%): ess=$(round(conv.ess, digits=1)) rhat=$(round(conv.rhat, digits=3))")
    println(io, "              current #clusters: $(length(chain.clusters))")
    println(io, "                current logprob: $(round(last(chain.logprob_chain), digits=2)) (best $(round(maximum(chain.logprob_chain), digits=2)))")
    if length(chain.samples_idx) > 20
        samples_convergence = ess_rhat([maximum(length.(s)) for s in chain.clusters_samples])
    else
        samples_convergence = (ess=NaN, rhat=NaN)
    end
    println(io, "nb samples (oldest latest) conv: $(length(chain.clusters_samples))/$(length(chain.clusters_samples.buffer)) ($(length(chain.samples_idx) > 0 ? chain.samples_idx[begin] : -1) $(length(chain.samples_idx) > 0 ? chain.samples_idx[end] : -1)) ess=$(round(samples_convergence.ess, digits=1)) rhat=$(round(samples_convergence.rhat, digits=3))")
    println(io, "                    last MAP at: $(chain.map_idx)")
    println(io, "               #clusters in MAP: $(length(chain.map_clusters))")
    print(io,   "                    MAP logprob: $(round(chain.map_logprob, digits=2))")
    # chain.hyperparams.nn !== nothing ? println(io, " (minus nn $(round(chain.map_logprob - nn_prior(chain.hyperparams.nn, similar(chain.hyperparams.nn_params) .= 0.0),digits=2)))") : println(io)
end

# load_chain(filename::AbstractString) = JLD2.load(filename)["chain"]

function FCChain(dataset::AbstractMatrix{T}, cluster_type::Type{<:AbstractCluster}=SetCluster; nb_samples=200, strategy=:sequential, ffjord_nn=nothing, perturb_data=false, seed=default_rng()) where T
    return FCChain(collect.(eachcol(dataset)), cluster_type; nb_samples=nb_samples, strategy=strategy, ffjord_nn=ffjord_nn, perturb_data=perturb_data, seed=seed)
end

function FCChain(
    data::AbstractVector{<:AbstractVector{T}},
    cluster_type::Type{C}=SetCluster;
    nb_samples=200,
    strategy=:sequential,
    perturb_data=false,
    ffjord_nn=nothing,
    seed=default_rng()
    ) where {T, C <: AbstractCluster}

    if isnothing(seed)
        rng = MersenneTwister()
    elseif seed isa AbstractRNG
        rng = seed
    elseif seed isa Int
        rng = MersenneTwister(seed)
    else
        @error("seed must be an AbstractRNG, an Int, or nothing")
        return nothing
    end

    D = length(first(data))
    if !all(length.(data) .== D)
        @error("All data points must have the same length")
        return nothing
    end

    hyperparams = FCHyperparams(T, D, ffjord_nn, rng=rng)

    # Keep unique observations only
    if perturb_data
        println("    Perturbing data...")
        data = [el .+ Vector{T}(1e-6 * randn(rng, D)) for el in data]
    end
    unique_data = unique(deepcopy(data))
    println("    Loaded $(length(unique_data)) unique data points into chain (found $(length(data) - length(unique_data)) duplicates)")

    if ffjord_nn isa Chain
        unique_matdata = reduce(hcat, unique_data)
        _, base_data = forwardffjord(rng, unique_matdata, hyperparams._, hyperparams.ffjord)
        base_data = collect.(eachcol(base_data))
    else
        base_data = deepcopy(unique_data)
    end

    # data and original_data are still aligned
    if cluster_type === SetCluster
        # If I don't use collect some weird things happen and keys pick up junk
        element_type = SVector{D, T}
        base2original = Dict{SVector{D, T}, SVector{D, T}}(collect.(base_data) .=> collect.(unique_data))
        initial_elements = [SVector{D, T}(el) for el in base_data]
    elseif (cluster_type === BitCluster) ||
           (cluster_type === IndexCluster)
        element_type = Int
        base2original = cat(reduce(hcat, base_data), reduce(hcat, unique_data), dims=3)
        initial_elements = collect(1:length(base_data))
    else
        @error("Cluster must be of type SetCluster, BitCluster, or IndexCluster")
        return nothing
    end

    clusters_samples = CircularBuffer{Vector{cluster_type{T, D, element_type}}}(nb_samples)
    hyperparams_samples = CircularBuffer{typeof(hyperparams)}(nb_samples)
    samples_idx = CircularBuffer{Int}(nb_samples)

    diagnostics = Diagnostics(T, D, hyperparams._)

    chain = FCChain{T, D, element_type, cluster_type{T, D, element_type}, typeof(hyperparams), typeof(diagnostics)}(
        cluster_type{T, D, element_type}[], hyperparams, diagnostics,
        typeof(hyperparams)[], T[], Int[], Int[], # chains of interest
        clusters_samples, hyperparams_samples, samples_idx, # MCMC samples of states
        cluster_type{T, D, element_type}[], deepcopy(hyperparams), # MAP state
        -Inf, 1, # MAP logprob and location in the chain
        rng
        )

    println("    Initializing clusters...")
    if strategy == :hot || strategy isa Float64
        push!(chain.clusters, cluster_type(initial_elements, base2original))
        if strategy == :hot
            temperature = 2.0
        else
            temperature = strategy
        end
        for i in 1:10
            advance_gibbs!(rng, chain.clusters, chain.hyperparams, temperature=temperature)
        end
    elseif strategy == :N
        append!(chain.clusters, [cluster_type(el, base2original) for el in initial_elements])
    elseif strategy isa Int
        append!(chain.clusters, [cluster_type(chunk, base2original) for chunk in chunkin(initial_elements, strategy)])
    elseif strategy == :sequential
        push!(chain.clusters, cluster_type(initial_elements[1], base2original))
        for element in initial_elements[2:end]
            advance_gibbs!(rng, element, chain.clusters, chain.hyperparams)
        end
    end
    filter!(c -> !isempty(c), chain.clusters)

    # if optimize
    #     println("    Initializing hyperparameters...")
    #     optimize_hyperparams!(chain.clusters, chain.hyperparams, verbose=true)
    # end

    push!(chain.hyperparams_chain, deepcopy(hyperparams))
    lp = logprobgenerative(chain.clusters, chain.hyperparams, chain.rng, ignorehyperpriors=false, ignoreffjord=false)
    push!(chain.logprob_chain, lp)
    push!(chain.nbclusters_chain, length(chain.clusters))
    push!(chain.largestcluster_chain, maximum(length.(chain.clusters)))

    chain.map_clusters = deepcopy(chain.clusters)
    chain.map_logprob = lp
    # map_hyperparams=hyperparams and map_idx=1 have already been
    # specified when calling FCChain, but let's be explicit
    chain.map_hyperparams = deepcopy(chain.hyperparams)
    chain.map_idx = 1

    return chain

end

hasnn(chain::FCChain) = hasnn(chain.hyperparams)

alpha_chain(chain::FCChain, burn=0) = [p._.crp.alpha for p in chain.hyperparams_chain[burn+1:end]]
mu_chain(chain::FCChain, burn=0) = [p._.crp.niw.mu[:] for p in chain.hyperparams_chain[burn+1:end]]
mu_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, mu_chain(chain, burn))
lambda_chain(chain::FCChain, burn=0) = [p._.crp.niw.lambda for p in chain.hyperparams_chain[burn+1:end]]
function psi_chain(chain::FCChain, burn=0; flatten=false)
    if flatten
        return [unfold(LowerTriangular(foldpsi(p._.crp.niw.flatL))) for p in chain.hyperparams_chain[burn+1:end]]
    else
        return [foldpsi(p._.crp.niw.flatL)[:, :] for p in chain.hyperparams_chain[burn+1:end]]
    end
end
psi_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, psi_chain(chain, burn, flatten=true))
psi_chain(::Type{Array}, chain::FCChain, burn=0) = reduce((x,y)->cat(x, y, dims=3), psi_chain(chain, burn))

flatL_chain(chain::FCChain, burn=0) = [p._.crp.niw.flatL[:] for p in chain.hyperparams_chain[burn+1:end]]
flatL_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, flatL_chain(chain, burn))

nu_chain(chain::FCChain, burn=0) = [p._.crp.niw.nu for p in chain.hyperparams_chain[burn+1:end]]

logprob_chain(chain::FCChain, burn=0) = chain.logprob_chain[burn+1:end]
nbclusters_chain(chain::FCChain, burn=0) = chain.nbclusters_chain[burn+1:end]
largestcluster_chain(chain::FCChain, burn=0) = chain.largestcluster_chain[burn+1:end]

nn_params_chain(chain::Vector{<:AbstractFCHyperparams}, burn=0) = hasnn(first(chain)) ? [p._.nn.params for p in chain[burn+1:end]] : nothing
nn_params_chain(chain::FCChain, burn=0) = nn_params_chain(chain.hyperparams_chain, burn)
nn_params_chain(::Type{Matrix}, chain::Union{FCChain, Vector{<:FCHyperparamsFFJORD}}, burn=0) = reduce(hcat, nn_params_chain(chain, burn))

nn_hyperparams_chain(chain::Vector{<:AbstractFCHyperparams}, burn=0) = hasnn(first(chain)) ? [p._.nn.prior for p in chain[burn+1:end]] : nothing
nn_hyperparams_chain(chain::FCChain, burn=0) = nn_hyperparams_chain(chain.hyperparams_chain, burn)
nn_hyperparams_chain(::Type{Matrix}, chain::Union{FCChain, Vector{<:FCHyperparamsFFJORD}}, burn=0) = reduce(hcat, nn_hyperparams_chain(chain, burn))

function burn!(chain::FCChain, n=0; burn_map=false, burn_samples=false)

    # # n given as a proportion of the chain length
    # if (n isa Float64) && (-1 < n < 1)
    #     n = round(Int, n * length(chain))
    # end

    # # if n is negative, burn from the end
    # if n < 0
    #     n = length(chain.logprob_chain) + n
    # end

    # if n >= length(chain.logprob_chain)
    #     @error("Can't burn the whole chain, n must be smaller than $(length(chain.logprob_chain))")
    #     return chain
    # end

    n = _burnlength(length(chain), n)

    if n > 0

        chain.hyperparams_chain = chain.hyperparams_chain[n+1:end]
        chain.logprob_chain = chain.logprob_chain[n+1:end]
        chain.nbclusters_chain = chain.nbclusters_chain[n+1:end]
        chain.largestcluster_chain = chain.largestcluster_chain[n+1:end]

        chain.samples_idx .-= n

        if burn_map && chain.map_idx <= n
            chain.map_clusters = deepcopy(chain.clusters)
            chain.map_hyperparams = deepcopy(chain.hyperparams)
            chain.map_base2original = deepcopy(chain.base2original)
            chain.map_logprob = logprobgenerative(chain.map_clusters, chain.map_hyperparams, chain.rng)
            chain.map_idx = length(chain.logprob_chain)
        else
            chain.map_idx -= n
        end

        if burn_samples
            while !isempty(chain.samples_idx) && first(chain.samples_idx) <= 0
                popfirst!(chain.clusters_samples)
                popfirst!(chain.hyperparams_samples)
                popfirst!(chain.samples_idx)
            end
        end

    end

    return chain

end

function Base.length(chain::FCChain)
    return length(chain.logprob_chain)
end

function Base.empty!(chain::FCChain)
    empty!(chain.samples_idx)
    empty!(chain.clusters_samples)
    empty!(chain.hyperparams_samples)
    return chain
end

function ess_rhat(chain::FCChain, burn=0)

    N = length(chain.hyperparams_chain)
    if burn >= N
        @error("Can't burn the whole chain, n must be smaller than $N")
        return nothing
    end
    N -= burn

    d = datadimension(chain.hyperparams)
    flatL_d = size(chain.hyperparams._.crp.niw.flatL, 1)
    nn_D = hasnn(chain.hyperparams) ? size(chain.hyperparams._.nn, 1) : 0

    return (;
    alpha = ess_rhat(alpha_chain(chain, burn)),
    mu = ess_rhat(reshape(mu_chain(Matrix, chain, burn)', N, 1, d)),
    lambda = ess_rhat(lambda_chain(chain, burn)),
    psi = ess_rhat(reshape(psi_chain(Matrix, chain, burn)', N, 1, flatL_d)),
    flatL = ess_rhat(reshape(flatL_chain(Matrix, chain, burn)', N, 1, flatL_d)),
    nu = ess_rhat(nu_chain(chain, burn)),
    nn = hasnn(chain.hyperparams) ? ess_rhat(reshape(nn_chain(Matrix, chain, burn)', N, 1, nn_D)) : nothing,
    logprob = ess_rhat(logprob_chain(chain, burn)),
    nbclusters = ess_rhat(nbclusters_chain(chain, burn)),
    largestcluster = ess_rhat(largestcluster_chain(chain, burn))
    )
end

function stats(chain::FCChain; burn=0)
    println("MAP state")
    println(" log prob: $(chain.map_logprob)")
    println(" #cluster: $(length(chain.map_clusters))")
    println("    alpha: $(chain.map_hyperparams._.crp.alpha)")
    println("       mu: $(chain.map_hyperparams._.crp.niw.mu)")
    println("   lambda: $(chain.map_hyperparams._.crp.niw.lambda)")
    println("      psi:")
    display(chain.map_hyperparams._.crp.niw.psi)
    println("       nu: $(chain.map_hyperparams._.crp.niw.nu)")
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