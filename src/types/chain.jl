mutable struct FCChain

    # Current state of the partition over samples
    # from the Chinese Restaurant Process
    clusters::Vector{Cluster}
    # Current value of hyperparameters
    hyperparams::FCHyperparams
    # Map from base data (ffjord) to original unique data
    base2original::Dict{Vector{Float64}, Vector{Float64}}

    diagnostics::Diagnostics

    # Some chains of interests
    hyperparams_chain::Vector{FCHyperparams}
    logprob_chain::Vector{Float64}
    nbclusters_chain::Vector{Int64}
    largestcluster_chain::Vector{Int64}

    clusters_samples::CircularBuffer{Vector{Cluster}}
    hyperparams_samples::CircularBuffer{FCHyperparams}
    base2original_samples::CircularBuffer{Dict{Vector{Float64}, Vector{Float64}}}
    samples_idx::CircularBuffer{Int64}

    # Maximum a-posteriori state and location
    map_clusters::Vector{Cluster}
    map_hyperparams::FCHyperparams
    map_base2original::Dict{Vector{Float64}, Vector{Float64}}
    map_logprob::Float64
    map_idx::Int64

end

function Base.show(io::IO, chain::FCChain)
    println(io, "FCChain")
    println(io, "                data dimensions: $(dimension(chain.hyperparams))")
    println(io, "                      #elements: $(sum(length.(chain.clusters)))")
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

function FCChain(filename::AbstractString)
    return JLD2.load(filename)["chain"]
end

function FCChain(dataset::Matrix{Float64}; nb_samples=200, strategy=:sequential, optimize=false, ffjord_nn=nothing)
    return FCChain(collect.(eachcol(dataset)), nb_samples=nb_samples, strategy=strategy, optimize=optimize, ffjord_nn=ffjord_nn)
end

function FCChain(
    data::Vector{Vector{Float64}};
    nb_samples=200,
    strategy=:sequential,
    optimize=false,
    ffjord_nn=nothing)

    d = length(first(data))

    @assert all(length.(data) .== d)

    hyperparams = ffjord_nn === nothing ? FCHyperparams(d) : FCHyperparams(d, ffjord_nn)
    hyperparams._.pyp.alpha = 10.0 / log(length(data))

    clusters_samples = CircularBuffer{Vector{Cluster}}(nb_samples)
    hyperparams_samples = CircularBuffer{FCHyperparams}(nb_samples)
    base2original_samples = CircularBuffer{Dict{Vector{Float64}, Vector{Float64}}}(nb_samples)
    samples_idx = CircularBuffer{Int64}(nb_samples)

    # Keep unique observations only
    unique_data = collect(Set{Vector{Float64}}(deepcopy(data)))
    println("    Loaded $(length(unique_data)) unique data points into chain (found $(length(data) - length(unique_data)) duplicates)")

    if ffjord_nn !== nothing
        unique_matdata = reduce(hcat, unique_data)
        ffjord_model = FFJORD(hyperparams.nn, (0.0, 1.0), (d,), Tsit5(), ad=AutoForwardDiff())
        ret, _ = ffjord_model(unique_matdata, hyperparams._.nn.params, hyperparams.nns)
        base_data = collect.(eachcol(ret.z))
    else
        base_data = deepcopy(unique_data)
    end

    # data and original_data are still aligned
    base2original = Dict{Vector{Float64}, Vector{Float64}}(base_data .=> unique_data)

    diagnostics = Diagnostics(d, hyperparams._.nn.params)

    chain = FCChain(
        [], hyperparams, base2original, # current state of the chain
        diagnostics,
        [], [], [], [],                 # chains of interest
        clusters_samples, hyperparams_samples, base2original_samples, samples_idx, # MCMC samples of states
        [], deepcopy(hyperparams), deepcopy(base2original), # MAP clusters, MAP hyperparams, MAP base2original
        -Inf, 1 # MAP logprob, MAP index
        )

    println("    Initializing clusters...")
    if strategy == :hot
        ##### 1st initialization method: fullseq
        chain.clusters = [Cluster(base_data)]
        for i in 1:10
            advance_gibbs!(chain.clusters, chain.hyperparams, temperature=1.2)
        end
    elseif strategy == :N
        chain.clusters = [Cluster([datum]) for datum in base_data]
    elseif strategy == :1
        chain.clusters = [Cluster(base_data)]
    elseif strategy == :sequential
        for element in base_data
            advance_gibbs!(element, chain.clusters, chain.hyperparams)
        end
    end
    if optimize
        println("    Initializing hyperparameters...")
        optimize_hyperparams!(chain.clusters, chain.hyperparams, verbose=true)
    end

    chain.nbclusters_chain = [length(chain.clusters)]
    chain.largestcluster_chain = [maximum(length.(chain.clusters))]
    chain.hyperparams_chain = [deepcopy(hyperparams)]

    chain.map_clusters = deepcopy(chain.clusters)
    lp = logprobgenerative(chain.clusters, chain.hyperparams, chain.base2original, hyperpriors=true, ffjord=true)
    chain.map_logprob = lp
    chain.logprob_chain = [lp]
    # map_hyperparams=hyperparams and map_idx=1 have already been
    # specified when calling FCChain, but let's be explicit
    chain.map_hyperparams = deepcopy(chain.hyperparams)
    chain.map_idx = 1

    chain.logprob_chain = [chain.map_logprob]

    return chain

end


alpha_chain(chain::FCChain, burn=0) = [p._.pyp.alpha for p in chain.hyperparams_chain[burn+1:end]]
mu_chain(chain::FCChain, burn=0) = [p._.niw.mu[:] for p in chain.hyperparams_chain[burn+1:end]]
mu_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, mu_chain(chain, burn))
lambda_chain(chain::FCChain, burn=0) = [p._.niw.lambda for p in chain.hyperparams_chain[burn+1:end]]
function psi_chain(chain::FCChain, burn=0; flatten=false)
    if flatten
        return [FlowingClusters.flatten(LowerTriangular(foldpsi(p._.niw.flatL))) for p in chain.hyperparams_chain[burn+1:end]]
    else
        return [foldpsi(p._.niw.flatL)[:, :] for p in chain.hyperparams_chain[burn+1:end]]
    end
end
psi_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, psi_chain(chain, burn, flatten=true))
psi_chain(::Type{Array}, chain::FCChain, burn=0) = reduce((x,y)->cat(x, y, dims=3), psi_chain(chain, burn))

flatL_chain(chain::FCChain, burn=0) = [p._.niw.flatL[:] for p in chain.hyperparams_chain[burn+1:end]]
flatL_chain(::Type{Matrix}, chain::FCChain, burn=0) = reduce(hcat, flatL_chain(chain, burn))

nu_chain(chain::FCChain, burn=0) = [p._.niw.nu for p in chain.hyperparams_chain[burn+1:end]]

logprob_chain(chain::FCChain, burn=0) = chain.logprob_chain[burn+1:end]
nbclusters_chain(chain::FCChain, burn=0) = chain.nbclusters_chain[burn+1:end]
largestcluster_chain(chain::FCChain, burn=0) = chain.largestcluster_chain[burn+1:end]

nn_chain(chain::Vector{FCHyperparams}, burn=0) = [p._.nn.params for p in chain[burn+1:end]]
nn_chain(chain::FCChain, burn=0) = [p._.nn.params for p in chain.hyperparams_chain[burn+1:end]]
nn_chain(::Type{Matrix}, chain::Union{FCChain, Vector{FCHyperparams}}, burn=0) = reduce(hcat, nn_chain(chain, burn))

nn_alpha_chain(chain::FCChain, burn=0) = chain.hyperparams.nn !== nothing ? [p._.nn.t.alpha for p in chain.hyperparams_chain[burn+1:end]] : nothing
nn_scale_chain(chain::FCChain, burn=0) = chain.hyperparams.nn !== nothing ? [p._.nn.t.scale for p in chain.hyperparams_chain[burn+1:end]] : nothing

elements(::Type{T}, chain::FCChain) where {T} = elements(T, chain.clusters)
elements(chain::FCChain) = elements(chain.clusters)

function burn!(chain::FCChain, n::Int64=0; burn_map=false)

    if n >= length(chain.logprob_chain)
        @error("Can't burn the whole chain, n must be smaller than $(length(chain.logprob_chain))")
    end

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
            chain.map_logprob = logprobgenerative(chain.map_clusters, chain.map_hyperparams, chain.map_base2original, ffjord=true)
            chain.map_idx = length(chain.logprob_chain)
        else
            chain.map_idx -= n
        end

    end

    return chain

end

function Base.length(chain::FCChain)
    return length(chain.logprob_chain)
end

function ess_rhat(chain::FCChain, burn=0)

    N = length(chain.hyperparams_chain)
    if burn >= N
        @error("Can't burn the whole chain, n must be smaller than $N")
    end
    N -= burn

    d = dimension(chain.hyperparams)
    flatL_d = size(chain.hyperparams.flatL, 1)
    nn_D = chain.hyperparams.nn_params !== nothing ? size(chain.hyperparams.nn_params, 1) : 0

    return (;
    alpha = ess_rhat(alpha_chain(chain, burn)),
    mu = ess_rhat(reshape(mu_chain(Matrix, chain, burn)', N, 1, d)),
    lambda = ess_rhat(lambda_chain(chain, burn)),
    psi = ess_rhat(reshape(psi_chain(Matrix, chain, burn)', N, 1, flatL_d)),
    flatL = ess_rhat(reshape(flatL_chain(Matrix, chain, burn)', N, 1, flatL_d)),
    nu = ess_rhat(nu_chain(chain, burn)),
    nn = chain.hyperparams.nn_params !== nothing ? ess_rhat(reshape(nn_chain(Matrix, chain, burn)', N, 1, nn_D)) : nothing,
    nn_alpha = chain.hyperparams.nn_params !== nothing ? ess_rhat(nn_alpha_chain(chain, burn)) : nothing,
    logprob = ess_rhat(logprob_chain(chain, burn)),
    nbclusters = ess_rhat(nbclusters_chain(chain, burn)),
    largestcluster = ess_rhat(largestcluster_chain(chain, burn))
    )
end

function stats(chain::FCChain; burn=0)
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