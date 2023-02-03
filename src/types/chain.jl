mutable struct MNCRPChain

    # Current state of the partition over samples
    # from the Chinese Restaurant Process
    clusters::Vector{Cluster}
    # Current value of hyperparameters
    hyperparams::MNCRPHyperparams

    # original_data::Dict{Vector{Float64}, Vector{<:Real}}
    data_mean::Vector{Float64}
    data_scale::Vector{Float64}

    # Some chains of interests
    nbclusters_chain::Vector{Int64}
    largestcluster_chain::Vector{Int64}
    hyperparams_chain::Vector{MNCRPHyperparams}
    logprob_chain::Vector{Float64}
    clusters_samples::CircularBuffer{Vector{Cluster}}
    hyperparams_samples::CircularBuffer{MNCRPHyperparams}

    # Maximum a-posteriori state and location
    map_clusters::Vector{Cluster}
    map_hyperparams::MNCRPHyperparams
    map_logprob::Float64
    map_idx::Int64

end

function show(io::IO, chain::MNCRPChain)
    println(io, "MNCRP chain")
    println(io, "          #elements: $(sum(length.(chain.clusters)))")
    println(io, "       chain length: $(length(chain))")
    println(io, "  current #clusters: $(length(chain.clusters))")
    println(io, "    current logprob: $(round(last(chain.logprob_chain), digits=2))")
    println(io, "   nb chain samples: $(length(chain.clusters_samples))/$(length(chain.clusters_samples.buffer))")
    println()
    println(io, "       last MAP at: $(chain.map_idx)")
    println(io, "  #clusters in MAP: $(length(chain.map_clusters))")
    println(io, "       MAP logprob: $(round(chain.map_logprob, digits=2))")
end

alpha_chain(chain::MNCRPChain) = [p.alpha for p in chain.hyperparams_chain]
mu_chain(chain::MNCRPChain) = [p.mu for p in chain.hyperparams_chain]
mu_chain(chain::MNCRPChain, i) = [p.mu[i] for p in chain.hyperparams_chain]
lambda_chain(chain::MNCRPChain) = [p.lambda for p in chain.hyperparams_chain]
psi_chain(chain::MNCRPChain) = [p.psi for p in chain.hyperparams_chain]
psi_chain(chain::MNCRPChain, i, j) = [p.psi[i, j] for p in chain.hyperparams_chain]
flatL_chain(chain::MNCRPChain) = [p.flatL for p in chain.hyperparams_chain]
flatL_chain(chain::MNCRPChain, i) = [p.flatL[i] for p in chain.hyperparams_chain]
nu_chain(chain::MNCRPChain) = [p.nu for p in chain.hyperparams_chain]
logprob_chain(chain::MNCRPChain) = chain.logprob_chain
nbclusters_chain(chain::MNCRPChain) = chain.nbclusters_chain
largestcluster_chain(chain::MNCRPChain) = chain.largestcluster_chain

function elements(chain::MNCRPChain; destandardize=false)
    if !destandardize
        return Vector{Float64}[x for cluster in chain.clusters for x in cluster]
    else
        return Vector{Float64}[chain.data_scale .* x .+ chain.data_mean for cluster in chain.clusters for x in cluster]
    end
end

function ess(param_chain::Vector{<:Number})
    ac = autocor(param_chain, 1:length(param_chain)-1)
    ac = ac[1:findfirst(x -> x < 0, ac)-1]
    return length(param_chain) / (1 + 2 * sum(ac))
end

function ess(chain::MNCRPChain)
    d = length(chain.hyperparams.mu)

    alpha_ess = ess(alpha_chain(chain))
    mu_ess = [ess(mu_chain(chain, i)) for i in 1:d]
    lambda_ess = ess(lambda_chain(chain))
    psi_ess = [ess(psi_chain(chain, i, j)) for i in 1:d, j in 1:d]
    flatL_ess = [ess(flatL_chain(chain, i)) for i in 1:length(chain.hyperparams.flatL)]
    nu_ess = ess(nu_chain(chain))
    logprob_ess = ess(logprob_chain(chain))
    nbclusters_ess = ess(nbclusters_chain(chain))
    largestcluster_ess = ess(largestcluster_chain(chain))
    min_ess = minimum(vcat([alpha_ess, 
                            minimum(mu_ess),                             
                            lambda_ess, 
                            minimum(psi_ess),
                            minimum(flatL_ess),
                            nu_ess, 
                            logprob_ess,
                            nbclusters_ess,
                            largestcluster_ess]))
    
    println("         logprob ESS: $(round(logprob_ess, digits=1))")
    println("       #clusters ESS: $(round(nbclusters_ess, digits=1))")
    println(" largest cluster ESS: $(round(largestcluster_ess, digits=1))")
    println("           alpha ESS: $(round(alpha_ess, digits=1))")
    println("              mu ESS: $(round.(mu_ess, digits=1))")
    println("          lambda ESS: $(round(lambda_ess, digits=1))")
    println("             Psi ESS: $(round.(psi_ess, digits=1))")
    println("           flatL ESS: $(round.(flatL_ess, digits=1))")
    println("              nu ESS: $(round(nu_ess, digits=1))")
    println("           alpha ESS: $(round(alpha_ess, digits=1))")
    println()
    println("             min ESS: $(round(min_ess, digits=1))")

end


function burn!(chain::MNCRPChain, nb_samples::Int64)

    if nb_samples >= length(chain.logprob_chain)
        @error("Can't burn the whole chain, nb_samples must be smaller than $(length(chain.logprob_chain))")
    end

    if nb_samples <= 0
        @error("nb_samples must be at least 1")
    end

    chain.logprob_chain = chain.logprob_chain[nb_samples+1:end]
    chain.hyperparams_chain = chain.hyperparams_chain[nb_samples+1:end]
    chain.nbclusters_chain = chain.nbclusters_chain[nb_samples+1:end]
    chain.largestcluster_chain = chain.largestcluster_chain[nb_samples+1:end]

    if chain.map_idx <= nb_samples        
        chain.map_clusters = deepcopy(chain.clusters)
        chain.map_hyperparams = deepcopy(chain.hyperparams)
        chain.map_logprob = log_Pgenerative(chain.clusters, chain.hyperparams)
        chain.map_idx = length(chain.logprob_chain)
    else
        chain.map_idx -= nb_samples
    end
    
    return chain

end

function length(chain::MNCRPChain)
    return length(chain.logprob_chain)
end