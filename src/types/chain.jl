mutable struct MNCRPchain

    # Current state of the partition over samples
    # from the Chinese Restaurant Process
    clusters::Vector{Cluster}
    # Current value of hyperparameters
    hyperparams::MNCRPhyperparams

    # original_data::Dict{Vector{Float64}, Vector{<:Real}}
    data_mean::Vector{Float64}
    data_scalematrix::Matrix{Float64}

    # Some chains of interests
    nbclusters_chain::Vector{Int64}
    largestcluster_chain::Vector{Int64}
    hyperparams_chain::Vector{MNCRPhyperparams}
    logprob_chain::Vector{Float64}
    clusters_samples::CircularBuffer{Vector{Cluster}}
    hyperparams_samples::CircularBuffer{MNCRPhyperparams}

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
logprob_chain(chain::MNCRPchain) = chain.logprob_chain
nbclusters_chain(chain::MNCRPchain) = chain.nbclusters_chain
largestcluster_chain(chain::MNCRPchain) = chain.largestcluster_chain

function elements(chain::MNCRPchain; destandardize=false)
    if !destandardize
        return Vector{Float64}[x for cluster in chain.clusters for x in cluster]
    else
        return Vector{Float64}[chain.data_scalematrix * x .+ chain.data_mean for cluster in chain.clusters for x in cluster]
    end
end

function ess(param_chain::Vector{<:Number})
    ac = autocor(param_chain, 1:length(param_chain)-1)
    ac = ac[1:findfirst(x -> x < 0, ac)-1]
    return length(param_chain) / (1 + 2 * sum(ac))
end

function ess(chain::MNCRPchain)
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


function burn!(chain::MNCRPchain, nb_samples::Int64)

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

function length(chain::MNCRPchain)
    return length(chain.logprob_chain)
end