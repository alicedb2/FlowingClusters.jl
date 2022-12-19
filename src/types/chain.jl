mutable struct MNCRPchain

    # Current state of the partition over samples
    # from the Chinese Restaurant Process
    clusters::Vector{Cluster}
    # Current value of hyperparameters
    hyperparams::MNCRPhyperparams

    data_offset::Vector{Float64}
    data_scale::Vector{Float64}

    # Some chains of interests
    nbclusters_chain::Vector{Int64}
    largestcluster_chain::Vector{Int64}
    hyperparams_chain::Vector{MNCRPhyperparams}
    logprob_chain::Vector{Float64}
    observation_chain::Vector{Vector{Float64}}

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