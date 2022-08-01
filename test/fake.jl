using Random: seed!
using Distributions: MultivariateNormal
using LinearAlgebra: diagind, LowerTriangular
using StatsBase: mean
using MultivariateNormalCRP
using Plots
using Revise

include("../src/PseudoAbsences.jl")

function synthetic(;nb_pseudoabsences=0)
    
    # Seed the synthetic data
    seed!(5558)
    # Generate some test data
    d = 2
    nb_clusters = 16
    samples_per_cluster = 25
    mu0 = fill(0.0, d)
    lambda0 = 0.03
    psi0 = -1.0 * rand(d, d)
    psi0 = 0.5 * (psi0 + psi0')
    psi0[diagind(psi0)] = 10.0 * rand(d)
    # psi0 = (L -> L * L')(d * LowerTriangular(rand(d, d) .- 0.5))
    nu0 = d - 1.0 + 25.0

    println("Synthetic precision matrix Ψ")
    display(psi0)

    samples = zeros(d, 0)
    synthetic_state = Vector{Set{Vector{Float64}}}()
    for i in 1:nb_clusters
        # Draw mean and covariance matrix from Normal-Inverse-Wishart distribution
        # and sample from a multivariate normal with this mean and covariance matrix
        mu, sigma = drawNIW(mu0, lambda0, psi0, nu0)
        cluster_samples = rand(MultivariateNormal(mu, sigma), samples_per_cluster)
        samples = hcat(samples, cluster_samples)
        push!(synthetic_state, Set([1.0 * col for col in eachcol(cluster_samples)]))
    end

    if nb_pseudoabsences > 0
        println("Generating pseudoabsences")
        pseudoabsences = PseudoAbsences.env_pseudoabsences(samples, 1.0, nb_pseudoabsences, region_factor=1.5, shape=:ball, verbose=true)
        push!(synthetic_state, Set([1.0 * col for col in eachcol(pseudoabsences)]))
        samples = hcat(samples, pseudoabsences)
    end

    display(plot_pi_state(synthetic_state, title="Fake data"))

    # Seed the MCMC
    seed!(41)
    
    # eachcol returns some very wild type so we explicitly recast it
    data = Vector{Vector{Float64}}(collect(eachcol(samples)))
    chain_state = initiate_chain(data)

    # Profile split-merge moves
    # @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=5, nb_paramsmh=0, fullseq_prob=0.03)

    # Profile Gibbs moves
    # @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=5, nb_splitmerge=0, nb_paramsmh=0, fullseq_prob=0.0)

    # Profile Metropolis-Hastings moves for parameters
    # @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=0, nb_paramsmh=10, fullseq_prob=0.0)

    # Profile all
    # @time @profview advance_chain!(chain_state, nb_steps=20)

    return synthetic_state, chain_state
end

# synthetic_state, chain_state = synthetic(nb_pseudoabsences=16 * 25); nothing;
synthetic_state, chain_state = synthetic(); nothing;


# advance_chain!(chain_state, nb_steps=300)
# plot_chain(chain_state, burn=100)