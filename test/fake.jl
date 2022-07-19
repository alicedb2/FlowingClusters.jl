using Random: seed!
using Distributions: MultivariateNormal
using LinearAlgebra: diagind
using StatsBase: mean
using MultivariateNormalCRP
using Plots
using Revise

function fake()
    
    # Seed the fake data
    seed!(234)
    # Generate some test data
    d = 2
    nb_clusters = 8
    samples_per_cluster = 50
    mu0 = fill(0.0, d)
    lambda0 = 0.05
    psi0 = -1.0 * rand(d, d)
    psi0 = 0.5 * (psi0 + psi0')
    psi0[diagind(psi0)] = 10.0 * rand(d)
    nu0 = d - 1.0 + 25.0

    println("Synthetic precision matrix Ψ")
    display(psi0)

    presences = zeros(d, 0)
    fake_state = Vector{Set{Vector{Float64}}}()
    for i in 1:nb_clusters
        # Draw mean and covariance matrix from Normal-Inverse-Wishart distribution
        # and sample from a multivariate normal with this mean and covariance matrix
        mu, sigma = drawNIW(mu0, lambda0, psi0, nu0)
        samples = rand(MultivariateNormal(mu, sigma), samples_per_cluster)
        presences = hcat(presences, samples)
        push!(fake_state, Set([1.0 * col for col in eachcol(samples)]))
    end
    
    display(plot_pi_state(fake_state, title="Fake data"))

    # Seed the MCMC
    seed!(41)
    
    # eachcol returns some very wild type so we explicitly recast it
    data = Vector{Vector{Float64}}(collect(eachcol(presences)))
    chain_state = initiate_chain(data)

    # Profile split-merge moves
    @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=5, nb_paramsmh=0, fullseq_prob=0.03)

    # Profile Gibbs moves
    # @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=5, nb_splitmerge=0, nb_paramsmh=0, fullseq_prob=0.0)

    # Profile Metropolis-Hastings moves for parameters
    # @time @profview advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=0, nb_paramsmh=10, fullseq_prob=0.0)

    # Profile all
    # @time @profview advance_chain!(chain_state, nb_steps=20)

    return fake_state, chain_state
end

fake_state, chain_state = fake(); nothing;

# advance_chain!(chain_state, nb_steps=300)
# plot_chain(chain_state, burn=100)