using Random: seed!
using Distributions: MultivariateNormal
using LinearAlgebra: diagind

using MultivariateNormalCRP

function fake()
    
    # Fake data seed
    seed!(1234)
    # Generate some test data
    d = 3
    nb_clusters = 8
    samples_per_cluster = 50
    mu0 = fill(0.0, d)
    lambda0 = 0.05
    # psi0 = [4.0 -1.0 ; -1.0 5.0]
    psi0 = -ones(d, d)
    psi0[diagind(psi0)] .= 2.0 * d
    nu0 = d - 1.0 + 10.0

    presences = zeros(d, 0)
    for i in 1:nb_clusters
        # Draw mean and covariance matrix from Normal-Inverse-Wishart distribution
        # and sample from a multivariate normal with this mean and covariance matrix
        mu, sigma = MultivariateNormalCRP.drawNIW(mu0, lambda0, psi0, nu0)
        samples = rand(MultivariateNormal(mu, sigma), samples_per_cluster)
        presences = hcat(presences, samples)
    end


    # Seed the chain
    seed!(41)
    
    # eachcol returns some very wild type so we explicitly recast it
    data = Vector{Vector{Float64}}(collect(eachcol(presences)))
    chain_state = MultivariateNormalCRP.initiate_chain(data)

    # Profile split-merge moves
    @time @profview MultivariateNormalCRP.advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=5, nb_paramsmh=0, fullseq_prob=0.0)

    # Profile Gibbs moves
    # @time @profview MultivariateNormalCRP.advance_chain!(chain_state, nb_steps=20, nb_gibbs=5, nb_splitmerge=0, nb_paramsmh=0, fullseq_prob=0.0)

    # Profile Metropolis-Hastings moves for parameters
    # @time @profview MultivariateNormalCRP.advance_chain!(chain_state, nb_steps=20, nb_gibbs=0, nb_splitmerge=0, nb_paramsmh=10, fullseq_prob=0.0)

    return chain_state
end

chain_state = fake();