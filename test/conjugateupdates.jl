rng = Xoshiro(60)
for T in [Float32, Float64, BigFloat], D in [1, 5, 20]
    @testset "updated_niw_hyperparams {$T, $D}" begin
        bitclusters, setclusters = generate_data(T=T, D=D, N=40, K=10, seed=rng)
        hyperparams = FCHyperparams(T, D)
        for _ in 1:100
            FlowingClusters.backtransform!(FlowingClusters.transform!(hyperparams._) .+= 0.3 * randn(rng, T, length(hyperparams._)))
            for (clb, cls) in zip(bitclusters, setclusters)
                @test all(updated_niw_hyperparams(clb, hyperparams) .≈ updated_niw_hyperparams(cls, hyperparams))
            end
        end
    end
end