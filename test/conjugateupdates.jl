rng = Xoshiro(60)
for T in [Float32, Float64, BigFloat], D in [1, 2, 5, 20]
    @testset "Empty and Bit ≈ Set updated_niw_hyperparams {$T, $D}" begin
        bitclusters, setclusters = generate_data(T=T, D=D, N=40, K=10, seed=rng)
        hyperparams = FCHyperparams(T, D)
        for _ in 1:100
            perturb!(rng, hyperparams)
            @test all(collect(updated_niw_hyperparams(EmptyCluster{T, D, Nothing}(), hyperparams)) .≈ collect(niwparams(hyperparams)))
            for (clb, cls) in zip(bitclusters, setclusters)
                @test all(collect(updated_niw_hyperparams(clb, hyperparams)) .≈ collect(updated_niw_hyperparams(cls, hyperparams)))
            end
        end
    end

    @testset "log_cluster_weight neutrality {$T, $D}" begin
        for N in rand(rng, 10:100, 5)
            for K in rand(rng, 1:1:20, 5)
                bitclusters, setclusters = generate_data(T=T, D=D, N=N, K=K, seed=rng)
                hyperparams = FCHyperparams(T, D)
                perturb!(rng, hyperparams)
                for el in shuffle!(rng, allelements(bitclusters))[1:10]
                    pop!(bitclusters, el)
                    for cl in bitclusters
                        pre_x, pre_xx = cl.sum_x, cl.sum_xx
                        log_cluster_weight(el, cl, hyperparams._.pyp.alpha, niwparams(hyperparams)...)
                        @test all(pre_x .≈ cl.sum_x)
                        @test all(pre_xx .≈ cl.sum_xx)
                    end
                end
                perturb!(rng, hyperparams)
                for el in shuffle!(rng, allelements(setclusters))[1:10]
                    pop!(setclusters, el)
                    for cl in setclusters
                        pre_x, pre_xx = cl.sum_x, cl.sum_xx
                        log_cluster_weight(el, cl, hyperparams._.pyp.alpha, niwparams(hyperparams)...)
                        @test all(pre_x .≈ cl.sum_x)
                        @test all(pre_xx .≈ cl.sum_xx)
                    end
                end
            end
        end
    end
end