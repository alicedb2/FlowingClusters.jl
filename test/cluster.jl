
# eb = SMSDataset("data/ebird_data/ebird_bioclim_landcover.csv", subsample=3500, splits=[1, 1, 1], seed=4242); t = eb.training.presence(:sp4).standardize(:BIO1, :BIO12)(:BIO1, :BIO12)
# dataset = eb.training.presence(:sp4).standardize(:BIO1, :BIO12)(:BIO1, :BIO12)

# Generate some data for tests

rng = Xoshiro(80)

@testset "Cluster creation" begin
    for N in rand(rng, 300:100:3000, 3)
        for K in rand(rng, 1:20, 3)
            @test generate_data(N=N, K=K, seed=rng, test=true)
        end
    end
end

@testset "Cluster validity" begin
    for N in rand(rng, 300:100:1000, 5)
        for K in rand(rng, 1:1:20, 5)
            bitclusters, setclusters = generate_data(N=N, K=K, seed=rng, test=false)
            
            @test isvalidpartition(bitclusters)
            @test !isvalidpartition(vcat(bitclusters, rand(rng, bitclusters)))

            @test isvalidpartition(setclusters)
            @test !isvalidpartition(vcat(setclusters, rand(rng, setclusters)))
        end
    end
end

@testset "Cluster math" begin
    for N in rand(rng, 300:100:1000, 5)
        for K in rand(rng, 1:1:20, 5)
            bitclusters, setclusters = generate_data(N=N, K=K, seed=rng, test=false)
            @test all(isapprox.(getfield.(setclusters, :sum_x), getfield.(bitclusters, :sum_x)))
            @test all(isapprox.(getfield.(setclusters, :sum_xx), getfield.(bitclusters, :sum_xx)))
        end
    end
end

@testset "Cluster operations" begin
    for N in rand(rng, 300:100:1000, 5)
        for K in rand(rng, 1:1:20, 5)
            bitclusters, setclusters = generate_data(N=N, K=K, seed=rng, test=false)
            

        end
    end
end