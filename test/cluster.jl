rng = Xoshiro(80)
for T in [Float32, Float64], D in [1, 5, 20]
    @testset "Cluster creation with {$T, $D}" begin
        for N in rand(rng, 10:300, 7)
            for K in rand(rng, 1:20, 7)
                @test generate_data(T=T, D=D, N=N, K=K, seed=rng, test=true)
            end
        end
    end

    @testset "BitCluster validity with {$T, $D}" begin
        for N in rand(rng, 300:100:1000, 5)
            for K in rand(rng, 1:20, 5)
                bitclusters, setclusters = generate_data(T=T, D=D, N=N, K=K, seed=rng, test=false)
                @test isvalidpartition(bitclusters)
                @test !isvalidpartition(vcat(bitclusters, rand(rng, bitclusters)))
                @test all([x in keys(setclusters[1].b2o) for x in Vector.(Ref(bitclusters), allelements(bitclusters))])
            end
        end
    end

    @testset "SetCluster validity with {$T, $D}" begin
        for N in rand(rng, 300:100:1000, 5)
            for K in rand(rng, 1:1:20, 5)
                _, setclusters = generate_data(T=T, D=D, N=N, K=K, seed=rng, test=false)
                @test isvalidpartition(bitclusters)
                @test !isvalidpartition(vcat(bitclusters, rand(rng, bitclusters)))
            end
        end
    end

    if !(T === Float16 && D === 1)
        @testset "Cluster sums with {$T, $D}" begin
            for N in rand(rng, 10:100, 5)
                for K in rand(rng, 1:1:20, 5)
                    bitclusters, setclusters = generate_data(T=T, D=D, N=N, K=K, seed=rng, test=false)
                    
                    tx = all(isapprox.(getfield.(setclusters, :sum_x), getfield.(bitclusters, :sum_x)))
                    if !tx
                        println("N=$N K=$K")
                        println("     ", getfield.(setclusters, :sum_x) .- getfield.(bitclusters, :sum_x))
                    end
                    @test all(isapprox.(getfield.(setclusters, :sum_x), getfield.(bitclusters, :sum_x)))
                    
                    txx = all(isapprox.(getfield.(setclusters, :sum_xx), getfield.(bitclusters, :sum_xx)))
                    if !txx
                        println("N=$N K=$K")
                        println(getfield.(setclusters, :sum_xx) .- getfield.(bitclusters, :sum_xx))
                    end
                    @test all(isapprox.(getfield.(setclusters, :sum_xx), getfield.(bitclusters, :sum_xx)))
                end
            end
        end
    end

    @testset "Cluster operations with {$T, $D}" begin
        for N in rand(rng, 10:200, 5)
            for K in rand(rng, 1:1:20, 5)
                bitclusters, setclusters = generate_data(T=T, D=D, N=N, K=K, seed=rng, test=false)
                # eli = rand(Int(rand(rng, bitclusters)))
                # elx = Vector(bitclusters, eli)
                # @test find(elx, setclusters)[2] === find(eli, bitclusters)[2]

                # Move a random element to a random cluster
                for _ in 1:20
                    oldc = rand(rng, 1:length(bitclusters))
                    eli = rand(collect(bitclusters[oldc]))
                    elx = Vector(bitclusters, eli)
                    newc = rand(rng, 1:length(bitclusters))
                    
                    pop!(bitclusters, eli)
                    @test !any(eli in cl for cl in bitclusters)
                    push!(bitclusters[newc], eli)
                    
                    pop!(setclusters, elx)
                    @test !any(elx in cl for cl in setclusters)
                    push!(setclusters[newc], elx)

                    @test find(elx, setclusters)[2] === find(eli, bitclusters)[2]
                    if !(T === Float16 && D === 1)
                        @test isapprox(bitclusters[oldc].sum_x, setclusters[oldc].sum_x)
                        @test isapprox(bitclusters[newc].sum_x, setclusters[newc].sum_x)
                        @test isapprox(bitclusters[oldc].sum_xx, setclusters[oldc].sum_xx)
                        @test isapprox(bitclusters[newc].sum_xx, setclusters[newc].sum_xx)
                    end
                end



            end
        end
    end
end