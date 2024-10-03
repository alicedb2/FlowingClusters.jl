rng = Xoshiro(60)
for T in [Float16, Float32, Float64, BigFloat], D in [1, 5, 20, 100]
    @testset "Diagnostics {$T, $D}" begin
        diag = Diagnostics(T, D)
        @test typeof(diag) === Diagnostics{T, D}

        @test diag.rejected.pyp.alpha === zero(Int)
        @test diag.rejected.niw.mu == zeros(Int, D)
        @test diag.rejected.niw.lambda === zero(Int)
        @test diag.rejected.niw.flatL == zeros(Int, div(D * (D + 1), 2))
        @test diag.rejected.niw.nu === zero(Int)
        @test diag.rejected.splitmerge.split === zero(Int)
        @test diag.rejected.splitmerge.merge === zero(Int)
        @test diag.rejected.splitmerge.splitper === zero(Int)
        @test diag.rejected.splitmerge.mergeper === zero(Int)

        if T !== BigFloat && T !== BigInt # These things create objects
            @test diag.amwg.nb_batches === zero(T)
            @test diag.amwg.logscales.pyp.alpha === zero(T)
            @test diag.amwg.logscales.niw.mu == zeros(T, D)
            @test diag.amwg.logscales.niw.lambda === zero(T)
            @test diag.amwg.logscales.niw.flatL == zeros(T, div(D * (D + 1), 2))
            @test diag.amwg.logscales.niw.nu === zero(T)
        end
    end

    for (T, f) in [(Float32, f32), (Float64, f64)], D in [1, 10]
        nn = Chain(Dense(D, 12, tanh), Dense(12, 12, tanh), Dense(12, D)) |> f
        nn_params = ComponentArray{T}(Lux.setup(Xoshiro(), nn)[1])
        diag = Diagnostics(T, D, nn_params)
        
        @test typeof(diag) === DiagnosticsFFJORD{T, D}

        @test diag.rejected.pyp.alpha === zero(Int)
        @test diag.rejected.niw.mu == zeros(Int, D)
        @test diag.rejected.niw.lambda === zero(Int)
        @test diag.rejected.niw.flatL == zeros(Int, div(D * (D + 1), 2))
        @test diag.rejected.niw.nu === zero(Int)
        @test diag.rejected.splitmerge.split === zero(Int)
        @test diag.rejected.splitmerge.merge === zero(Int)
        @test diag.rejected.splitmerge.splitper === zero(Int)
        @test diag.rejected.splitmerge.mergeper === zero(Int)
        @test diag.rejected.nn.params === zero(Int)
        @test diag.rejected.nn.t.alpha === zero(Int)
        @test diag.rejected.nn.t.scale === zero(Int)

        if T !== BigFloat && T !== BigInt # These things create objects
            @test diag.amwg.nb_batches === zero(T)
            @test diag.amwg.logscales.pyp.alpha === zero(T)
            @test diag.amwg.logscales.niw.mu == zeros(T, D)
            @test diag.amwg.logscales.niw.lambda === zero(T)
            @test diag.amwg.logscales.niw.flatL == zeros(T, div(D * (D + 1), 2))
            @test diag.amwg.logscales.niw.nu === zero(T)
            @test diag.amwg.logscales.nn.t.alpha === zero(T)
            @test diag.amwg.logscales.nn.t.scale === zero(T)
        end
    end
end