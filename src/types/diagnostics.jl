mutable struct Diagnostics
    
    acceptance::ComponentArray{Int64}
    amwg::ComponentArray{Float64}
    am::Union{Nothing, ComponentArray{Float64}}

end

function Diagnostics(d, nn_params=nothing)

    sizeflatL = div(d * (d + 1), 2)

    if isnothing(nn_params)
        acceptance = ComponentArray{Int64}(
                        pyp=(alpha=(acc=0, rej=0), sigma=(acc=0, rej=0)), 
                        niw=(mu=(acc=zeros(d), rej=zeros(d)), 
                            lambda=(acc=0, rej=0), 
                            flatL=(acc=zeros(sizeflatL), rej=zeros(sizeflatL)), 
                            nu=(acc=0, rej=0)
                            ), 
                        splitmerge=(split=(acc=0, rej=0), merge=0, 
                                    splitper=(acc=0, rej=0), mergeper=(acc=0, rej=0)
                                    )
                    )
        awmg = ComponentArray{Float64}(
                    nb_batches=0.0, 
                    logscales=(pyp=(alpha=0.0, sigma=0.0), 
                               niw=(mu=zeros(d), 
                                       lambda=0.0, 
                                       flatL=zeros(sizeflatL), 
                                       nu=0.0)
                               )
                    )
        am = nothing
    
    else

        am_x = zeros(Float64, size(nn_params, 1))
        am_xx = zeros(Float64, size(nn_params, 1), size(nn_params, 1))

        acceptance = ComponentArray{Int64}(
                        pyp=(alpha=(acc=0, rej=0), sigma=(acc=0, rej=0)), 
                        niw=(mu=(acc=zeros(d), rej=zeros(d)), 
                            lambda=(acc=0, rej=0), 
                            flatL=(acc=zeros(sizeflatL), rej=zeros(sizeflatL)), 
                            nu=(acc=0, rej=0)
                            ), 
                        splitmerge=(split=(acc=0, rej=0), merge=0, 
                                    splitper=(acc=0, rej=0), mergeper=(acc=0, rej=0)
                                    ),
                        nn=(params=(acc=0, rej=0), 
                            t=(alpha=(acc=0, rej=0), 
                            scale=(acc=0, rej=0))
                            )
                    )

        awmg = ComponentArray{Float64}(
                    nb_batches=0.0, 
                    logscales=(pyp=(alpha=0.0, sigma=0.0), 
                               niw=(mu=zeros(d), 
                                       lambda=0.0, 
                                       flatL=zeros(sizeflatL), 
                                       nu=0.0),
                                nn=(t=(alpha=0.0, scale=0.0))
                               )
                    )

        am = ComponentArray{Float64}(L=0.0, x=am_x, xx=am_xx)

    end

    return Diagnostics(acceptance, awmg, am)
end

function Base.show(io::IO, diagnostics::Diagnostics)
    println(io, "Diagnostics")
    println(io, "  acceptance rates")
    println(io, "    pyp")
    println(io, "        alpha: $(round(diagnostics.acceptance.pyp.alpha.acc / (diagnostics.acceptance.pyp.alpha.acc + diagnostics.acceptance.pyp.alpha.rej), digits=2))")
    println(io, "        sigma: $(round(diagnostics.acceptance.pyp.sigma.acc / (diagnostics.acceptance.pyp.sigma.acc + diagnostics.acceptance.pyp.sigma.rej), digits=2))")
    println(io, "    niw")
    println(io, "           mu: $(round.(diagnostics.acceptance.niw.mu.acc ./ (diagnostics.acceptance.niw.mu.acc .+ diagnostics.acceptance.niw.mu.rej), digits=2))")
    println(io, "       lambda: $(round(diagnostics.acceptance.niw.lambda.acc / (diagnostics.acceptance.niw.lambda.acc + diagnostics.acceptance.niw.lambda.rej), digits=2))")
    println(io, "        flatL: $(round.(diagnostics.acceptance.niw.flatL.acc ./ (diagnostics.acceptance.niw.flatL.acc .+ diagnostics.acceptance.niw.flatL.rej), digits=2))")
    print(io,   "           nu: $(round(diagnostics.acceptance.niw.nu.acc / (diagnostics.acceptance.niw.nu.acc + diagnostics.acceptance.niw.nu.rej), digits=2))")
    if !isnothing(diagnostics.am)
        println()
        println(io, "    nn")
        println(io, "       params: $(round(diagnostics.acceptance.nn.params.acc / (diagnostics.acceptance.nn.params.acc + diagnostics.acceptance.nn.params.rej), digits=2))")
        println(io, "      t_alpha: $(round(diagnostics.acceptance.nn.t.alpha.acc / (diagnostics.acceptance.nn.t.alpha.acc + diagnostics.acceptance.nn.t.alpha.rej), digits=2))")
        print(io,   "      t_scale: $(round(diagnostics.acceptance.nn.t.scale.acc / (diagnostics.acceptance.nn.t.scale.acc + diagnostics.acceptance.nn.t.scale.rej), digits=2))")
    end

end

function clear_diagnostics!(diagnostics::Diagnostics; clearhyperparams=true, clearsplitmerge=true, clearnn=true, clearamwg=false, clearam=false)

    if clearhyperparams
        diagnostics.acc.pyp .= 0
        diagnostics.acc.niw .= 0
        if !isnothing(diagnostics.am)
            diagnostics.acc.nn.t .= 0
        end
    end

    if !isnothing(diagnostics.am) && clearnn
        diagnostics.acc.nn.nn .= 0
    end

    if clearsplitmerge
        diagnostics.acc.splitmerge .= 0
    end

    if clearamwg
        diagnostics.amwg .= 0
    end

    if !isnothing(diagnostics.am) && clearam
        diagnostics.am .= 0
    end

    return diagnostics
end

function am_sigma(L::Int64, x::Vector{Float64}, xx::Matrix{Float64}; correction=true, eps=1e-10)
    sigma = (xx - x * x' / L) / (L - 1)
    if correction
        sigma = (sigma + sigma') / 2 + eps * I
    end
    return sigma
end

am_sigma(diagnostics::Diagnostics; correction=true, eps=1e-10) = !isnothing(diagnostics.am) ? am_sigma(diagnostics.am.L, diagnostics.am.x, diagnostics.xx, correction=correction, eps=eps) : zeros(Float64, 0, 0)

