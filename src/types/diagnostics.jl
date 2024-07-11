abstract type AbstractDiagnostics{T} end

struct Diagnostics{T} <: AbstractDiagnostics{T}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
end

struct DiagnosticsFFJORD{T} <: AbstractDiagnostics{T}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
    am::ComponentArray{T}
end

hasnn(diagnostics::AbstractDiagnostics) = diagnostics isa DiagnosticsFFJORD

function Diagnostics(::Type{T}, d) where T

    sizeflatL = div(d * (d + 1), 2)

    accepted = ComponentArray{Int}(
                    pyp=(alpha=0,),#, sigma=0), 
                    niw=(mu=zeros(d), 
                        lambda=0, 
                        flatL=zeros(sizeflatL), 
                        nu=0
                        ), 
                    splitmerge=(split=0, merge=0, 
                                splitper=0, mergeper=0
                                )
                )
    awmg = ComponentArray{T}(
                nb_batches=zero(T), 
                logscales=(pyp=(alpha=zero(T), sigma=zero(T)), 
                            niw=(mu=zeros(T, d), 
                                lambda=zero(T), 
                                flatL=zeros(T, sizeflatL), 
                                nu=zero(T))
                            )
                )
    return Diagnostics{T}(accepted, fill!(similar(accepted), 0), awmg)
end

function Diagnostics(::Type{T}, d, nn_params::ComponentArray{T}) where T

    sizeflatL = div(d * (d + 1), 2)

    am_x = zeros(T, size(nn_params, 1))
    am_xx = zeros(T, size(nn_params, 1), size(nn_params, 1))

    accepted = ComponentArray{Int}(
                    pyp=(alpha=0,),# , sigma=0), 
                    niw=(mu=zeros(d), 
                        lambda=0, 
                        flatL=zeros(sizeflatL), 
                        nu=0
                        ), 
                    splitmerge=(split=0, merge=0, 
                                splitper=0, mergeper=0
                                ),
                    nn=(params=0, 
                        t=(alpha=0, 
                            scale=0)
                        )
                )

    awmg = ComponentArray{T}(
                nb_batches=zero(T), 
                logscales=(pyp=(alpha=zero(T),),#, sigma=zero(T)), 
                            niw=(mu=zeros(T, d), 
                                lambda=zero(T), 
                                flatL=zeros(T, sizeflatL), 
                                nu=zero(T)),
                            nn=(t=(alpha=zero(T), scale=zero(T)))
                            )
                )

    am = ComponentArray{T}(L=0.0, x=am_x, xx=am_xx)

    return DiagnosticsFFJORD{T}(accepted, fill!(similar(accepted), 0), awmg, am)

end

function Base.show(io::IO, diagnostics::AbstractDiagnostics{T}) where T
    println(io, "Diagnostics")
    println(io, "  acceptance rates")
    println(io, "    pyp")
    println(io, "        alpha: $(round(diagnostics.accepted.pyp.alpha / (diagnostics.rejected.pyp.alpha + diagnostics.accepted.pyp.alpha), digits=2))")
    # println(io, "        sigma: $(round(diagnostics.accepted.pyp.sigma / (diagnostics.rejected.pyp.sigma + diagnostics.accepted.pyp.sigma), digits=2))")
    println(io, "    niw")
    println(io, "           mu: $(round.(diagnostics.accepted.niw.mu ./ (diagnostics.rejected.niw.mu .+ diagnostics.accepted.niw.mu), digits=2))")
    println(io, "       lambda: $(round(diagnostics.accepted.niw.lambda / (diagnostics.rejected.niw.lambda + diagnostics.accepted.niw.lambda), digits=2))")
    println(io, "        flatL: $(round.(diagnostics.accepted.niw.flatL ./ (diagnostics.rejected.niw.flatL .+ diagnostics.accepted.niw.flatL), digits=2))")
    print(io,   "           nu: $(round(diagnostics.accepted.niw.nu / (diagnostics.rejected.niw.nu + diagnostics.accepted.niw.nu), digits=2))")
    if hasnn(diagnostics)
        println()
        println(io, "    nn")
        println(io, "       params: $(round(diagnostics.accepted.nn.params / (diagnostics.rejected.nn.params + diagnostics.accepted.nn.params), digits=2))")
        println(io, "      t_alpha: $(round(diagnostics.accepted.nn.t.alpha / (diagnostics.rejected.nn.t.alpha + diagnostics.accepted.nn.t.alpha), digits=2))")
        print(io,   "      t_scale: $(round(diagnostics.accepted.nn.t.scale / (diagnostics.rejected.nn.t.scale + diagnostics.accepted.nn.t.scale), digits=2))")
    end

end

function clear_diagnostics!(diagnostics::AbstractDiagnostics; 
                            resethyperparams=true, resetplitmerge=false, 
                            resetnn=false, resetawmg=false, resetam=false
                            )

    if resethyperparams
        diagnostics.accepted.pyp .= 0
        diagnostics.rejected.pyp .= 0
        diagnostics.accepted.niw .= 0
        diagnostics.rejected.niw .= 0
        if hasnn(diagnostics)
            diagnostics.accepted.nn.t .= 0
            diagnostics.rejected.nn.t .= 0
        end
    end

    if hasnn(diagnostics) && resetnn
        diagnostics.accepted.nn.params = 0
        diagnostics.rejected.nn.params = 0
    end

    if resetsplitmerge
        diagnostics.accepted.splitmerge .= 0
        diagnostics.rejected.splitmerge .= 0
    end

    if resetawmg
        diagnostics.amwg .= 0
    end

    if hasnn(diagnostics) && resetam
        diagnostics.am .= 0
    end

    return diagnostics
end

function am_sigma(L::Int, x::Vector{T}, xx::Matrix{T}; correction=true, eps::T=1e-10) where T
    sigma = (xx - x * x' / L) / (L - 1)
    if correction
        sigma = (sigma + sigma') / 2 + eps * I
    end
    return sigma
end

am_sigma(diagnostics::DiagnosticsFFJORD{T}; correction=true, eps::T=1e-10) where T = am_sigma(diagnostics.am.L, diagnostics.am.x, diagnostics.xx, correction=correction, eps=eps) : zeros(Float64, 0, 0)


