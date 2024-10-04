abstract type AbstractDiagnostics{T, D} end

struct Diagnostics{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
end

struct DiagnosticsFFJORD{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
    am::ComponentArray{T}
end

hasnn(diagnostics::AbstractDiagnostics) = diagnostics isa DiagnosticsFFJORD

function Diagnostics(::Type{T}, D, nn_params::Union{Nothing, ComponentArray{T}}=nothing) where T

    sizeflatL = div(D * (D + 1), 2)

    if nn_params === nothing
        accepted = ComponentArray{Int}(
                        pyp=(alpha=0,),#, sigma=0),
                        niw=(mu=zeros(D),
                            lambda=0,
                            flatL=zeros(sizeflatL),
                            nu=0
                            ),
                        splitmerge=(split=0, merge=0,
                                    splitper=0, mergeper=0
                                    )
                    )

        amwg = ComponentArray{T}(
                nbbatches=zero(T),
                logscales=(pyp=(alpha=zero(T),),# sigma=zero(T)),
                            niw=(mu=zeros(T, D),
                                lambda=zero(T),
                                flatL=zeros(T, sizeflatL),
                                nu=zero(T))
                            )
                )

        return Diagnostics{T, D}(accepted, fill!(similar(accepted), 0), amwg)
    else
        am_x = zeros(T, size(nn_params, 1))
        am_xx = zeros(T, size(nn_params, 1), size(nn_params, 1))

        accepted = ComponentArray{Int}(
                        pyp=(alpha=0,),# , sigma=0),
                        niw=(mu=zeros(D),
                             lambda=0,
                             flatL=zeros(sizeflatL),
                             nu=0),
                        splitmerge=(split=0, merge=0,
                                    splitper=0, mergeper=0),
                        nn=(params=0,
                            prior=(alpha=0,
                                   scale=0))
                    )

        amwg = ComponentArray{T}(
                    nbbatches=zero(T),
                    logscales=(pyp=(alpha=zero(T),),#, sigma=zero(T)),
                               niw=(mu=zeros(T, D),
                                       lambda=zero(T),
                                       flatL=zeros(T, sizeflatL),
                                       nu=zero(T)
                                       ),
                               nn=(params=zero(T),
                                   prior=(alpha=zero(T),
                                          scale=zero(T)
                                          )
                                )
                            )
                    )

        am = ComponentArray{T}(L=0.0, x=am_x, xx=am_xx)

        return DiagnosticsFFJORD{T, D}(accepted, fill!(similar(accepted), 0), amwg, am)
    end
end

function Base.show(io::IO, diagnostics::AbstractDiagnostics{T, D}) where {T, D}
    println(io, "Diagnostics")
    println(io, "  acceptance rates")
    println(io, "    pyp")
    println(io, "        alpha: $(round(diagnostics.accepted.pyp.alpha / (diagnostics.rejected.pyp.alpha + diagnostics.accepted.pyp.alpha), digits=4))")
    # println(io, "        sigma: $(round(diagnostics.accepted.pyp.sigma / (diagnostics.rejected.pyp.sigma + diagnostics.accepted.pyp.sigma), digits=4))")
    println(io, "    niw")
    println(io, "           mu: $(round.(diagnostics.accepted.niw.mu ./ (diagnostics.rejected.niw.mu .+ diagnostics.accepted.niw.mu), digits=4))")
    println(io, "       lambda: $(round(diagnostics.accepted.niw.lambda / (diagnostics.rejected.niw.lambda + diagnostics.accepted.niw.lambda), digits=4))")
    println(io, "        flatL: $(round.(diagnostics.accepted.niw.flatL ./ (diagnostics.rejected.niw.flatL .+ diagnostics.accepted.niw.flatL), digits=4))")
    print(io,   "           nu: $(round(diagnostics.accepted.niw.nu / (diagnostics.rejected.niw.nu + diagnostics.accepted.niw.nu), digits=4))")
    if hasnn(diagnostics)
        println()
        println(io, "    nn")
        println(io, "       params: $(round(diagnostics.accepted.nn.params / (diagnostics.rejected.nn.params + diagnostics.accepted.nn.params), digits=4))")
        println(io, "      t_alpha: $(round(diagnostics.accepted.nn.prior.alpha / (diagnostics.rejected.nn.prior.alpha + diagnostics.accepted.nn.prior.alpha), digits=4))")
        print(io,   "      t_scale: $(round(diagnostics.accepted.nn.prior.scale / (diagnostics.rejected.nn.prior.scale + diagnostics.accepted.nn.prior.scale), digits=4))")
    end

end

function clear_diagnostics!(diagnostics::AbstractDiagnostics;
                            resethyperparams=true, resetsplitmerge=false,
                            resetnn=false, resetamwg=false, resetam=false
                            )

    if resethyperparams
        diagnostics.accepted.pyp .= 0
        diagnostics.rejected.pyp .= 0
        diagnostics.accepted.niw .= 0
        diagnostics.rejected.niw .= 0
        if hasnn(diagnostics)
            diagnostics.accepted.nn.prior .= 0
            diagnostics.rejected.nn.prior .= 0
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

    if resetamwg
        diagnostics.amwg .= 0
    end

    if hasnn(diagnostics) && resetam
        diagnostics.am .= 0
    end

    return diagnostics
end