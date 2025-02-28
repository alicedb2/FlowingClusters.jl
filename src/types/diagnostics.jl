abstract type AbstractDiagnostics{T, D} end

struct Diagnostics{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    ram::ComponentArray{T}
    ramΣ::Cholesky{T, <: AbstractMatrix{T}}
    crp_ram::ComponentArray{T}
    crp_ramΣ::Cholesky{T, <: AbstractMatrix{T}}
    splitmerge::ComponentArray{T}
end

struct DiagnosticsFFJORD{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    ram::ComponentArray{T}
    ramΣ::Cholesky{T, <: AbstractMatrix{T}}
    crp_ram::ComponentArray{T}
    crp_ramΣ::Cholesky{T, <: AbstractMatrix{T}}
    ffjord_ram::ComponentArray{T}
    ffjord_ramΣ::Cholesky{T, <: AbstractMatrix{T}}
    splitmerge::ComponentArray{T}
end

hasnn(diagnostics::AbstractDiagnostics) = diagnostics isa DiagnosticsFFJORD

function Diagnostics(::Type{T}, D, hpparams::Union{Nothing, ComponentArray{T}}=nothing) where T

    sizeflatL = div(D * (D + 1), 2)

    splitmerge = ComponentArray{T}(alpha=0.1,
                                   lambda=1.0,
                                   K_p=0.5,
                                   K_i=0.1,
                                   nb_splitmerge=50.0,
                                   target_per_step=1.0,
                                   integral_error=0.0)

    accepted = ComponentArray{Int}(
                            crp=(alpha=0,
                                 niw=(mu=zeros(D),
                                     lambda=0,
                                     flatL=zeros(sizeflatL),
                                     nu=0
                                     )
                                ),
                            splitmerge=(split=0, merge=0,
                                        splitper=0, mergeper=0
                                        )
                        )

    ram = ComponentArray{T}(i=1.0,
                            γ=0.501,
                            previous_h=0.234,
                            acceptance_target=0.234,
                        )

    modelD = modeldimension(hpparams, include_nn=true)
    ramL0 = 2.38 / sqrt(modelD - 1) * I(modelD)
    ramΣ = Cholesky(LowerTriangular(Matrix{T}(ramL0)))

    crp_ram = ComponentArray{T}(i=1.0,
                                γ=0.501,
                                previous_h=0.234,
                                acceptance_target=0.234,
                            )

    crpD = modeldimension(hpparams, include_nn=false)
    crp_ramL0 = 2.38 / sqrt(crpD - 1) / 10 * I(crpD)
    crp_ramΣ = Cholesky(LowerTriangular(Matrix{T}(crp_ramL0)))

    if !hasnn(hpparams)

        return Diagnostics{T, D}(accepted, fill!(similar(accepted), 0), ram, ramΣ, crp_ram, crp_ramΣ, splitmerge)

    else

        accepted = vcat(accepted, ComponentArray(nn=0))

        ffjordD = modeldimension(hpparams, include_nn=true) - modeldimension(hpparams, include_nn=false)
        # we're not including log(t) for now
        ffjordD -= 1

        ffjord_ram = ComponentArray{T}(i=1.0,
                                       γ=0.501,
                                       previous_h=0.234,
                                       acceptance_target=0.234,
                                   )

        ffjord_ramL0 = 2.38 / sqrt(ffjordD - 1) * I(ffjordD)
        ffjord_ramΣ = Cholesky(LowerTriangular(Matrix{T}(ffjord_ramL0)))

        return DiagnosticsFFJORD{T, D}(accepted, fill!(similar(accepted), 0), ram, ramΣ, crp_ram, crp_ramΣ, ffjord_ram, ffjord_ramΣ, splitmerge)

    end
end

function Base.show(io::IO, diagnostics::AbstractDiagnostics{T, D}) where {T, D}
    println(io, "Diagnostics")
    println(io, "  acceptance rates")
    println(io, "    crp")
    println(io, "      alpha: $(round(diagnostics.accepted.crp.alpha / (diagnostics.rejected.crp.alpha + diagnostics.accepted.crp.alpha), digits=4))")
    # println(io, "        sigma: $(round(diagnostics.accepted.crp.sigma / (diagnostics.rejected.crp.sigma + diagnostics.accepted.crp.sigma), digits=4))")
    println(io, "      niw")
    println(io, "             mu: $(round.(diagnostics.accepted.crp.niw.mu ./ (diagnostics.rejected.crp.niw.mu .+ diagnostics.accepted.crp.niw.mu), digits=4))")
    println(io, "         lambda: $(round(diagnostics.accepted.crp.niw.lambda / (diagnostics.rejected.crp.niw.lambda + diagnostics.accepted.crp.niw.lambda), digits=4))")
    println(io, "          flatL: $(round.(diagnostics.accepted.crp.niw.flatL ./ (diagnostics.rejected.crp.niw.flatL .+ diagnostics.accepted.crp.niw.flatL), digits=4))")
    print(io,   "             nu: $(round(diagnostics.accepted.crp.niw.nu / (diagnostics.rejected.crp.niw.nu + diagnostics.accepted.crp.niw.nu), digits=4))")
    if hasnn(diagnostics)
        println("\n")
        println(io, "    nn")
        print(io,   "      params: $(round(diagnostics.accepted.nn / (diagnostics.rejected.nn + diagnostics.accepted.nn), digits=4))")
        # println(io, "      t_alpha: $(round(diagnostics.accepted.nn.prior.alpha / (diagnostics.rejected.nn.prior.alpha + diagnostics.accepted.nn.prior.alpha), digits=4))")
        # print(io,   "      t_scale: $(round(diagnostics.accepted.nn.prior.scale / (diagnostics.rejected.nn.prior.scale + diagnostics.accepted.nn.prior.scale), digits=4))")
    end

end

function clear_diagnostics!(diagnostics::AbstractDiagnostics{T, D};
                            resethyperparams=false, resetsplitmerge=false,
                            resetnn=false, resetcrpram=false, resetffjordram=false
                            ) where {T, D}

    if resethyperparams
        diagnostics.accepted.crp .= 0
        diagnostics.rejected.crp .= 0
    end

    if resetsplitmerge
        diagnostics.accepted.splitmerge .= 0
        diagnostics.rejected.splitmerge .= 0
    end

    if hasnn(diagnostics) && resetnn
        diagnostics.accepted.nn = 0
        diagnostics.rejected.nn = 0
    end

    if resetcrpram
        diagnostics.crp_ram.i = 1.0
        diagnostics.cp_ram.previous_h = diagnostics.crp_ram.target_acceptance
    end

    if hasnn(diagnostics) && resetffjordram
        diagnostics.ffjord_ram.i = 1.0
        diagnostics.ffjord_ram.previous_h = diagnostics.ffjord_ram.target_acceptance
    end

    return diagnostics
end