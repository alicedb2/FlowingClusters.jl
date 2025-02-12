abstract type AbstractDiagnostics{T, D} end

struct Diagnostics{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
    splitmerge::ComponentArray{T}
end

struct DiagnosticsFFJORD{T, D} <: AbstractDiagnostics{T, D}
    accepted::ComponentArray{Int}
    rejected::ComponentArray{Int}
    amwg::ComponentArray{T}
    am::ComponentArray{T}
    splitmerge::ComponentArray{T}
end

hasnn(diagnostics::AbstractDiagnostics) = diagnostics isa DiagnosticsFFJORD

function Diagnostics(::Type{T}, D, hpparams::Union{Nothing, ComponentArray{T}}=nothing) where T
# function Diagnostics(::Type{T}, D, nn_params::Union{Nothing, ComponentArray{T}}=nothing) where T

    sizeflatL = div(D * (D + 1), 2)

    amwg = ComponentArray{T}(
                    batch_size=30,
                    nb_batches=0,
                    batch_iter=0,
                    acceptance_target=0.234,
                    min_delta=0.01,
                    logscales=(pyp=(alpha=zero(T),),#, sigma=zero(T)),
                               niw=(mu=zeros(T, D),
                                       lambda=zero(T),
                                       flatL=zeros(T, sizeflatL),
                                       nu=zero(T)
                                       )
                            )
                    )

    splitmerge = ComponentArray{T}(alpha=0.1,
                                   lambda=1.0,
                                   K_p=0.5,
                                   K_i=0.1,
                                   nb_splitmerge=50.0,
                                   target_per_step=1.0,
                                   integral_error=0.0)

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

    if !hasnn(hpparams)

        return Diagnostics{T, D}(accepted, fill!(similar(accepted), 0), amwg, splitmerge)

    else

        accepted = vcat(accepted, ComponentArray(nn=0))

        nn_D = size(hpparams.nn.params, 1)
        # nn_D = size(hpparams, 1)

        mu0 = deepcopy(hpparams.nn.params)
        sigma0 = Matrix{T}(I(nn_D))
        logscale0 = (2 * log(2.38) - log(nn_D)) - 1
        # sigma0 = T(2.38^2 / nn_D / 10) * Matrix{T}(I(nn_D))
        # logscale0 = zero(T)

        am = ComponentArray{T}(i=1.0,
                               previous_h=0.0,
                               acceptance_target=0.234,
                            #    lambda=0.9,
                               lambda_λ = 0.6,
                               lambda_μ = 0.7,
                               lambda_Σ = 0.8,
                               C=1.0,
                               logscale=logscale0,
                               mu=mu0,
                               sigma=sigma0,
                               eps=1e-7
                               )

        return DiagnosticsFFJORD{T, D}(accepted, fill!(similar(accepted), 0), amwg, am, splitmerge)

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
        println("\n")
        println(io, "     nn")
        print(io, "       params: $(round(diagnostics.accepted.nn / (diagnostics.rejected.nn + diagnostics.accepted.nn), digits=4))")
        # println(io, "      t_alpha: $(round(diagnostics.accepted.nn.prior.alpha / (diagnostics.rejected.nn.prior.alpha + diagnostics.accepted.nn.prior.alpha), digits=4))")
        # print(io,   "      t_scale: $(round(diagnostics.accepted.nn.prior.scale / (diagnostics.rejected.nn.prior.scale + diagnostics.accepted.nn.prior.scale), digits=4))")
    end

end

function clear_diagnostics!(diagnostics::AbstractDiagnostics{T, D};
                            resethyperparams=false, resetsplitmerge=false,
                            resetnn=false, resetamwg=false, resetam=false
                            ) where {T, D}

    if resethyperparams
        diagnostics.accepted.pyp .= 0
        diagnostics.rejected.pyp .= 0
        diagnostics.accepted.niw .= 0
        diagnostics.rejected.niw .= 0
    end

    if resetsplitmerge
        diagnostics.accepted.splitmerge .= 0
        diagnostics.rejected.splitmerge .= 0
    end

    if resetamwg
        diagnostics.amwg.nb_batches = 0
        diagnostics.amwg.batch_iter = 0
        diagnostics.amwg.logscales .= 0
    end

    if hasnn(diagnostics) && resetnn
        diagnostics.accepted.nn = 0
        diagnostics.rejected.nn = 0
    end

    if hasnn(diagnostics) && resetam
        diagnostics.am.i = 1.0
        diagnostics.am.previous_h = 1.0
        diagnostics.am.mu .= 0
        diagnostics.am.sigma .= Matrix{T}(I(size(diagnostics.am.mu, 1)))
        diagnostics.am.logscale = 2 * log(2.38) - log(size(diagnostics.am.mu, 1)) - 2
    end

    return diagnostics
end