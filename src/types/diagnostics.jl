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

function Diagnostics(::Type{T}, D, nn_params::Union{Nothing, ComponentArray{T}}=nothing) where T

    sizeflatL = div(D * (D + 1), 2)

    splitmerge = ComponentArray{T}(alpha=0.003,
                                   K_p=0.5,
                                   K_i=0.1,
                                   nb_splitmerge=200.0,
                                   target_per_step=1.0,
                                   integral_error=0.0)


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

        return Diagnostics{T, D}(accepted, fill!(similar(accepted), 0), amwg, splitmerge)

    else

        accepted = ComponentArray{Int}(
                        pyp=(alpha=0,),# , sigma=0),
                        niw=(mu=zeros(D),
                             lambda=0,
                             flatL=zeros(sizeflatL),
                             nu=0),
                        splitmerge=(split=0, merge=0,
                                    splitper=0, mergeper=0),
                        nn=0
                    )

        amwg = ComponentArray{T}(
                    nbbatches=zero(T),
                    logscales=(pyp=(alpha=zero(T),),#, sigma=zero(T)),
                               niw=(mu=zeros(T, D),
                                       lambda=zero(T),
                                       flatL=zeros(T, sizeflatL),
                                       nu=zero(T)
                                       )
                            )
                    )

        nn_D = size(nn_params, 1)

        am_mu = nn_params[:]
        am_sigma = Matrix{T}(I(nn_D))
        logscale0 = (2 * log(2.38) - log(nn_D))
        # Seems to be too much for tanh networks,
        # let's reduce the step size multiplier
        # by two orders of magnitude and 
        # start the exploration gently.
        logscale0 -= 2

        # Add eps * I to the covariance matrix
        # to avoid singular matrix. Practically
        # speaking, it reactivates dead directions
        # in the proposal distribution which the
        # adaptive Metropolis algorithm can then
        # latch onto. This improves convergence
        # and mixing trememdously!
        eps = 1e-3

        am = ComponentArray{T}(i=1.0,
                               previous_h=1.0,
                               acceptance_target=0.234,
                               lambda=1.0,
                               logscale=logscale0,
                               mu=am_mu,
                               sigma=am_sigma,
                               eps=eps
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

function clear_diagnostics!(diagnostics::AbstractDiagnostics;
                            resethyperparams=true, resetsplitmerge=false,
                            resetnn=false, resetamwg=false#, resetam=false
                            )

    if resethyperparams
        diagnostics.accepted.pyp .= 0
        diagnostics.rejected.pyp .= 0
        diagnostics.accepted.niw .= 0
        diagnostics.rejected.niw .= 0
        # if hasnn(diagnostics)
        #     diagnostics.accepted.nn = 0
        #     diagnostics.rejected.nn = 0
        # end
    end

    # if hasnn(diagnostics) && resetnn
    #     diagnostics.accepted.nn.params = 0
    #     diagnostics.rejected.nn.params = 0
    # end

    if resetsplitmerge
        diagnostics.accepted.splitmerge .= 0
        diagnostics.rejected.splitmerge .= 0
    end

    if resetamwg
        diagnostics.amwg .= 0
    end

    if hasnn(diagnostics) && resetnn
        diagnostics.am.nn = 0
    end

    return diagnostics
end