mutable struct Diagnostics
    D::Int64

    accepted_alpha::Int64
    rejected_alpha::Int64

    accepted_mu::Vector{Int64}
    rejected_mu::Vector{Int64}

    accepted_lambda::Int64
    rejected_lambda::Int64

    accepted_flatL::Vector{Int64}
    rejected_flatL::Vector{Int64}

    accepted_nu::Int64
    rejected_nu::Int64

    accepted_split::Int64
    rejected_split::Int64

    accepted_merge::Int64
    rejected_merge::Int64

    accepted_nn::Vector{Int64}
    rejected_nn::Vector{Int64}

    amwg_nbbatches::Int64
    amwg_logscales::Vector{Float64}

    am_L::Int64
    am_N::Vector{Float64}
    am_NN::Matrix{Float64}

end

function acceptance_rates(d::Diagnostics)
    return (;
        alpha = d.accepted_alpha / (d.accepted_alpha + d.rejected_alpha),
        mu = d.accepted_mu ./ (d.accepted_mu .+ d.rejected_mu),
        lambda = d.accepted_lambda / (d.accepted_lambda + d.rejected_lambda),
        flatL = d.accepted_flatL ./ (d.accepted_flatL .+ d.rejected_flatL),
        nu = d.accepted_nu / (d.accepted_nu + d.rejected_nu),
        split = d.accepted_split / (d.accepted_split + d.rejected_split),
        merge = d.accepted_merge / (d.accepted_merge + d.rejected_merge),
        nn = d.accepted_nn ./ (d.accepted_nn .+ d.rejected_nn)
    )
end

function Base.show(io::IO, d::Diagnostics)
    ar = acceptance_rates(d)
    println(io, "   Accepted/Rejected")
    println(io, "                alpha : $(round(ar.alpha, digits=2))")
    println(io, "    (mean,min,max) mu : $(round(mean(ar.mu), digits=2)), $(round(minimum(ar.mu), digits=2)), $(round(maximum(ar.mu), digits=2))")
    println(io, "               lambda : $(round(ar.lambda, digits=2))")
    println(io, "     (mean,min,max) L : $(round(mean(ar.flatL), digits=2)), $(round(minimum(ar.flatL), digits=2)), $(round(maximum(ar.flatL), digits=2))")
    println(io, "                   nu : $(round(ar.nu, digits=2))")
    println(io)
    println(io, "                split : $(round(ar.split, digits=4))")
    println(io, "                merge : $(round(ar.merge, digits=4))")
    println(io)
    println(io, "    (mean,min,max) nn : $(round(mean(ar.nn), digits=2)), $(round(minimum(ar.nn, init=0), digits=2)), $(round(maximum(ar.nn, init=0), digits=2))")
    println(io)
    print(io, "        nb parameters : $(d.D)")
end


function Diagnostics(d; nn_params=nothing)

    sizeflatL = div(d * (d + 1), 2)

    D = 3 + d + sizeflatL
    nn_D = nn_params === nothing ? 0 : size(nn_params, 1)

    return Diagnostics(
        D + nn_D,                         # D
        0, 0,                             # alpha
        zeros(Int64, d), zeros(Int64, d), # mu
        0, 0,                             # lambda
        zeros(Int64, sizeflatL), zeros(Int64, sizeflatL), # flatL (Psi)
        0, 0,                             # nu
        0, 0,                             # split
        0, 0,                             # merge
        zeros(Int64, nn_D), zeros(Int64, nn_D), # ffjord nn 
        0,                                # amwg_nbbatches
        zeros(D + nn_D),                  # amwg_logscales
        0, zeros(nn_D), zeros(nn_D, nn_D) # am sigma
        )
end

function clear_diagnostics!(diagnostics::Diagnostics; clearhyperparams=true, clearsplitmerge=true, keepstepscale=true)
    
    if clearhyperparams
        diagnostics.accepted_alpha = 0
        diagnostics.rejected_alpha = 0

        diagnostics.accepted_mu = zeros(length(diagnostics.accepted_mu))
        diagnostics.rejected_mu = zeros(length(diagnostics.rejected_mu))

        diagnostics.accepted_lambda = 0
        diagnostics.rejected_lambda = 0

        diagnostics.accepted_flatL = zeros(length(diagnostics.accepted_flatL))
        diagnostics.rejected_flatL = zeros(length(diagnostics.rejected_flatL))

        diagnostics.accepted_nu = 0
        diagnostics.rejected_nu = 0
    end

    if clearsplitmerge
        diagnostics.accepted_split = 0
        diagnostics.rejected_split = 0

        diagnostics.accepted_merge = 0
        diagnostics.rejected_merge = 0
    end

    
    if !keepstepscale
        diagnostics.amwg_nbbatches = 0
        diagnostics.amwg_logscales = zeros(size(diagnostics.amwg_logscales, 1))
    end

    return diagnostics
end

function add_one_dimension!(diagnostics::Diagnostics)
    
    d = length(diagnostics.accepted_mu)
    
    diagnostics.accepted_mu = vcat(diagnostics.accepted_mu, 0)
    diagnostics.rejected_mu = vcat(diagnostics.rejected_mu, 0)
    diagnostics.accepted_flatL = vcat(diagnostics.accepted_flatL, zeros(Int64, d + 1))
    diagnostics.rejected_flatL = vcat(diagnostics.rejected_flatL, zeros(Int64, d + 1))
    
    return diagnostics
end