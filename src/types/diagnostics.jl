mutable struct Diagnostics
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

    step_scale::Matrix{Float64}
    slice_s::Vector{Float64}

    # Global AM with componentwise adaptive scaling
    step_sigma::Matrix{Float64}
    step_mu::Vector{Float64}
    step_loglambda::Vector{Float64}
    step_i::Int64

    slice_sampler_scales::Vector{Float64}
end

function Base.show(io::IO, d::Diagnostics)
    println(io, "   Accepted/Rejected")
    println(io, "                alpha : $(round(d.accepted_alpha/d.rejected_alpha, digits=2))")
    println(io, "    (mean,min,max) mu : $(round(sum(d.accepted_mu)/sum(d.rejected_mu), digits=2)), $(round(minimum(d.accepted_mu ./ d.rejected_mu), digits=2)), $(round(maximum(d.accepted_mu ./ d.rejected_mu), digits=2))")
    println(io, "               lambda : $(round(d.accepted_lambda/d.rejected_lambda, digits=2))")
    println(io, "     (mean,min,max) L : $(round(sum(d.accepted_flatL)/sum(d.rejected_flatL), digits=2)), $(round(minimum(d.accepted_flatL ./ d.rejected_flatL), digits=2)), $(round(maximum(d.accepted_flatL ./ d.rejected_flatL), digits=2))")
    println(io, "                   nu : $(round(d.accepted_nu/d.rejected_nu, digits=2))")
    println(io)
    println(io, "                split : $(round(d.accepted_split/d.rejected_split, digits=4))")
    println(io, "                merge : $(round(d.accepted_merge/d.rejected_merge, digits=4))")
    println(io)
    print(io, "        nb parameters : $(size(d.step_scale, 1))")
end


function Diagnostics(d)

    sizeflatL = div(d * (d + 1), 2)

    D = 3 + d + sizeflatL

    return Diagnostics(
        0, 0, 
        zeros(Int64, d), zeros(Int64, d), 
        0, 0, 
        zeros(Int64, sizeflatL), zeros(Int64, sizeflatL), 
        0, 0, 
        0, 0, 
        0, 0,
        0.01 / D * diagm(ones(D)),
        ones(D),
        0.01 / D * diagm(ones(D)),
        vcat(0.0, zeros(d), 0.0, flatten(LowerTriangular(diagm(fill(1.0, d)))), 0.0),
        zeros(D),
        0,
        ones(D))
end

function clear_diagnostics!(diagnostics::Diagnostics; keepstepscale=true)
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

    diagnostics.accepted_split = 0
    diagnostics.rejected_split = 0

    diagnostics.accepted_merge = 0
    diagnostics.rejected_merge = 0

    D = size(diagnostics.step_scale, 1)

    if !keepstepscale
        diagnostics.step_scale = 0.01 ./ D * diagm(ones(D))
    end

    diagnostics.slice_s = ones(D)

    return diagnostics
end

function add_one_dimension!(diagnostics::Diagnostics)
    
    d = length(diagnostics.accepted_mu)
    
    diagnostics.accepted_mu = vcat(diagnostics.accepted_mu, 0)
    diagnostics.rejected_mu = vcat(diagnostics.rejected_mu, 0)
    diagnostics.accepted_flatL = vcat(diagnostics.accepted_flatL, zeros(Int64, d + 1))
    diagnostics.rejected_flatL = vcat(diagnostics.rejected_flatL, zeros(Int64, d + 1))
    
    new_D = 3 + (d + 1) + div((d + 1) * (d + 2), 2)
    diagnostics.step_scale = 0.01 ./ new_D * diagm(ones(new_D))
    
    diagnostics.slice_s = vcat(diagnostics.slice_s, 1.0)

    return diagnostics
end