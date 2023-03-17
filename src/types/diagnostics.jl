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

end

function show(io::IO, d::Diagnostics)
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

    D = 3 + d + div(d * (d + 1), 2)

    return Diagnostics(
        0, 0, 
        zeros(Int64, d), zeros(Int64, d), 
        0, 0, 
        zeros(Int64, sizeflatL), zeros(Int64, sizeflatL), 
        0, 0, 
        0, 0, 
        0, 0,
        0.01 / D * diagm(ones(D)))
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

    if !keepstepscale
        D = size(diagnostics.step_scale, 1)
        diagnostics.step_scale = 0.01 ./ D * diagm(ones(D))
    end

    return diagnostics
end