# using Distributions: Uniform

# function CLR(
#         vector::Vector{<:AbstractFloat}; 
#         ignore_zeros=false,
#         random_imputation=true,
#         detection_limit=0.01)
    
#     @assert isapprox(sum(vector), 1.0, atol=1e-10) "Sum of vector elements should be close to 1"
    
#     vector = copy(vector)
    
#     if ignore_zeros == true
        
#         nonzero_idx = vector .> 0
#         nonzero_vector = vector[nonzero_idx]
#         geom_mean = exp(sum(log.(nonzero_vector))/length(nonzero_vector))
#         vector[nonzero_idx] = log.(nonzero_vector) .- log(geom_mean)
        
#     elseif (!ignore_zeros)
        
#         zero_idx = vector .== 0

#         if random_imputation
#             vector[zero_idx] .= rand(Uniform(0.1 * detection_limit, detection_limit), length(vector[zero_idx]))
#         else
#             vector[zero_idx] .= 0.65 * detection_limit
#         end
        
#         sum_imputed = sum(vector[zero_idx])
#         @assert sum_imputed < 1.0
#         vector[.!zero_idx] = vector[.!zero_idx] .* (1.0 - sum_imputed)
#         geom_mean = exp(sum(log.(vector))/length(vector))
#         vector = log.(vector) .- log(geom_mean)
        
#     end
    
#     return vector
            
# end

function clr(layers::Array{T}; detection_limit=1.0, uniform=true) where {T <: SimpleSDMLayer}
    @assert SimpleSDMLayers._layers_are_compatible(layers)
    l, r, b, t = first(layers).left, first(layers).right, first(layers).bottom, first(layers).top
    clr_layers = [SimpleSDMPredictor(Array{Union{Nothing, Float64}}(nothing, size(first(layers).grid)), l, r, b, t) for layer in layers]
    
    p = Progress(prod(size(first(layers).grid)), showspeed=true)
    
    Threads.@threads for i in eachindex(first(layers).grid)
        x = [isnothing(layer.grid[i]) ? nothing : Float64(layer.grid[i]) for layer in layers]
        if any(isnothing.(x))
            for l in clr_layers
                l.grid[i] = nothing
            end
        else
            c = sum(x)
            zeros_mask = x .== 0
            x[zeros_mask] .= uniform ? detection_limit * (0.1 .+ 0.55 * rand(sum(zeros_mask))) : 0.65 * detection_limit
            x[.!zeros_mask] *= 1 - sum(x[zeros_mask])/c
        end
        for (xi, l) in zip(x, clr_layers)
            l.grid[i] = log(xi) - 1/length(x) * sum(log.(x))
        end
        next!(p)
    end
    return clr_layers
end
