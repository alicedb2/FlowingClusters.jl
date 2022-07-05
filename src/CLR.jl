using Distributions: Uniform

function CLR(
        vector::Vector{<:AbstractFloat}; 
        ignore_zeros=false,
        random_imputation=true,
        detection_limit=0.01)
    
    @assert isapprox(sum(vector), 1.0, atol=1e-10) "Sum of vector elements should be close to 1"
    
    vector = copy(vector)
    
    if ignore_zeros == true
        
        nonzero_idx = vector .> 0
        nonzero_vector = vector[nonzero_idx]
        geom_mean = exp(sum(log.(nonzero_vector))/length(nonzero_vector))
        vector[nonzero_idx] = log.(nonzero_vector) .- log(geom_mean)
        
    elseif (!ignore_zeros)
        
        zero_idx = vector .== 0

        if random_imputation
            vector[zero_idx] .= rand(Uniform(0.1 * detection_limit, detection_limit), length(vector[zero_idx]))
        else
            vector[zero_idx] .= 0.65 * detection_limit
        end
        
        sum_imputed = sum(vector[zero_idx])
        @assert sum_imputed < 1.0
        vector[.!zero_idx] = vector[.!zero_idx] .* (1.0 - sum_imputed)
        geom_mean = exp(sum(log.(vector))/length(vector))
        vector = log.(vector) .- log(geom_mean)
        
    end
    
    return vector
            
end