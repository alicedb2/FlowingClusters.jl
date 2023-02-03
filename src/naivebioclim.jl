module NaiveBIOCLIM

using DataFrames: DataFrame
using SimpleSDMLayers: SimpleSDMLayer

export bioclim_predictor

function bioclim_predictor(training_observations::DataFrame, layers::Vector{T}; longlatcols=(:decimalLongitude, :decimalLatitude) ) where {T <: SimpleSDMLayer}
    
    maprange(x, mininput, maxinput, minoutput, maxoutput) = minoutput + (x - mininput) * (maxoutput - minoutput) / (maxinput - mininput)
    rankscore(x) = 2.0 * (x > 0.5 ? 1.0 - x : x)
    
    long, lat = longlatcols
    
    training_observations = filter(row -> !ismissing(row[long])&& !isnothing(row[long]) && !ismissing(row[lat]) && !isnothing(row[lat]), obs)
    
    # Vector of vectors V such that
    # V[k][l] gives value of layer l for observation k
    obs_layer_vals = [
        [predictor[o[long], o[lat]] 
        for predictor in layers] 
        for o in eachrow(training_observations)
            ]
            
    filter!(x -> all(!isnothing, x), obs_layer_vals)
    obs_layer_vals = Vector{Float64}.(obs_layer_vals)
    # Transpose, get distribution of 
    # training observation values
    # for each layers
    obs_layer_vals = collect.(eachrow(reduce(hcat, obs_layer_vals)))
    println(typeof.(obs_layer_vals))
    
    # Rank each layer value 
    
    function score_new_observation(longitude::T, latitude::T) where {T <: Real}
        
        if ismissing(longitude) || isnothing(longitude) || ismissing(latitude) || isnothing(latitude)
            return nothing
        end
        
        if any(isnothing.([predictor[longitude, latitude] for predictor in layers]))
            return 0.0
        end
        
        obs_layer_scores = [rankscore(quantilerank(lvals, predictor[longitude, latitude])) 
                            for (predictor, lvals) in zip(predictors, obs_layer_vals) 
                                    if !isnothing(predictor[longitude, latitude])]
            
        obs_score = minimum(obs_layer_scores)
        
        obs_score = obs_score > 0.05 ? obs_score : 0.0
        
        return obs_score
        
    end
        
    function score_new_observation(observations::DataFrame; longlatcols=(:decimalLongitude, :decimalLatitude))
        
        long, lat = longlatcols
        
        return score_new_observation.(observations[!, long], observations[!, lat])
    end
        
    return score_new_observation

end

end