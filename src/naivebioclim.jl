module NaiveBIOCLIM

using DataFrames: DataFrame
using SimpleSDMLayers: SimpleSDMLayer
using StatsBase: quantilerank

export bioclim_predictor


function bioclim_predictor(training_longitudes::Vector{<:Real}, training_latitudes::Vector{<:Real}, layers::Vector{<:SimpleSDMLayer})

    
    # maprange(x, mininput, maxinput, minoutput, maxoutput) = minoutput + (x - mininput) * (maxoutput - minoutput) / (maxinput - mininput)
    rankscore(x) = 2.0 * (x > 0.5 ? 1.0 - x : x)

    @assert !any(ismissing.(training_longitudes)) && !any(ismissing.(training_latitudes))

    # [training observations] x [layers]
    training_layers_values = [[layers[long, lat] for layers in layers] 
                                for (long, lat) in zip(training_longitudes, training_latitudes)]

    training_layers_values = Vector{Vector{Float64}}(filter(x -> all(!isnothing, x), training_layers_values))
    
    # transpose
    # [layers] x [training observations]
    training_marginals = 1.0 .* eachrow(reduce(hcat, training_layers_values))


    function score_new_observation(longitude::T, latitude::T) where {T <: Real}
        
        predictor_values = [predictor[longitude, latitude] for predictor in layers]
        if any(isnothing, predictor_values)
            return nothing
        end
        marginal_scores = [rankscore(quantilerank(training_marginal, predictor_value)) 
                            for (training_marginal, predictor_value) in zip(training_marginals, predictor_values)]
        
        return minimum(marginal_scores)

    end

    function score_new_observation(longitude::Vector{<:Real}, latitude::Vector{<:Real})
        return score_new_observation.(longitude, latitude)
    end

    function score_new_observation(longlat::Vector{<:Real})
        return score_new_observation(longlat...)
    end

    function score_new_observation(longlats::Vector{Vector{<:Real}})
        return score_new_observation.(longlats)
    end

    function score_new_observation(observations::DataFrame; longlatcols=[:decimalLongitude, :decimalLatitude])

        return Union{Float64, Nothing}[score_new_observation(long, lat) for (long, lat) in eachrow(observations[!, longlatcols])]

    end

    return score_new_observation
    
end

function bioclim_predictor(training_observations::DataFrame, layers::Vector{<:SimpleSDMLayer}; longlatcols=[:decimalLongitude, :decimalLatitude])
    
    maprange(x, mininput, maxinput, minoutput, maxoutput) = minoutput + (x - mininput) * (maxoutput - minoutput) / (maxinput - mininput)
    rankscore(x) = 2.0 * (x > 0.5 ? 1.0 - x : x)
    
    long, lat = longlatcols
    
    training_observations = filter(row -> !ismissing(row[long])&& !isnothing(row[long]) && !ismissing(row[lat]) && !isnothing(row[lat]), training_observations)

    longs, lats = 1.0 .* eachcol(train_df[:, longlatcols])

    return bioclim_predictor(longs, lats, layeres)

end

end