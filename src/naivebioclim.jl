module NaiveBIOCLIM

    using DataFrames: DataFrame
    using SpeciesDistributionToolkit: SimpleSDMLayer
    using StatsBase: quantilerank

    export bioclim_predictor


    function bioclim_predictor(training_predictors::AbstractArray)

        rankscore(x) = 2.0 * (x > 0.5 ? 1.0 - x : x)

        function scorefun(predictor::AbstractArray)
            if predictor isa AbstractVector
                marginal_scores = rankscore.(quantilerank.(eachrow(training_predictors), predictor))
                return minimum(marginal_scores)
            else
                d = first(size(predictor))
                marginal_scores = mapslices(p -> rankscore.(quantilerank.(eachrow(training_predictors), p)), reshape(predictor, d, :), dims=1)
                scores = minimum.(eachcol(marginal_scores))
                return reshape(scores, size(predictor)[2:end]...)
            end
        end

        return scorefun

    end

end