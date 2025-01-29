using SplitMaskStandardize
using StatsBase: quantilerank

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

function evaluate_bioclim(dataset::SMSDataset, species, predictors, perfstat=:MCC)

    # Get BIOCLIM predictor using training set (presence-only)
    training_predictors = dataset.training.standardize(predictors...).presence(species)(predictors...)
    bioclim = bioclim_predictor(training_predictors)

    ####### Determine best threshold with validation set
    validation_predictors = dataset.validation.standardize(predictors...)(predictors...)
    validation_pres_mask = dataset.validation.presmask(species)
    validation_abs_mask = dataset.validation.absmask(species)
    bioclim_validation_scores = bioclim(validation_predictors)
    best_thresh = best_score_threshold(bioclim_validation_scores[validation_pres_mask], bioclim_validation_scores[validation_abs_mask], statistic=perfstat)

    ####### Test set performances
    test_predictors = dataset.test.standardize(predictors...)(predictors...)
    test_pres_mask = dataset.test.presmask(species)
    test_abs_mask = dataset.test.absmask(species)

    bioclim_test_scores = bioclim(test_predictors)
    bioclim_test_performances = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask]))
    bioclim_test_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask], threshold=best_thresh))
    println("#####################")
    println("    BIOCLIM stats without thresh: $perfstat=$(getindex(bioclim_test_performances, perfstat))")
    println("    BIOCLIM stats at best thresh: $perfstat=$(getindex(bioclim_test_performances_atthresh, perfstat))")

    return (test_performances=bioclim_test_performances, test_performances_atthresh=bioclim_test_performances_atthresh, best_thresh=best_thresh)
end

# Separate training predictors from dataset
# containing validation and test sets
# This is a temporary hack because we
# truncate the training set such that there
# are as many presences as absences
function evaluate_bioclim(training_presence_predictors, dataset, species, predictors, perfstat=:MCC)

    # Get BIOCLIM predictor using training set (presence-only)
    # training_predictors = dataset.training.standardize(predictors...).presence(species)(predictors...)
    bioclim = bioclim_predictor(training_presence_predictors)

    ####### Determine best threshold with validation set
    if length(dataset) == 2
        validation_dataset = dataset.training
    else
        validation_dataset = dataset.validation
    end
    validation_predictors = validation_dataset.standardize(predictors...)(predictors...)
    validation_pres_mask = validation_dataset.presmask(species)
    validation_abs_mask = validation_dataset.absmask(species)
    bioclim_validation_scores = bioclim(validation_predictors)
    best_thresh = best_score_threshold(bioclim_validation_scores[validation_pres_mask], bioclim_validation_scores[validation_abs_mask], statistic=perfstat)

    ####### Test set performances
    test_predictors = dataset.test.standardize(predictors...)(predictors...)
    test_pres_mask = dataset.test.presmask(species)
    test_abs_mask = dataset.test.absmask(species)

    bioclim_test_scores = bioclim(test_predictors)
    test_performances = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask]))
    test_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask], threshold=best_thresh))
    println("#####################")
    println("    BIOCLIM stats without thresh: $perfstat=$(getindex(test_performances, perfstat))")
    println("    BIOCLIM stats at best thresh: $perfstat=$(getindex(test_performances_atthresh, perfstat))")

    return (; test_performances,
              test_performances_atthresh,
              best_thresh)
end

# bioclim_perfs = evaluate_bioclim(dataset, species, predictors, perfstat);
# bioclim_perfs = evaluate_bioclim(training_presence_predictors, dataset, species, predictors, perfstat);