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

# bioclim_perfs = evaluate_bioclim(dataset, species, predictors, perfstat);
# bioclim_perfs = evaluate_bioclim(training_presence_predictors, dataset, species, predictors, perfstat);