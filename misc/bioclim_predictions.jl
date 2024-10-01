function evaluate_bioclim(dataset::SMSDataset, species, predictors, perfstat=:MCC)
    
    training = dataset.training.presence(species).standardize(predictors...)(predictors...)
    validation = dataset.validation.standardize(predictors...)(predictors...)
    test = dataset.test.standardize(predictors...)(predictors...)

    validation_pres_mask = dataset.validation(species) .== 1
    validation_abs_mask = dataset.validation(species) .== 0
    test_pres_mask = dataset.test(species) .== 1
    test_abs_mask = dataset.test(species) .== 0

    bioclim = bioclim_predictor(training)

    bioclim_validation_scores = bioclim(validation)
    best_thresh = best_score_threshold(bioclim_validation_scores[validation_pres_mask], bioclim_validation_scores[validation_abs_mask], statistic=perfstat)
    bioclim_test_scores = bioclim(test)
        
    bioclim_test_performances = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask]))
    bioclim_test_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(bioclim_test_scores[test_pres_mask], bioclim_test_scores[test_abs_mask], threshold=best_thresh))
    println("BIOCLIM stats:\n$bioclim_test_performances")
    println("BIOCLIM stats at best thresh:\n$bioclim_test_performances_atthresh")

    return (test_performances=bioclim_test_performances, test_performances_atthresh=bioclim_test_performances_atthresh, best_thresh=best_thresh)
end


species = :sp1
predictors = (:BIO1, :BIO12)
perfstat = :MCC
dataset = eb

bioclim_perfs = evaluate_bioclim(dataset, species, predictors, perfstat)