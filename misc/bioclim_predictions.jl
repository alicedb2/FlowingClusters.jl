species = :sp1
predictors = (:BIO1, :BIO12)
perfstat = :MCC
dataset = eb
# dataset = SMSDataset("data/ebird_data/ebird_bioclim_landcover.csv")

function evaluate_bioclim(dataset::SMSDataset, species, predictors, perfstat=:MCC)
    training = dataset.training.presence(species).standardize(predictors...)(predictors...)
    validation_pres = dataset.validation.presence(species).standardize(predictors...)(predictors...)
    validation_abs = dataset.validation.absence(species).standardize(predictors...)(predictors...)
    test_pres = dataset.test.presence(species).standardize(predictors...)(predictors...)
    test_abs = dataset.test.absence(species).standardize(predictors...)(predictors...)

    bioclim = bioclim_predictor(training)

    bioclim_validation_presence_scores = bioclim(validation_pres)
    bioclim_validation_absence_scores = bioclim(validation_abs)
    bioclim_test_presence_scores = bioclim(test_pres)
    bioclim_test_absence_scores = bioclim(test_abs)

    best_thresh = best_score_threshold(bioclim_validation_presence_scores, bioclim_validation_absence_scores, statistic=perfstat)

    bioclim_test_performances = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores))
    bioclim_test_performances_atthresh = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores, threshold=best_thresh))
    println("BIOCLIM stats:\n$bioclim_test_performances")
    println("BIOCLIM stats at best thresh:\n$bioclim_test_performances_atthresh")

    return (test_performances=bioclim_test_performances, test_performances_atthresh=bioclim_test_performances_atthresh, best_thresh=best_bioclim_thresh)
end

bioclim_perfs = evaluate_bioclim(dataset, species, predictors, perfstat)