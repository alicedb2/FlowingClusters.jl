using EvoTrees

function evaluate_brt(dataset::SMSDataset, species, predictors, perfstat=:MCC)

    _truncate(x, minval, maxval) = min.(maxval, max.(minval, x))

    # Use training set and validation set to train BRT
    # Validation set is also used to find best threshold
    train_predictors = transpose(dataset.standardize(predictors...).training(predictors...))
    train_presabs = dataset.training(species)
    
    # Train BRT and predict validation set
    validation_predictors = transpose(dataset.standardize(predictors...).validation(predictors...))
    validation_presabs = dataset.validation(species)
    gaussian_tree_parameters = EvoTreeGaussian(; loss=:gaussian, metric=:gaussian, nrounds=100, nbins=100, λ=0.0, γ=0.0, η=0.1, max_depth=7, min_weight=1.0, rowsample=0.5, colsample=1.0)
    brt_model = fit_evotree(gaussian_tree_parameters; x_train=train_predictors, y_train=train_presabs, metric=:logloss, x_eval=validation_predictors, y_eval=validation_presabs)
    brt_validation_predictions = _truncate(EvoTrees.predict(brt_model, validation_predictors)[:, 1], 0, 1)

    validation_pres_mask = dataset.validation.presmask(species)
    validation_abs_mask = dataset.validation.absmask(species)    
    brt_best_thresh = best_score_threshold(brt_validation_predictions[validation_pres_mask], brt_validation_predictions[validation_abs_mask], statistic=perfstat)

    # Predict test set and evaluate performance
    test_predictors = transpose(dataset.test.standardize(predictors...)(predictors...))
    test_predictions = _truncate(EvoTrees.predict(brt_model, test_predictors)[:, 1], 0, 1)
    test_pres_mask = dataset.test.presmask(species)
    test_abs_mask = dataset.test.absmask(species)
    brt_test_performances = map(x -> round(x, digits=3), performance_statistics(test_predictions[test_pres_mask], test_predictions[test_abs_mask]));
    brt_test_performances_atthresh = map(x -> round(x, digits=3), performance_statistics(test_predictions[test_pres_mask], test_predictions[test_abs_mask], threshold=brt_best_thresh));

    println("#####################")
    println("    BRT stats without thresh: $perfstat=$(getindex(brt_test_performances, perfstat))")
    println("    BRT stats at best thresh: $perfstat=$(getindex(brt_test_performances_atthresh, perfstat))")

    return (;brt_test_performances, brt_test_performances_atthresh, brt_best_thresh)
end

function evaluate_brt(training_predictors, training_presabs, dataset::SMSDataset, species, predictors, perfstat=:MCC)

    _truncate(x, minval, maxval) = min.(maxval, max.(minval, x))

    # Use training set and validation set to train BRT
    # Validation set is also used to find best threshold
    # train_predictors = transpose(dataset.standardize(predictors...).training(predictors...))
    # train_presabs = dataset.training(species)
    
    # Train BRT and predict validation set
    validation_predictors = transpose(dataset.standardize(predictors...).validation(predictors...))
    validation_presabs = dataset.validation(species)
    gaussian_tree_parameters = EvoTreeGaussian(; loss=:gaussian, metric=:gaussian, nrounds=100, nbins=100, λ=0.0, γ=0.0, η=0.1, max_depth=7, min_weight=1.0, rowsample=0.5, colsample=1.0)
    brt_model = fit_evotree(gaussian_tree_parameters; x_train=training_predictors, y_train=training_presabs, metric=:logloss, x_eval=validation_predictors, y_eval=validation_presabs)
    brt_validation_predictions = _truncate(EvoTrees.predict(brt_model, validation_predictors)[:, 1], 0, 1)

    validation_pres_mask = dataset.validation.presmask(species)
    validation_abs_mask = dataset.validation.absmask(species)    
    brt_best_thresh = best_score_threshold(brt_validation_predictions[validation_pres_mask], brt_validation_predictions[validation_abs_mask], statistic=perfstat)

    # Predict test set and evaluate performance
    test_predictors = transpose(dataset.test.standardize(predictors...)(predictors...))
    test_predictions = _truncate(EvoTrees.predict(brt_model, test_predictors)[:, 1], 0, 1)
    test_pres_mask = dataset.test.presmask(species)
    test_abs_mask = dataset.test.absmask(species)
    brt_test_performances = map(x -> round(x, digits=3), performance_statistics(test_predictions[test_pres_mask], test_predictions[test_abs_mask]));
    brt_test_performances_atthresh = map(x -> round(x, digits=3), performance_statistics(test_predictions[test_pres_mask], test_predictions[test_abs_mask], threshold=brt_best_thresh));

    println("#####################")
    println("    BRT stats without thresh: $perfstat=$(getindex(brt_test_performances, perfstat))")
    println("    BRT stats at best thresh: $perfstat=$(getindex(brt_test_performances_atthresh, perfstat))")

    return (;brt_test_performances, brt_test_performances_atthresh, brt_best_thresh)
end

# brt_perfs = evaluate_brt(dataset, species, predictors, perfstat);
# brt_perfs = evaluate_brt(training_predictors_for_brt', training_presabs_for_brt, dataset, species, predictors, perfstat);