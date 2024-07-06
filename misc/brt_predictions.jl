using EvoTrees

###########
species = :sp1
predictors = (:BIO1, :BIO12)
dataset = eb
###########

function evaluate_brt(dataset::SMSDataset, species, predictors, perfstat=:MCC)

    train_predictors = dataset.training.standardize(predictors...)(predictors...)'
    train_presabs = dataset.training(species)

    validation_predictors = dataset.validation.standardize(predictors...)(predictors...)'
    validation_presabs = dataset.validation(species)

    test_predictors = dataset.test.standardize(predictors...)(predictors...)'
    test_presabs = dataset.test(species)

    # Presence/absence masks
    validation_presence_mask = validation_presabs .== 1
    validation_absence_mask = validation_presabs .== 0
    test_presence_mask = test_presabs .== 1
    test_absence_mask = test_presabs .== 0


    gaussian_tree_parameters = EvoTreeGaussian(; loss=:gaussian, metric=:gaussian, nrounds=100, nbins=100, λ=0.0, γ=0.0, η=0.1, max_depth=7, min_weight=1.0, rowsample=0.5, colsample=1.0)
    brt_model = fit_evotree(gaussian_tree_parameters; x_train=train_predictors, y_train=train_presabs, x_eval=validation_predictors, y_eval=validation_presabs)

    brt_validation_predictions = min.(1, max.(0, getindex.(eachrow(EvoTrees.predict(brt_model, validation_predictors)), 1)));
    best_thresh = best_score_threshold(brt_validation_predictions[validation_presence_mask], brt_validation_predictions[validation_absence_mask], statistic=perfstat)

    brt_test_predictions = min.(1, max.(0, EvoTrees.predict(brt_model, test_predictors)[:, 1]));

    brt_test_performances = map(x -> round(x, digits=3), performance_statistics(brt_test_predictions[test_presence_mask], brt_test_predictions[test_absence_mask]));
    brt_test_performances_atthresh = map(x -> round(x, digits=3), performance_statistics(brt_test_predictions[test_presence_mask], brt_test_predictions[test_absence_mask], threshold=best_thresh));

    println("#####################")
    println("BRT stats:\n$brt_test_performances")
    println("BRT stats at best tresh:\n$brt_test_performances_atthresh")

    return (;brt_test_performances, brt_test_performances_atthresh, best_thresh)
end

brt_perfs = evaluate_brt(dataset, species, predictors, :MCC);