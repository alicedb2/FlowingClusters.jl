function evaluate_flowingclusters(chain::FCChain, dataset::SMSDataset, species, predictors; perfstat=:MCC, nb_rejection_samples=50_000)
    
    # Presence/absence masks
    validation_presence_mask = dataset.validation.presmask(species)
    validation_absence_mask = dataset.validation.absmask(species)
    test_presence_mask = dataset.test.presmask(species)
    test_absence_mask = dataset.test.absmask(species)
    
    # Chain has already been trained on the training set

    # MAP and chain summary prediction functions
    maptpfun = tail_probability(chain.map_clusters, chain.map_hyperparams)
    summfun = tail_probability_summary(chain.clusters_samples, chain.hyperparams_samples, nb_rejection_samples=nb_rejection_samples)
    
    # MAP predictions on validation set to find best threshold
    validation_map_presabs_tailprobs = maptpfun(dataset.validation.standardize(predictors...)(predictors...))
    best_map_thresh = best_score_threshold(validation_map_presabs_tailprobs[validation_presence_mask], validation_map_presabs_tailprobs[validation_absence_mask], statistic=perfstat)
    
    # MAP predictions on test set
    test_map_presabs_tailprobs = maptpfun(dataset.test.standardize(predictors...)(predictors...))

    # Performance of MAP predictions with and without threshold
    test_map_performances = map(x -> round(x, digits=5), performance_statistics(test_map_presabs_tailprobs[test_presence_mask], test_map_presabs_tailprobs[test_absence_mask]))
    test_map_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(test_map_presabs_tailprobs[test_presence_mask], test_map_presabs_tailprobs[test_absence_mask], threshold=best_map_thresh))
    
    println("FlowingClusters MAP performances")
    println("    MAP without threshold: $perfstat=$(getindex(test_map_performances, perfstat))")
    println("       MAP with threshold: $perfstat=$(getindex(test_map_performances_atthresh, perfstat))")
    println()

    # Chain predictions on validation set
    validation_presabs_tailprob_summaries = summfun(dataset.validation.standardize(predictors...)(predictors...))
    
    # Find best scoring and threshold using validation set
    best_thresh = nothing
    validation_performances = (MCC=-Inf, J=-Inf, kappa=-Inf)
    validation_performances_atthresh = (MCC=-Inf, J=-Inf, kappa=-Inf)
    best_scoring, best_scoring_atthresh = nothing, nothing
    for scoring in [:mean, :median, :modefd, :modedoane]
        __validation_performances = map(x -> round(x, digits=5), performance_statistics(validation_presabs_tailprob_summaries[scoring][validation_presence_mask], validation_presabs_tailprob_summaries[scoring][validation_absence_mask]))
        if __validation_performances[perfstat] > validation_performances[perfstat]
            validation_performances = __validation_performances
            best_scoring = scoring
        end

        __best_thresh = best_score_threshold(validation_presabs_tailprob_summaries[scoring][validation_presence_mask], validation_presabs_tailprob_summaries[scoring][validation_absence_mask], statistic=perfstat)
        __validation_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(validation_presabs_tailprob_summaries[scoring][validation_presence_mask], validation_presabs_tailprob_summaries[scoring][validation_absence_mask], threshold=__best_thresh))
        if __validation_performances_atthresh[perfstat] > validation_performances_atthresh[perfstat]
            best_thresh = __best_thresh
            validation_performances_atthresh = __validation_performances_atthresh
            best_scoring_atthresh = scoring
        end
    end

    # Chain predictions on test set
    test_presabs_tailprob_summaries = summfun(dataset.test.standardize(predictors...)(predictors...))

    # Performance of chain predictions
    test_performances = map(x -> round(x, digits=5), performance_statistics(test_presabs_tailprob_summaries[best_scoring][test_presence_mask], test_presabs_tailprob_summaries[best_scoring][test_absence_mask]))
    test_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(test_presabs_tailprob_summaries[best_scoring_atthresh][test_presence_mask], test_presabs_tailprob_summaries[best_scoring_atthresh][test_absence_mask], threshold=best_thresh))
    
    println("FlowingClusters chain performances")
    println("    Best without threshold: $best_scoring, $perfstat=$(getindex(test_performances, perfstat))")
    println("       Best with threshold: $best_scoring_atthresh, $perfstat=$(getindex(test_performances_atthresh, perfstat))")

    return (
    MAP=(;
        test_map_performances, 
        test_map_performances_atthresh, 
        best_map_thresh
    ),
    chain=(;
        test_performances, 
        best_scoring,
        test_performances_atthresh,
        best_thresh, 
        best_scoring_atthresh)
    )
end


###########
species = :sp4
predictors = (:BIO1, :BIO12)
dataset = eb
###########

fc_perfs = evaluate_flowingclusters(_presence_chain, dataset, species, predictors, nb_rejection_samples=50_000)
fc_perfs = evaluate_flowingclusters(chainnn, dataset, species, predictors, nb_rejection_samples=50_000)
