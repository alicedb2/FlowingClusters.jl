
###########
species = :sp1
predictors = (:BIO1, :BIO12)
dataset = eb
###########


## FlowingClusters without FFJORD

# _presence_chain = MNCRPChain(eb.training.presence(:sp1).standardize(:BIO1, :BIO12), nb_samples=200)
# advance_chain!(_presence_chain, Inf, nb_splitmerge=150, nb_hyperparams=2)


### FlowingClusters with FFJORD

# nn2d = Chain(
#     Dense(2, 16, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
#     Dense(16, 2, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
#     )
# presence_chain = MNCRPChain(eb.training.presence(:sp1).standardize(:BIO1, :BIO12), ffjord_nn=nn2d, nb_samples=200)
# advance_chain!(presence_chain, Inf, nb_splitmerge=150, nb_hyperparams=2)


# nn3d = Chain(
#     Dense(3, 24, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
#     Dense(24, 3, tanh, init_bias=zeros32, init_weight=identity_init(gain=0.0)), 
#     )
# presence_chain = MNCRPChain(eb.training.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12), ffjord_nn=nn3d, nb_samples=200)

function evaluate_flowingclusters(chain::MNCRPChain, dataset::SMSDataset, species, predictors; perfstat=:MCC, nb_rejection_samples=50_000)
    
    # Presence/absence masks
    validation_presence_mask = dataset.validation(species) .== 1
    validation_absence_mask = dataset.validation(species) .== 0
    test_presence_mask = dataset.test(species) .== 1
    test_absence_mask = dataset.test(species) .== 0
    
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

fc_perfs = evaluate_flowingclusters(_presence_chain, dataset, species, predictors, nb_rejection_samples=50_000)

fcffjord_perfs = evaluate_flowingclusters(presence_chain, dataset, species, predictors, nb_rejection_samples=50_000)
