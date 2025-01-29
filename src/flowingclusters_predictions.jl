function evaluate_flowingclusters(chain::FCChain, dataset::SMSDataset, species, predictors, perfstat=:MCC; nb_rejection_samples=50_000, per_cluster=false)

    # Chain has already been trained on the training set

    # MAP and chain summary prediction functions
    map_tailprob_fun = tail_probability(chain.map_clusters, chain.map_hyperparams, nb_rejection_samples=nb_rejection_samples, per_cluster=per_cluster)
    map_nb_clusters = length(chain.map_clusters)

    # MAP predictions on validation set to find best threshold
    validation_map_presabs_tailprobs = map_tailprob_fun(dataset.validation.standardize(predictors...)(predictors...))
    validation_presence_mask = dataset.validation.presmask(species)
    validation_absence_mask = dataset.validation.absmask(species)
    best_map_thresh = best_score_threshold(
        validation_map_presabs_tailprobs[validation_presence_mask],
        validation_map_presabs_tailprobs[validation_absence_mask],
        statistic=perfstat, nbsteps=10_000)

    # MAP predictions on test set
    test_map_presabs_tailprobs = map_tailprob_fun(dataset.test.standardize(predictors...)(predictors...))
    test_presence_mask = dataset.test.presmask(species)
    test_absence_mask = dataset.test.absmask(species)
    # Performance of MAP predictions with and without threshold
    test_map_performances = map(x -> round(x, digits=5), performance_statistics(test_map_presabs_tailprobs[test_presence_mask], test_map_presabs_tailprobs[test_absence_mask]))
    test_map_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(test_map_presabs_tailprobs[test_presence_mask], test_map_presabs_tailprobs[test_absence_mask], threshold=best_map_thresh))

    println("#####################")
    println("FlowingClusters MAP performances")
    println("    MAP without threshold: $perfstat=$(getindex(test_map_performances, perfstat))")
    println("       MAP with threshold: $perfstat=$(getindex(test_map_performances_atthresh, perfstat))")
    println()

    if chain.clusters_samples.length >= 3
        # Chain predictions on validation set
        chain_tailprob_fun = tail_probability_summary(chain.clusters_samples, chain.hyperparams_samples, nb_rejection_samples=nb_rejection_samples, per_cluster=per_cluster)

        validation_presabs_tailprob_summaries = chain_tailprob_fun(dataset.validation.standardize(predictors...)(predictors...))

        # Find best scoring and threshold using validation set
        best_thresh = nothing
        validation_performances = (MCC=-Inf, J=-Inf, kappa=-Inf)
        validation_performances_atthresh = (MCC=-Inf, J=-Inf, kappa=-Inf)
        best_scoring_method, best_scoring_atthresh = nothing, nothing
        for scoring in [:mean, :median, :modefd, :modedoane]
            __validation_performances = map(x -> round(x, digits=5), performance_statistics(validation_presabs_tailprob_summaries[scoring][validation_presence_mask], validation_presabs_tailprob_summaries[scoring][validation_absence_mask]))
            if __validation_performances[perfstat] > validation_performances[perfstat]
                validation_performances = __validation_performances
                best_scoring_method = scoring
            end

            __best_thresh = best_score_threshold(
                validation_presabs_tailprob_summaries[scoring][validation_presence_mask],
                validation_presabs_tailprob_summaries[scoring][validation_absence_mask],
                statistic=perfstat, nbsteps=10_000)

            __validation_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(validation_presabs_tailprob_summaries[scoring][validation_presence_mask], validation_presabs_tailprob_summaries[scoring][validation_absence_mask], threshold=__best_thresh))
            if __validation_performances_atthresh[perfstat] > validation_performances_atthresh[perfstat]
                best_thresh = __best_thresh
                validation_performances_atthresh = __validation_performances_atthresh
                best_scoring_atthresh = scoring
            end
        end

        # Chain predictions on test set
        test_presabs_tailprob_summaries = chain_tailprob_fun(dataset.test.standardize(predictors...)(predictors...))

        # Performance of chain predictions
        test_performances = map(x -> round(x, digits=5), performance_statistics(test_presabs_tailprob_summaries[best_scoring_method][test_presence_mask], test_presabs_tailprob_summaries[best_scoring_method][test_absence_mask]))
        test_performances_atthresh = map(x -> round(x, digits=5), performance_statistics(test_presabs_tailprob_summaries[best_scoring_atthresh][test_presence_mask], test_presabs_tailprob_summaries[best_scoring_atthresh][test_absence_mask], threshold=best_thresh))

        println("FlowingClusters chain performances")
        println("    Best without threshold: $best_scoring_method, $perfstat=$(getindex(test_performances, perfstat))")
        println("       Best with threshold: $best_scoring_atthresh, $perfstat=$(getindex(test_performances_atthresh, perfstat))")
        return (
            fc_MAP=(;
                nb_clusters=map_nb_clusters,
                test_performances=test_map_performances,
                test_performances_atthresh=test_map_performances_atthresh,
                best_thresh=best_map_thresh
            ),
            fc_chain=(;
                test_performances,
                test_performances_atthresh,
                best_thresh,
                best_scoring_method,
                best_scoring_atthresh
                )
            )
    else
        @info "Chain has less than 3 samples, can't evaluate its performance yet"
        return (
            fc_MAP=(;
                nb_clusters=map_nb_clusters,
                test_map_performances,
                test_map_performances_atthresh,
                best_map_thresh
            ),
            fc_chain=nothing
        )
    end
end

# fc_perfs = evaluate_flowingclusters(chain, dataset, species, predictors, perfstat, nb_rejection_samples=50_000);
