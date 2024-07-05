bioclim = bioclim_predictor(reduce(hcat, train_presences.data))

bioclim_validation_presence_scores = bioclim(reduce(hcat, validation_presences.data))
bioclim_validation_absence_scores = bioclim(reduce(hcat, validation_absences.data))
bioclim_test_presence_scores = bioclim(reduce(hcat, test_presences.data))
bioclim_test_absence_scores = bioclim(reduce(hcat, test_absences.data))

best_bioclim_thresh = best_score_threshold(bioclim_validation_presence_scores, bioclim_validation_absence_scores, statistic=perfstat)

bioclim_test_statistics = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores))
bioclim_test_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores, threshold=best_bioclim_thresh))
println("BIOCLIM stats:\n$bioclim_test_statistics")
println("BIOCLIM stats at best tresh:\n$bioclim_test_statistics_atbestthresh")

