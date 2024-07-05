

bioclim = bioclim_predictor(eb.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12))

bioclim_validation_presence_scores = bioclim(eb.validation.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12))
bioclim_validation_absence_scores = bioclim(eb.validation.absence(:sp5).standardize(:BIO1, :BIO3, :BIO12))
bioclim_test_presence_scores = bioclim(eb.test.presence(:sp5).standardize(:BIO1, :BIO3, :BIO12))
bioclim_test_absence_scores = bioclim(eb.test.absence(:sp5).standardize(:BIO1, :BIO3, :BIO12))

perfstat = :MCC
best_bioclim_thresh = best_score_threshold(bioclim_validation_presence_scores, bioclim_validation_absence_scores, statistic=perfstat)

bioclim_test_statistics = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores))
bioclim_test_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(bioclim_test_presence_scores, bioclim_test_absence_scores, threshold=best_bioclim_thresh))
println("BIOCLIM stats:\n$bioclim_test_statistics")
println("BIOCLIM stats at best tresh:\n$bioclim_test_statistics_atbestthresh")

