## FlowingClusters without FFJORD

_maptpfun = tail_probability(_presence_chain.map_clusters, _presence_chain.map_hyperparams)
_summfun = tail_probability_summary(_presence_chain.clusters_samples, _presence_chain.hyperparams_samples)

_validation_map_presence_tailprobs = _maptpfun(reduce(hcat, validation_presences.data))
_validation_map_absence_tailprobs = _maptpfun(reduce(hcat, validation_absences.data))
_test_map_presence_tailprobs = _maptpfun(reduce(hcat, test_presences.data))
_test_map_absence_tailprobs = _maptpfun(reduce(hcat, test_absences.data))

_validation_presence_summaries = _summfun(reduce(hcat, validation_presences.data))
_validation_absence_summaries = _summfun(reduce(hcat, validation_absences.data))
_test_presence_summaries = _summfun(reduce(hcat, test_presences.data))
_test_absence_summaries = _summfun(reduce(hcat, test_absences.data))

_validation_presence_tailprobs = _validation_presence_summaries[tailprobscore]
_validation_absence_tailprobs = _validation_absence_summaries[tailprobscore]
_test_presence_tailprobs = _test_presence_summaries[tailprobscore]
_test_absence_tailprobs = _test_absence_summaries[tailprobscore]


###############
tailprobscore = :modefd
perfstat = :J
###############

_best_map_thresh = best_score_threshold(_validation_map_presence_tailprobs, _validation_map_absence_tailprobs, statistic=perfstat)
_test_map_statistics = map(x -> round(x, digits=3), performance_statistics(_test_map_presence_tailprobs, _test_map_absence_tailprobs))
_test_map_statistics_atbesttresh = map(x -> round(x, digits=3), performance_statistics(_test_map_presence_tailprobs, _test_map_absence_tailprobs, threshold=_best_map_thresh))

_best_thresh = best_score_threshold(_validation_presence_tailprobs, _validation_absence_tailprobs, statistic=perfstat)
_test_statistics = map(x -> round(x, digits=3), performance_statistics(_test_presence_tailprobs, _test_absence_tailprobs))
_test_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(_test_presence_tailprobs, _test_absence_tailprobs, threshold=_best_thresh))

println("#####################")
println("   FC MAP stats:\n$_test_map_statistics")
println("   FC MAP stats at best tresh:\n$_test_statistics_atbestthresh")
println("   FC stats:\n$_test_statistics")
println("   FC stats at best tresh:\n$_test_statistics_atbestthresh")
println("#####################")



### FlowingClusters with FFJORD

maptpfun = tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams)
summfun = tail_probability_summary(presence_chain.clusters_samples, presence_chain.hyperparams_samples)

validation_map_presence_tailprobs = maptpfun(reduce(hcat, validation_presences.data))
validation_map_absence_tailprobs = maptpfun(reduce(hcat, validation_absences.data))
test_map_presence_tailprobs = maptpfun(reduce(hcat, test_presences.data))
test_map_absence_tailprobs = maptpfun(reduce(hcat, test_absences.data))

validation_presence_summaries = summfun(reduce(hcat, validation_presences.data))
validation_absence_summaries = summfun(reduce(hcat, validation_absences.data))
test_presence_summaries = summfun(reduce(hcat, test_presences.data))
test_absence_summaries = summfun(reduce(hcat, test_absences.data))

tailprobscore = :modefd
validation_presence_tailprobs = validation_presence_summaries[tailprobscore]
validation_absence_tailprobs = validation_absence_summaries[tailprobscore]
test_presence_tailprobs = test_presence_summaries[tailprobscore]
test_absence_tailprobs = test_absence_summaries[tailprobscore]

best_map_thresh = best_score_threshold(validation_map_presence_tailprobs, validation_map_absence_tailprobs, statistic=perfstat)

test_map_statistics = map(x -> round(x, digits=3), performance_statistics(test_map_presence_tailprobs, test_map_absence_tailprobs))
test_map_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(test_map_presence_tailprobs, test_map_absence_tailprobs, threshold=best_map_thresh))

println("########################")
println("FC FFJORD MAP stats:\n$test_map_statistics")
println("FC FFJORD MAP stats at best tresh:\n$test_map_statistics_atbestthresh")

best_thresh = best_score_threshold(validation_presence_tailprobs, validation_absence_tailprobs, statistic=perfstat)
test_statistics = map(x -> round(x, digits=3), performance_statistics(test_presence_tailprobs, test_absence_tailprobs))
test_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(test_presence_tailprobs, test_absence_tailprobs, threshold=best_thresh))

println("FC stats:\n$test_statistics")
println("FC stats at best tresh:\n$test_statistics_atbestthresh")
