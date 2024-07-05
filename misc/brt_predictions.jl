using EvoTrees

gaussian_tree_parameters = EvoTreeGaussian(; loss=:gaussian, metric=:gaussian, nrounds=100, nbins=100, λ=0.0, γ=0.0, η=0.1, max_depth=7, min_weight=1.0, rowsample=0.5, colsample=1.0)

brt_train_presences = collect(reduce(hcat, train_presences.data)')
brt_train_absences = collect(reduce(hcat, train_absences.data)')
brt_train_X = vcat(brt_train_presences, brt_train_absences)
brt_train_y = vcat(fill(1.0, size(brt_train_presences, 1)), fill(0.0, size(brt_train_absences, 1)))
brt_validation_presences = collect(reduce(hcat, validation_presences.data)')
brt_validation_absences = collect(reduce(hcat, validation_absences.data)')
brt_validation_X = vcat(brt_validation_presences, brt_validation_absences)
brt_validation_y = vcat(fill(1.0, size(brt_validation_presences, 1)), fill(0.0, size(brt_validation_absences, 1)))
brt_test_presences = collect(reduce(hcat, test_presences.data)')
brt_test_absences = collect(reduce(hcat, test_absences.data)')
brt_test_X = vcat(brt_test_presences, brt_test_absences)
brt_test_y = vcat(fill(1.0, size(brt_test_presences, 1)), fill(0.0, size(brt_test_absences, 1)))

brt_model = fit_evotree(gaussian_tree_parameters; x_train=brt_train_X, y_train=brt_train_y, x_eval=brt_validation_X, y_eval=brt_validation_y)

brt_validation_predictions = min.(1, max.(0, getindex.(eachrow(EvoTrees.predict(brt_model, brt_validation_X)), 1)));
brt_validation_presence_predictions = brt_validation_predictions[1:size(brt_validation_presences, 1)]
brt_validation_absence_predictions = brt_validation_predictions[size(brt_validation_presences, 1)+1:end]


best_brt_thresh = best_score_threshold(brt_validation_presence_predictions, brt_validation_absence_predictions, statistic=perfstat)

brt_test_predictions = min.(1, max.(0, EvoTrees.predict(brt_model, brt_test_X)[:, 1]));
brt_test_presence_predictions = brt_test_predictions[1:size(brt_test_presences, 1)];
brt_test_absence_predictions = brt_test_predictions[size(brt_test_presences, 1)+1:end];

brt_test_statistics = map(x -> round(x, digits=3), performance_statistics(brt_test_presence_predictions, brt_test_absence_predictions));
brt_test_statistics_atbestthresh = map(x -> round(x, digits=3), performance_statistics(brt_test_presence_predictions, brt_test_absence_predictions, threshold=best_brt_thresh));

println("#####################")
println("BRT stats:\n$brt_test_statistics")
println("BRT stats at best tresh:\n$brt_test_statistics_atbestthresh")

