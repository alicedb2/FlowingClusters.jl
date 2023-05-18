push!(LOAD_PATH, "/Users/alice/Documents/Postdoc/MultivariateNormalCRP")

using MultivariateNormalCRP
using SpeciesDistributionToolkit
using CSV
using DataFrames
using JLD2
using CodecBzip2
using Random
using EvoTrees

species, ebird_csv, output_prefix = ARGS[1], ARGS[2], ARGS[3]
nb_iter_burn, nb_iter, sample_every = parse.(Int64, [ARGS[4], ARGS[5], ARGS[6]])

bioclim_layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])
landcover_layernames = collect(keys(layerdescriptions(RasterData(EarthEnv, LandCover))))
deleteat!(landcover_layernames, 5) # Remove open water to remove null direction

layernames = vcat(bioclim_layernames, landcover_layernames)


ebird_df = DataFrame(CSV.File(joinpath(pwd(), ebird_csv), delim="\t"))

ebird_pres_dataset = MNCRPDataset(subset(ebird_df, species => x -> x .== 1.0), layernames)
ebird_abs_dataset = MNCRPDataset(subset(ebird_df, species => x -> x .== 0.0), layernames)

Random.seed!(7651)
train_pres, valid_pres, test_pres = split(ebird_pres_dataset, 3)
train_abs, valid_abs, test_abs = split(ebird_abs_dataset, 3)

standardize!(train_pres)
standardize!(valid_pres, with=train_pres)
standardize!(test_pres, with=train_pres)
standardize!(train_abs, with=train_pres)
standardize!(valid_abs, with=train_pres)
standardize!(test_abs, with=train_pres)

presence_chain = MNCRPChain(train_pres, chain_samples=100)
advance_chain!(presence_chain, nb_iter_burn; nb_splitmerge=200, 
               sample_every=nothing, attempt_map=false, 
               pretty_progress=false)
advance_chain!(presence_chain, nb_iter; nb_splitmerge=200, 
               sample_every=sample_every, attempt_map=true, 
               pretty_progress=false)


JLD2.jldsave(joinpath(output_prefix, "$(species)_presence_chain.jld2"), Bzip2Compressor(blocksize100k=9); chain=presence_chain)

absence_chain = MNCRPChain(train_abs, chain_samples=100)
advance_chain!(absence_chain, nb_iter_burn; nb_splitmerge=200, 
               sample_every=nothing, attempt_map=false, 
               pretty_progress=false)
advance_chain!(absence_chain, nb_iter; nb_splitmerge=200, 
               sample_every=sample_every, attempt_map=true, 
               pretty_progress=false)

JLD2.jldsave(joinpath(output_prefix, "$(species)_absence_chain.jld2"), Bzip2Compressor(blocksize100k=9); chain=absence_chain)


thresholds = 0.001:0.001:0.999;

# Tail MAP
maptail_validation_pres_predictions = tail_probability(valid_pres.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_validation_abs_predictions = tail_probability(valid_abs.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
best_maptail_J = -Inf;
best_maptail_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(maptail_validation_pres_predictions, maptail_validation_abs_predictions, threshold=thresh);
    if perfscores.J > best_maptail_J
        best_maptail_J = perfscores.J
        best_maptail_thresh = thresh
    end
end
maptail_test_pres_predictions = tail_probability(test_pres.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_test_abs_predictions = tail_probability(test_abs.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_test_scores = performance_scores(maptail_test_pres_predictions, maptail_test_abs_predictions);
maptail_test_scores_atbestJ = performance_scores(maptail_test_pres_predictions, maptail_test_abs_predictions, threshold=best_maptail_thresh);
println("Tail MAP results")
println(map(x -> round(x, digits=2), maptail_test_scores))
println(map(x -> round(x, digits=2), maptail_test_scores_atbestJ))
println()

# P-class MAP
mappclass_validation_pres_predictions = presence_probability(valid_pres.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_validation_abs_predictions = presence_probability(valid_abs.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
best_mappclass_J = -Inf;
best_mappclass_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(mappclass_validation_pres_predictions, mappclass_validation_abs_predictions, threshold=thresh);
    if perfscores.J > best_mappclass_J
        best_mappclass_J = perfscores.J
        best_mappclass_thresh = thresh
    end
end
mappclass_test_pres_predictions = presence_probability(test_pres.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_abs_predictions = presence_probability(test_abs.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_scores = performance_scores(mappclass_test_pres_predictions, mappclass_test_abs_predictions);
mappclass_test_scores_atbestJ = performance_scores(mappclass_test_pres_predictions, mappclass_test_abs_predictions, threshold=best_mappclass_thresh);
println("P-class MAP results")
println(map(x -> round(x, digits=2), mappclass_test_scores))
println(map(x -> round(x, digits=2), mappclass_test_scores_atbestJ))
println()

# Tail median
mediantail_validation_pres_predictions = tail_probability_summary(valid_pres.data, presence_chain).median;
mediantail_validation_abs_predictions = tail_probability_summary(valid_abs.data, presence_chain).median;
best_mediantail_J = -Inf;
best_mediantail_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(mediantail_validation_pres_predictions, mediantail_validation_abs_predictions, threshold=thresh);
    if perfscores.J > best_mediantail_J
        best_mediantail_J = perfscores.J
        best_mediantail_thresh = thresh
    end
end
mediantail_test_pres_predictions = tail_probability_summary(test_pres.data, presence_chain).median;
mediantail_test_abs_predictions = tail_probability_summary(test_abs.data, presence_chain).median;
mediantail_test_scores = performance_scores(mediantail_test_pres_predictions, mediantail_test_abs_predictions);
mediantail_test_scores_atbestJ = performance_scores(mediantail_test_pres_predictions, mediantail_test_abs_predictions, threshold=best_mediantail_thresh);
println("Tail median results")
println(map(x -> round(x, digits=2), mediantail_test_scores))
println(map(x -> round(x, digits=2), mediantail_test_scores_atbestJ))
println()

# P-class median
medianpclass_validation_pres_predictions = presence_probability_summary(valid_pres.data, presence_chain, absence_chain).median;
medianpclass_validation_abs_predictions = presence_probability_summary(valid_abs.data, presence_chain, absence_chain).median;
best_medianpclass_J = -Inf;
best_medianpclass_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(medianpclass_validation_pres_predictions, medianpclass_validation_abs_predictions, threshold=thresh);
    if perfscores.J > best_medianpclass_J
        best_medianpclass_J = perfscores.J
        best_medianpclass_thresh = thresh
    end
end
medianpclass_test_pres_predictions = presence_probability_summary(test_pres.data, presence_chain, absence_chain).median;
medianpclass_test_abs_predictions = presence_probability_summary(test_abs.data, presence_chain, absence_chain).median;
medianpclass_test_scores = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_abs_predictions);
medianpclass_test_scores_atbestJ = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_abs_predictions, threshold=best_medianpclass_thresh);
println("P-class median results")
println(map(x -> round(x, digits=2), medianpclass_test_scores))
println(map(x -> round(x, digits=2), medianpclass_test_scores_atbestJ))
println()


#### BRT

gaussian_tree_parameters = EvoTreeGaussian(;
    loss=:gaussian,
    metric=:gaussian,
    nrounds=100,
    nbins=100,
    λ=0.0,
    γ=0.0,
    η=0.1,
    max_depth=7,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
);


brt_train_pres_predictors = collect(reduce(hcat, train_pres.data)');
brt_train_abs_predictors = collect(reduce(hcat, train_abs.data)');
brt_train_X = vcat(brt_train_pres_predictors, brt_train_abs_predictors);
brt_train_y = vcat(fill(1.0, size(brt_train_pres_predictors, 1)), fill(0.0, size(brt_train_abs_predictors, 1)));

brt_validation_pres_predictors = collect(reduce(hcat, valid_pres.data)');
brt_valid_abs_predictors = collect(reduce(hcat, valid_abs.data)');
brt_validation_X = vcat(brt_validation_pres_predictors, brt_valid_abs_predictors);
brt_validation_y = vcat(fill(1.0, size(brt_validation_pres_predictors, 1)), fill(0.0, size(brt_valid_abs_predictors, 1)));

brt_model = fit_evotree(gaussian_tree_parameters; x_train=brt_train_X, y_train=brt_train_y, x_eval=brt_validation_X, y_eval=brt_validation_y)

brt_validation_predictions = getindex.(eachrow(EvoTrees.predict(brt_model, brt_validation_X)), 1);
brt_validation_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_validation_predictions);
brt_validation_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_validation_predictions);
brt_validation_pres_pedictions = brt_validation_predictions[1:size(brt_validation_pres_predictors, 1)];
brt_validation_abs_predictions = brt_validation_predictions[size(brt_validation_pres_predictors, 1)+1:end];

best_brt_J = -Inf;
best_brt_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(brt_validation_pres_pedictions, brt_validation_abs_predictions, threshold=thresh);
    if perfscores.J > best_brt_J
        best_brt_J = perfscores.J
        best_brt_thresh = thresh
    end
end

brt_test_pres_predictors = collect(reduce(hcat, test_pres.data)');
brt_test_abs_predictors = collect(reduce(hcat, test_abs.data)');
brt_test_X = vcat(brt_test_pres_predictors, brt_test_abs_predictors);
brt_test_y = vcat(fill(1.0, size(brt_test_pres_predictors, 1)), fill(0.0, size(brt_test_abs_predictors, 1)));
brt_test_predictions = EvoTrees.predict(brt_model, brt_test_X)[:, 1];
brt_test_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_test_predictions);
brt_test_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_test_predictions);

brt_test_pres_predictions = brt_test_predictions[1:size(brt_test_pres_predictors, 1)];
brt_test_abs_predictions = brt_test_predictions[size(brt_test_pres_predictors, 1)+1:end];

brt_test_scores = performance_scores(brt_test_pres_predictions, brt_test_abs_predictions);
brt_test_scores_atbestJ = performance_scores(brt_test_pres_predictions, brt_test_abs_predictions, threshold=best_brt_thresh);
println("BRT results")
println(map(x -> round(x, digits=2), brt_test_scores))
println(map(x -> round(x, digits=2), brt_test_scores_atbestJ))
println()


JLD2.jldsave(joinpath(output_prefix, "$(species)_SDM_results.jld2");
brt_test_scores=brt_test_scores, 
brt_test_scores_atbestJ=brt_test_scores_atbestJ,
medianpclass_test_scores=medianpclass_test_scores, 
medianpclass_test_scores_atbestJ=medianpclass_test_scores_atbestJ,
mediantail_test_scores=mediantail_test_scores, 
mediantail_test_scores_atbestJ=mediantail_test_scores_atbestJ,
mappclass_test_scores=mappclass_test_scores, 
mappclass_test_scores_atbestJ=mappclass_test_scores_atbestJ,
maptail_test_scores=maptail_test_scores, 
maptail_test_scores_atbestJ=maptail_test_scores_atbestJ
)