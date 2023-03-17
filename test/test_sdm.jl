using Pkg
Pkg.activate(".")
using Revise
using SpeciesDistributionToolkit
using CSV
using DataFrames
using StatsBase
using LinearAlgebra
using ColorSchemes
using Random
# using StatsPlots
using Plots
using EvoTrees
using MultivariateNormalCRP
import GeoMakie
import CairoMakie
include("src/helpers.jl")

# layernames = "BIO" .* string.([1, 3, 12])
layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])
all_layernames = "BIO" .* string.(1:19)
_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), layer=layer, resolution=10.0) 
            for layer in layernames]
all_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), layer=layer, resolution=10.0) 
              for layer in all_layernames]


dataset_all = MNCRPDataset("data/Urocyon_cinereoargenteus.csv", _layers, layernames=layernames)
dataset = MNCRPDataset("data/Urocyon_cinereoargenteus.csv", _layers, layernames=layernames)

train_dataset, validation_dataset, test_dataset = split(dataset, 3, standardize_with_first=true)

train_df = unique(train_dataset.dataframe, train_dataset.layernames)
validation_df = unique(validation_dataset.dataframe, validation_dataset.layernames)
test_df = unique(test_dataset.dataframe, test_dataset.layernames)

train_psabs = generate_pseudoabsences(train_df, train_dataset.layers, method=SurfaceRangeEnvelope);
validation_psabs = generate_pseudoabsences(validation_df, validation_dataset.layers, method=SurfaceRangeEnvelope);
test_psabs = generate_pseudoabsences(test_df, test_dataset.layers, method=SurfaceRangeEnvelope);

presence_chain = initiate_chain(train_dataset, chain_samples=100)
advance_chain!(presence_chain, 200; nb_splitmerge=300, 
               mh_stepscale=[0.6, 0.32, 0.32, 0.15, 0.55], 
               sample_every=10, attempt_map=false)
dump(presence_chain.hyperparams.diagnostics)
clear_diagnostics!(presence_chain.hyperparams.diagnostics)

absence_chain = initiate_chain(standardize_with(train_psabs.predictors, train_dataset), 
                               standardize=false, chain_samples=100)
advance_chain!(absence_chain, 100; nb_splitmerge=300, 
               mh_stepscale=[0.6, 0.4, 0.4, 0.15, 0.55], 
               sample_every=50, attempt_map=false)


left, right, bottom, top = -140.0, -51.0, -4.0, 60.0
gat(f, t) = GeoMakie.GeoAxis(f; 
            dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
            coastlines=true,
            lonlims=(left, right), latlims=(bottom, top),
            title=t
);

p = clip(_layers[1], left=left, right=right, bottom=bottom, top=top)
northam_longs, northam_lats = 1.0 .* eachrow(reduce(hcat, keys(p)))
northam_predictors = [Float64[p[lo, la] for p in _layers] for (lo, la) in zip(northam_longs, northam_lats)]
northam_predictors = [x + 1e-8 * rand(length(x)) for x in northam_predictors]
# northam_standardized_predictors = standardize_with(northam_predictors, train_dataset)
northam_standardized_predictors = standardize_with(northam_predictors, dataset)

####### North-America tail median figure

northam_tail_summary = tail_probability_summary(northam_standardized_predictors, presence_chain)
northam_pclass_summary = presence_probability_summary(northam_standardized_predictors, presence_chain, absence_chain)


# fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
# ga = gat(fig[1, 1], "Median tail probability (6D)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, getindex.(summ_northam_tail_ps, :median), shading=false, colorrange=(0, 1))
# # GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
# # GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
# GeoMakie.Colorbar(fig[1, 2], summ_surf)

# ga = gat(fig[2, 1], "Median presence class probability (6D)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, getindex.(summ_northam_ps, :median), shading=false, colorrange=(0, 1))
# # GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
# # GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
# GeoMakie.Colorbar(fig[2, 2], summ_surf)

# display(fig)


#### BIOCLIM

bcp = bioclim_predictor(train_dataset.data)

bc_northam_scores = bcp(northam_standardized_predictors)

fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
        dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
        lonlims=(-140, -51), latlims=(-4, 60),
        title="BIOCLIM at threshold=0.2"
);
bioclim_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, bioclim_northam_scores, shading=false)#, colorrange=(0, 1));
# bioclim_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, bioclim_score .>= 0.05, shading=false, colorrange=(0, 1));
# GeoMakie.scatter!(ga, train_pas.longitudes, train_pas.latitudes, strokewidth=0, markersize=3, color=:gray5);
# GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3);
GeoMakie.Colorbar(fig[1, 2], bioclim_surf)
display(fig)

bc_validation_pres_redictions = bcp(validation_dataset.data)
bc_validation_psabs_predictions = bcp(standardize_with(validation_psabs.predictors, train_dataset))
best_bc_J = -Inf;
best_bc_thresh = 0.0;
thresholds = 0.01:01:0.99;
for thresh in thresholds
    perfscores = performance_scores(bc_validation_pres_redictions, bc_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_bc_J
        best_bc_J = perfscores.J
        best_bc_thresh = thresh
    end
end
bc_test_pres_redictions = bcp(test_dataset.data);
bc_test_psabs_predictions = bcp(standardize_with(test_psabs.predictors, train_dataset))
bc_test_scores = performance_scores(bc_test_pres_redictions, bc_test_psabs_predictions);
bc_test_scores_atbestJ = performance_scores(bc_test_pres_redictions, bc_test_psabs_predictions, threshold=best_bc_thresh);
map(x -> round(x, digits=2), bc_test_scores)
map(x -> round(x, digits=2), bc_test_scores_atbestJ)



fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "BIOCLIM score")
bc_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, bc_northam_scores, shading=false, colorrange=(0, 1), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, flatten=true), latitudes(test_dataset, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[1, 2], bc_surf)

ga = gat(fig[2, 1], "BIOCLIM threshold")
bc_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, bc_northam_scores .> best_bc_thresh, shading=false, colorrange=(-0.3, 1.3), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, flatten=true), latitudes(test_dataset, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[2, 2], bc_surf)
display(fig)


thresholds = 0.01:0.01:0.99;

# #### MNCRP tail MAP
best_maptail_J = -Inf;
best_maptail_thresh = 0.0;
maptail_validation_pres_predictions = tail_probability(validation_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_validation_psabs_predictions = tail_probability(standardize_with(validation_psabs.predictors, train_dataset), presence_chain.map_clusters, presence_chain.map_hyperparams);
for thresh in thresholds
    perfscores = performance_scores(maptail_validation_pres_predictions, maptail_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_maptail_J
        best_maptail_J = perfscores.J
        best_maptail_thresh = thresh
    end
end
maptail_test_pres_predictions = tail_probability(test_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_test_psabs_predictions = tail_probability(standardize_with(test_psabs.predictors, train_dataset), presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_test_scores = performance_scores(maptail_test_pres_predictions, maptail_test_psabs_predictions);
maptail_test_scores_atbestJ = performance_scores(maptail_test_pres_predictions, maptail_test_psabs_predictions, threshold=best_maptail_thresh);
map(x -> round(x, digits=2), maptail_test_scores)
map(x -> round(x, digits=2), maptail_test_scores_atbestJ)


####### North-America train MAP figure

map_tail = tail_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams)
map_pclass = presence_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams)

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP tail probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_tail, shading=false, colorrange=(0, 1))
# GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], map_surf)

ga = gat(fig[2, 1], "MAP tail threshold")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_tail .>= best_maptail_thresh, shading=false, colorrange=(0, 1))
# GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[2, 2], map_surf)
display(fig)


fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP presence class probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_pclass, shading=false, colorrange=(0, 1))
# GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], map_surf)

ga = gat(fig[2, 1], "MAP presence class threshold")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_pclass .>= best_mappclass_thresh, shading=false, colorrange=(0, 1))
# GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[2, 2], map_surf)
display(fig)


##### MNCRP pclass MAP
best_mappclass_J = -Inf;
best_mappclass_thresh = 0.0;
mappclass_validation_pres_predictions = presence_probability(validation_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_validation_psabs_predictions = presence_probability(standardize_with(validation_psabs.predictors, train_dataset), presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
for thresh in thresholds
    perfscores = performance_scores(mappclass_validation_pres_predictions, mappclass_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_mappclass_J
        best_mappclass_J = perfscores.J
        best_mappclass_thresh = thresh
    end
end
mappclass_test_pres_predictions = presence_probability(test_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_psabs_predictions = presence_probability(standardize_with(test_psabs.predictors, train_dataset), presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_scores = performance_scores(mappclass_test_pres_predictions, mappclass_test_psabs_predictions);
mappclass_test_scores_atbestJ = performance_scores(mappclass_test_pres_predictions, mappclass_test_psabs_predictions, threshold=best_mappclass_thresh);
map(x -> round(x, digits=2), mappclass_test_scores)
map(x -> round(x, digits=2), mappclass_test_scores_atbestJ)



#### MAP cluster assignements
map_tail = tail_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams)
map_idx = map_cluster_assignment_idx.(northam_standardized_predictors, Ref(presence_chain.map_clusters), Ref(presence_chain.map_hyperparams))
idx_mask = map_idx .<= 19
map_idx[.!idx_mask] .= 0


fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP tail probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_tail, shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], map_surf)

# ga = gat(fig[2, 1], "MAP cluster assignment (first 19 and others)")
# map_surf = GeoMakie.surface!(ga, 
# northam_longs, northam_lats, map_idx, 
# shading=false, 
# colormap=cgrad(:tab20, 20, categorical=true), colorrange=(-0.5, 19.5))
# GeoMakie.Colorbar(fig[2, 2], map_surf)
# display(fig)
ga = gat(fig[2, 1], "MAP cluster assignment")
map_surf = GeoMakie.surface!(ga, 
northam_longs, northam_lats, map_idx, 
shading=false, 
colormap=:glasbey_bw_minc_20_hue_150_280_n256)
GeoMakie.Colorbar(fig[2, 2], map_surf)
display(fig)


map_local_geometry = local_geometry.(northam_standardized_predictors, Ref(presence_chain.map_clusters), Ref(presence_chain.map_hyperparams))
map_local_entropy = getproperty.(map_local_geometry, :entropy)
map_local_widest = [first(first(geom.importances).sorted_evecs)[2] for geom in map_local_geometry]

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP widest predictor")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, 
map_local_widest,
shading=false, colorrange=(0.5, 3.5), colormap=cgrad(:Spectral_4, 3, categorical=true))
GeoMakie.Colorbar(fig[1, 2], map_surf, ticks=[1, 2, 3])

ga = gat(fig[2, 1], "MAP local entropy")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, 
map_local_entropy, shading=false, colorrange=(0, log(3)))
GeoMakie.Colorbar(fig[2, 2], map_surf)

display(fig)

#### MNCRP tail median
mediantail_validation_pres_predictions = tail_probability_summary(validation_dataset.data, presence_chain).median
mediantail_validation_psabs_predictions = tail_probability_summary(standardize_with(validation_psabs.predictors, train_dataset), presence_chain).median
thresholds = 0.01:0.01:0.99;
best_mediantail_J = -Inf;
best_mediantail_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(mediantail_validation_pres_predictions, mediantail_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_mediantail_J
        best_mediantail_J = perfscores.J
        best_mediantail_thresh = thresh
    end
end
mediantail_test_pres_predictions = tail_probability_summary(test_dataset.data, presence_chain).median
mediantail_test_psabs_predictions = tail_probability_summary(standardize_with(test_psabs.predictors, train_dataset), presence_chain).median
mediantail_test_scores = performance_scores(mediantail_test_pres_predictions, mediantail_test_psabs_predictions);
mediantail_test_scores_atbestJ = performance_scores(mediantail_test_pres_predictions, mediantail_test_psabs_predictions, threshold=best_mediantail_thresh);
map(x -> round(x, digits=2), mediantail_test_scores)
map(x -> round(x, digits=2), mediantail_test_scores_atbestJ)



fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Median tail probability")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median, shading=false, colormap=:acton, colorrange=(0, 1))
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "Median tail threshold")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median .>= best_mediantail_thresh, shading=false, colormap=:acton, colorrange=(-0.3, 1.3))
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[2, 2], summ_surf)
display(fig)




fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Median tail presence predictions (true positives/false negative)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median .>= best_mediantail_thresh, shading=false, colormap=:acton, colorrange=(-0.1, 1.1))
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      color=mediantail_test_pres_predictions .>= best_mediantail_thresh,
                      strokewidth=0, markersize=6, colormap=cgrad(:bwr, rev=true));
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "Median tail absence predictions (true negatives/false positives)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median .>= best_mediantail_thresh, shading=false, colormap=:acton, colorrange=(-0.1, 1.1))
GeoMakie.scatter!(ga, test_psabs.longitudes, 
                      test_psabs.latitudes, 
                      color=mediantail_test_psabs_predictions .<= best_mediantail_thresh,
                      strokewidth=0, markersize=6, colormap=cgrad(:bwr, rev=true));
GeoMakie.Colorbar(fig[2, 2], summ_surf)
display(fig)


# fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
# ga = gat(fig[1, 1], "Median tail false negatives (6D)")
# GeoMakie.scatter!(ga, longitudes(test_dataset, flatten=true, unique=true), 
#                       latitudes(test_dataset, flatten=true, unique=true),
#                       color=median_test_p_predictions .<= best_median_thresh,
#                       colormap=:Greys, strokewidth=0, markersize=3);
# # GeoMakie.Colorbar(fig[1, 2], summ_surf)
# ga = gat(fig[2, 1], "Median tail false positives (6D)")
# GeoMakie.scatter!(ga, test_psabs.longitudes,
#                       test_psabs.latitudes,
#                       color=median_test_psabs_predictions .>= best_median_thresh,
#                       colormap=:Greys, strokewidth=0, markersize=3);
# # GeoMakie.Colorbar(fig[2, 2], summ_surf)
# display(fig)


#### MNCRP presence class median
medianpclass_validation_pres_predictions = presence_probability_summary(validation_dataset.data, presence_chain, absence_chain).median
medianpclass_validation_psabs_predictions = presence_probability_summary(standardize_with(validation_psabs.predictors, train_dataset), presence_chain, absence_chain).median
thresholds = 0.01:0.01:0.99;
best_medianpclass_J = -Inf;
best_medianpclass_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(medianpclass_validation_pres_predictions, medianpclass_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_medianpclass_J
        best_medianpclass_J = perfscores.J
        best_medianpclass_thresh = thresh
    end
end
medianpclass_test_pres_predictions = presence_probability_summary(test_dataset.data, presence_chain, absence_chain).median
medianpclass_test_psabs_predictions = presence_probability_summary(standardize_with(test_psabs.predictors, train_dataset), presence_chain, absence_chain).median
medianpclass_test_scores = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_psabs_predictions);
medianpclass_test_scores_atbestJ = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_psabs_predictions, threshold=best_medianpclass_thresh);
map(x -> round(x, digits=2), medianpclass_test_scores)
map(x -> round(x, digits=2), medianpclass_test_scores_atbestJ)



fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Median presence class probability")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_pclass_summary.median, shading=false, colorrange=(0, 1), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, flatten=true), latitudes(test_dataset, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "Median presence class threshold")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_pclass_summary.median .>= best_medianpclass_thresh, shading=false, colorrange=(-0.3, 1.3), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, flatten=true), latitudes(test_dataset, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[2, 2], summ_surf)

display(fig)



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


brt_train_pres_predictors = collect(reduce(hcat, train_dataset.data)')
brt_train_psabs_predictors = collect(reduce(hcat, standardize_with(train_psabs.predictors, train_dataset))')
brt_train_X = vcat(brt_train_pres_predictors, brt_train_psabs_predictors)
brt_train_y = vcat(fill(1.0, size(brt_train_pres_predictors, 1)), fill(0.0, size(brt_train_psabs_predictors, 1)))

brt_validation_pres_predictors = collect(reduce(hcat, validation_dataset.data)')
brt_validation_psabs_predictors = collect(reduce(hcat, standardize_with(validation_psabs.predictors, train_dataset))')
brt_validation_X = vcat(brt_validation_pres_predictors, brt_validation_psabs_predictors)
brt_validation_y = vcat(fill(1.0, size(brt_validation_pres_predictors, 1)), fill(0.0, size(brt_validation_psabs_predictors, 1)))

brt_model = fit_evotree(gaussian_tree_parameters; x_train=brt_train_X, y_train=brt_train_y, x_eval=brt_validation_X, y_eval=brt_validation_y)

brt_validation_predictions = getindex.(eachrow(EvoTrees.predict(brt_model, brt_validation_X)), 1);
brt_validation_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_validation_predictions);
brt_validation_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_validation_predictions);
brt_validation_pres_pedictions = brt_validation_predictions[1:size(brt_validation_pres_predictors, 1)]
brt_validation_psabs_predictions = brt_validation_predictions[size(brt_validation_pres_predictors, 1)+1:end]
thresholds = 0.01:0.01:0.99;
best_brt_J = -Inf;
best_brt_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(brt_validation_pres_pedictions, brt_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_brt_J
        best_brt_J = perfscores.J
        best_brt_thresh = thresh
    end
end

brt_test_pres_predictors = collect(reduce(hcat, test_dataset.data)')
brt_test_psabs_predictors = collect(reduce(hcat, standardize_with(test_psabs.predictors, train_dataset))')
brt_test_X = vcat(brt_test_pres_predictors, brt_test_psabs_predictors)
brt_test_y = vcat(fill(1.0, size(brt_test_pres_predictors, 1)), fill(0.0, size(brt_test_psabs_predictors, 1)))
brt_test_predictions = EvoTrees.predict(brt_model, brt_test_X)[:, 1]
brt_test_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_test_predictions);
brt_test_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_test_predictions);

brt_test_pres_predictions = brt_test_predictions[1:size(brt_test_pres_predictors, 1)]
brt_test_psabs_predictions = brt_test_predictions[size(brt_test_pres_predictors, 1)+1:end]

brt_test_scores = performance_scores(brt_test_pres_predictions, brt_test_psabs_predictions);
brt_test_scores_atbestJ = performance_scores(brt_test_pres_predictions, brt_test_psabs_predictions, threshold=best_brt_thresh);
map(x -> round(x, digits=2), brt_test_scores)
map(x -> round(x, digits=2), brt_test_scores_atbestJ)




brt_northam_predictions = getindex.(eachrow(EvoTrees.predict(brt_model, reduce(hcat, northam_standardized_predictors)')), 1)
brt_northam_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_northam_predictions);
brt_northam_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_northam_predictions);


fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "BRT probability")
brt_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions, shading=false, colorrange=(0, 1), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[1, 2], brt_surf)

ga = gat(fig[2, 1], "BRT threshold")
brt_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions .> best_brt_thresh, shading=false, colorrange=(-0.3, 1.3), colormap=:acton)
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=:black);
GeoMakie.Colorbar(fig[2, 2], brt_surf)
display(fig)


    

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "BRT presence predictions (true positives/false negative)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions .>= best_brt_thresh, shading=false, colormap=:acton, colorrange=(-0.1, 1.1))
# GeoMakie.scatter!(ga, test_psabs.longitudes, test_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      color=brt_test_pres_predictions .>= best_brt_thresh,
                      strokewidth=0, markersize=6, colormap=cgrad(:bwr, rev=true));
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "BRT absences predictions (true negatives/false positives)")
# summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions .>= best_brt_thresh, shading=false, colormap=:acton, colorrange=(-0.1, 1.1))
GeoMakie.scatter!(ga, test_psabs.longitudes, 
                      test_psabs.latitudes, 
                      color=brt_test_psabs_predictions .< best_brt_thresh,
                      strokewidth=0, markersize=6, colormap=cgrad(:bwr, rev=true));
GeoMakie.Colorbar(fig[2, 2], summ_surf)
display(fig)


