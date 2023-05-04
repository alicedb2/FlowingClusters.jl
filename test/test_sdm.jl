using Pkg
Pkg.activate(".")
using Revise
using SpeciesDistributionToolkit
using CSV
using ProgressMeter
using MCMCDiagnosticTools
using DataFrames
using StatsBase
using StatsFuns
using LinearAlgebra
using ColorSchemes
using Random: seed!
using SpecialFunctions
using Distributions
# using StatsPlots
using Plots
using EvoTrees
using MultivariateNormalCRP
import CairoMakie
import GeoMakie
include("src/helpers.jl")


ebird_df = DataFrame(CSV.File("data/ebird_data/ebird.csv"))
left, right, bottom, top = (1.2 * minimum(ebird_df.lon), 
                            1.2 * maximum(ebird_df.lon),
                            1.2 * minimum(ebird_df.lat), 
                            1.2 * maximum(ebird_df.lat))
lon_0, lat_0 = (left + right)/2, (top + bottom)/2
left, right, bottom, top = max(left, -180.0), min(right, 180.0), max(bottom, -90.0), min(top, 90.0)

# layernames = "BIO" .* string.([1, 12])
bioclim_layernames = "BIO" .* string.([1, 3, 12])
# bioclim_layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])
# layernames = "BIO" .* string.([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# all_layernames = "BIO" .* string.(1:19)
bioclim_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), 
                                     layer=layer, resolution=10.0,
                                     left=left, right=right, bottom=bottom, top=top)
                 for layer in bioclim_layernames]
# all_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), layer=layer, resolution=10.0) 
#               for layer in all_layernames]

landcover_layernames = collect(keys(layerdescriptions(RasterData(EarthEnv, LandCover))))
landcover_layers = [SimpleSDMPredictor(RasterData(EarthEnv, LandCover), 
                                      layer=layer,
                                      left=left, right=right, bottom=bottom, top=top)
                                    for layer in landcover_layernames]
clr_landcover_layers = clr(landcover_layers)

# Remove one landcover layer (open water)
# to prevent singular direction caused
# by compositional data
deleteat!(clr_landcover_layers, 5)
deleteat!(landcover_layernames, 5)

# landcover_layernames = String[]
# landcover_layers = SimpleSDMPredictor[]

_layers = Vector{SimpleSDMPredictor}(vcat(bioclim_layers, clr_landcover_layers))
layernames = Vector{String}(vcat(bioclim_layernames, landcover_layernames))




# dataset_all = MNCRPDataset("data/Urocyon_cinereoargenteus.csv", all_layers, layernames=all_layernames)
# dataset = MNCRPDataset("data/Urocyon_cinereoargenteus.csv", _layers, layernames=layernames)
# dataset = MNCRPDataset("data/aedes_albopictus.csv", _layers, layernames=layernames)

eb_3000_p = reduce(push!, sample(eachrow(subset(ebird_df, :sp6 => x -> x .== 1.0)), 3000, replace=false), init=DataFrame())
eb_3000_pa = reduce(push!, sample(eachrow(subset(ebird_df, :sp6 => x -> x .== 0.0)), 3000, replace=false), init=DataFrame())

dataset = MNCRPDataset(eb_3000_p, _layers, longlatcols=["lon", "lat"], layernames=layernames)
abs_dataset = MNCRPDataset(eb_3000_pa, _layers, longlatcols=["lon", "lat"], layernames=layernames)

# data = shuffle(dataset.data)
# c = initiate_chain(data, chain_samples=100, strategy=:sequential, standardize=false, optimize=false)
# c2 = initiate_chain(data, chain_samples=100, strategy=:sequential, standardize=false, optimize=false)

# GBIF
# train_dataset, validation_dataset, test_dataset = split(dataset, 3, standardize_with_first=true)
# train_df = unique(train_dataset.dataframe, train_dataset.layernames)
# validation_df = unique(validation_dataset.dataframe, validation_dataset.layernames)
# test_df = unique(test_dataset.dataframe, test_dataset.layernames)
# train_psabs = generate_pseudoabsences(train_df, train_dataset.layers, method=SurfaceRangeEnvelope);
# validation_psabs = generate_pseudoabsences(validation_df, validation_dataset.layers, method=SurfaceRangeEnvelope);
# test_psabs = generate_pseudoabsences(test_df, test_dataset.layers, method=SurfaceRangeEnvelope);


# eBird

train_dataset, validation_dataset, test_dataset = split(dataset, 3, standardize_with_first=false)
train_psabs, validation_psabs, test_psabs = split(abs_dataset, 3, standardize_with_first=false)

standardize!(train_dataset)
standardize_with!(validation_dataset, train_dataset)
standardize_with!(test_dataset, train_dataset)
standardize_with!(train_psabs, train_dataset)
standardize_with!(validation_psabs, train_dataset)
standardize_with!(test_psabs, train_dataset)


presence_chain = initiate_chain(train_dataset, chain_samples=100)
advance_chain!(presence_chain, 50; nb_splitmerge=200, 
               sample_every=20, attempt_map=false)
advance_chain!(presence_chain, 50; nb_splitmerge=200, 
               sample_every=20, attempt_map=true)

absence_chain = initiate_chain(train_psabs, chain_samples=100)
advance_chain!(absence_chain, 2000; nb_splitmerge=200, 
               sample_every=20, attempt_map=true)

# dump(presence_chain.hyperparams.diagnostics)
# clear_diagnostics!(presence_chain.hyperparams.diagnostics)

absence_chain = initiate_chain(standardize_with(train_psabs.predictors, train_dataset), 
                               standardize=false, chain_samples=100)
advance_chain!(absence_chain, 100; nb_splitmerge=300, 
               mh_stepscale=[0.6, 0.4, 0.4, 0.15, 0.55], 
               sample_every=50, attempt_map=false)


# left, right, bottom, top = -140.0, -51.0, -4.0, 60.0
# left, right, bottom, top = (1.2 * minimum(dataset.dataframe[!, dataset.longlatcols[1]]), 
#                             1.2 * maximum(dataset.dataframe[!, dataset.longlatcols[1]]),
#                             1.2 * minimum(dataset.dataframe[!, dataset.longlatcols[2]]), 
#                             1.2 * maximum(dataset.dataframe[!, dataset.longlatcols[2]]))
# lon_0, lat_0 = (left + right)/2, (top + bottom)/2
# left, right, bottom, top = max(left, -180.0), min(right, 180.0), max(bottom, -90.0), min(top, 90.0)

gat(f, t) = GeoMakie.GeoAxis(f; 
            dest = "+proj=wintri +lon_0=$lon_0 +lat_0=$lat_0", 
            # coastlines=true,
            lonlims=(left, right), latlims=(bottom, top),
            title=t
);

p = clip(_layers[1], left=left, right=right, bottom=bottom, top=top)
northam_longs, northam_lats = 1.0 .* eachrow(reduce(hcat, keys(p)))
northam_predictors = [Float64[p[lo, la] for p in _layers] for (lo, la) in zip(northam_longs, northam_lats)]
northam_predictors = [x + 1e-8 * rand(length(x)) for x in northam_predictors]
# northam_standardized_predictors = standardize_with(northam_predictors, train_dataset)
northam_standardized_predictors = standardize_with(northam_predictors, train_dataset)

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
bc_validation_psabs_predictions = bcp(standardize_with(validation_psabs, train_dataset).data)
best_bc_J = -Inf;
best_bc_thresh = 0.0;
thresholds = 0.001:001:0.999;
for thresh in thresholds
    perfscores = performance_scores(bc_validation_pres_redictions, bc_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_bc_J
        best_bc_J = perfscores.J
        best_bc_thresh = thresh
    end
end
bc_test_pres_redictions = bcp(test_dataset.data);
bc_test_psabs_predictions = bcp(standardize_with(test_psabs, train_dataset).data)
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


thresholds = 0.001:0.001:0.999;

# #### MNCRP tail MAP

maptail_validation_pres_predictions = tail_probability(validation_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_validation_psabs_predictions = tail_probability(validation_psabs.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
# sort!(presence_chain.map_clusters, by=length, rev=true)
# maptail_validation_pres_predictions = clustered_tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams).(validation_dataset.data)
# maptail_validation_psabs_predictions = clustered_tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams).(validation_psabs.data)
thresholds = 0.001:0.001:0.999;
best_maptail_J = -Inf;
best_maptail_thresh = 0.0;
for thresh in thresholds
    perfscores = performance_scores(maptail_validation_pres_predictions, maptail_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_maptail_J
        best_maptail_J = perfscores.J
        best_maptail_thresh = thresh
    end
end
maptail_test_pres_predictions = tail_probability(test_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
maptail_test_psabs_predictions = tail_probability(test_psabs.data, presence_chain.map_clusters, presence_chain.map_hyperparams);
# maptail_test_pres_predictions = clustered_tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams).(test_dataset.data);
# maptail_test_psabs_predictions = clustered_tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams).(test_psabs.data);
maptail_test_scores = performance_scores(maptail_test_pres_predictions, maptail_test_psabs_predictions);
maptail_test_scores_atbestJ = performance_scores(maptail_test_pres_predictions, maptail_test_psabs_predictions, threshold=best_maptail_thresh);
map(x -> round(x, digits=2), maptail_test_scores)
map(x -> round(x, digits=2), maptail_test_scores_atbestJ)


####### North-America train MAP tail figure

# cidx = 4
# map_tail = tail_probability(northam_standardized_predictors, presence_chain.map_clusters[cidx:cidx], presence_chain.map_hyperparams)

map_tail = tail_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams)
# map_tail = clustered_tail_probability(presence_chain.map_clusters, presence_chain.map_hyperparams).(northam_standardized_predictors)

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP tail probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_tail, shading=false, colorrange=(0, 1.5))
GeoMakie.scatter!(ga, longitudes(train_dataset, unique=true, flatten=true), 
                      latitudes(train_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));

# cluster_lonlat = [(d -> (d.lon[1], d.lat[1]))(train_dataset.unique_map[x]) for x in presence_chain.map_clusters[cidx]]         
# GeoMakie.scatter!(ga, getindex.(cluster_lonlat, 1), 
#                       getindex.(cluster_lonlat, 2), 
#                       strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));


GeoMakie.scatter!(ga, longitudes(train_psabs, unique=true, flatten=true), 
                      latitudes(train_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:gray95);
GeoMakie.Colorbar(fig[1, 2], map_surf)

ga = gat(fig[2, 1], "MAP tail threshold")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_tail .>= best_maptail_thresh, shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(train_dataset, unique=true, flatten=true), 
                      latitudes(train_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.scatter!(ga, longitudes(train_psabs, unique=true, flatten=true), 
                      latitudes(train_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:grey95);
GeoMakie.Colorbar(fig[2, 2], map_surf)
display(fig)


##### MNCRP pclass MAP
best_mappclass_J = -Inf;
best_mappclass_thresh = 0.0;
mappclass_validation_pres_predictions = presence_probability(validation_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_validation_psabs_predictions = presence_probability(validation_psabs.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
thresholds = 0.001:0.001:0.999;
for thresh in thresholds
    perfscores = performance_scores(mappclass_validation_pres_predictions, mappclass_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_mappclass_J
        best_mappclass_J = perfscores.J
        best_mappclass_thresh = thresh
    end
end
mappclass_test_pres_predictions = presence_probability(test_dataset.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_psabs_predictions = presence_probability(test_psabs.data, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams);
mappclass_test_scores = performance_scores(mappclass_test_pres_predictions, mappclass_test_psabs_predictions);
mappclass_test_scores_atbestJ = performance_scores(mappclass_test_pres_predictions, mappclass_test_psabs_predictions, threshold=best_mappclass_thresh);
map(x -> round(x, digits=2), mappclass_test_scores)
map(x -> round(x, digits=2), mappclass_test_scores_atbestJ)


####### North-America MAP pclass figure

map_pclass = presence_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams)

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP presence class probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_pclass, shading=false, colorrange=(0, 1.5))
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], map_surf)

ga = gat(fig[2, 1], "MAP presence class threshold")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, map_pclass .>= best_mappclass_thresh, shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), latitudes(test_psabs, unique=true, flatten=true), strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[2, 2], map_surf)
display(fig)



#### MAP cluster assignements
map_tail = tail_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams)
map_idx = map_cluster_assignment_idx.(northam_standardized_predictors, Ref(presence_chain.map_clusters), Ref(presence_chain.map_hyperparams))
idx_mask = map_idx .<= 19
map_idx[.!idx_mask] .= 0


fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "MAP tail probability")
map_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, logit.(map_tail), shading=false)#, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(dataset, unique=true, flatten=true), latitudes(dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
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
mediantail_validation_psabs_predictions = tail_probability_summary(validation_psabs.data, presence_chain).median
thresholds = 0.001:0.001:0.999;
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
mediantail_test_psabs_predictions = tail_probability_summary(test_psabs.data, presence_chain).median
mediantail_test_scores = performance_scores(mediantail_test_pres_predictions, mediantail_test_psabs_predictions);
mediantail_test_scores_atbestJ = performance_scores(mediantail_test_pres_predictions, mediantail_test_psabs_predictions, threshold=best_mediantail_thresh);
map(x -> round(x, digits=2), mediantail_test_scores)
map(x -> round(x, digits=2), mediantail_test_scores_atbestJ)



fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Median tail probability")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median, shading=false, colormap=:acton, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(train_psabs, unique=true, flatten=true), latitudes(train_psabs, unique=true, flatten=true), strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "Median tail threshold")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_tail_summary.median .>= best_mediantail_thresh, shading=false, colormap=:acton, colorrange=(-0.3, 1.3))
GeoMakie.scatter!(ga, longitudes(train_psabs, unique=true, flatten=true), latitudes(train_psabs, unique=true, flatten=true), strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
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


#### MNCRP pclass median
medianpclass_validation_pres_predictions = presence_probability_summary(validation_dataset.data, presence_chain, absence_chain).median
medianpclass_validation_psabs_predictions = presence_probability_summary(validation_psabs.data, presence_chain, absence_chain).median
thresholds = 0.001:0.001:0.999;
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
medianpclass_test_psabs_predictions = presence_probability_summary(test_psabs.data, presence_chain, absence_chain).median
medianpclass_test_scores = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_psabs_predictions);
medianpclass_test_scores_atbestJ = performance_scores(medianpclass_test_pres_predictions, medianpclass_test_psabs_predictions, threshold=best_medianpclass_thresh);
map(x -> round(x, digits=2), medianpclass_test_scores)
map(x -> round(x, digits=2), medianpclass_test_scores_atbestJ)




fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Median presence class probability")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_pclass_summary.median, shading=false, colorrange=(0, 1.5), colormap=:viridis)
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], summ_surf)

ga = gat(fig[2, 1], "Median presence class threshold")
summ_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, northam_pclass_summary.median .>= best_medianpclass_thresh, shading=false, colorrange=(-0.3, 1.3), colormap=:viridis)
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[2, 2], summ_surf)

display(fig)

#### MNDP Variational

thresholds = 0.001:0.001:0.999;
best_variationaltail_J = -Inf;
best_variationaltail_thresh = 0.0;
variationaltail_validation_pres_predictions = tail_probability(validation_dataset.data, v);
variationaltail_validation_psabs_predictions = tail_probability(standardize_with(validation_psabs, train_dataset).data, v);
for thresh in thresholds
    perfscores = performance_scores(variationaltail_validation_pres_predictions, variationaltail_validation_psabs_predictions, threshold=thresh);
    if perfscores.J > best_variationaltail_J
        best_variationaltail_J = perfscores.J
        best_variationaltail_thresh = thresh
    end
end
variationaltail_test_pres_predictions = tail_probability(test_dataset.data, v);
variationaltail_test_psabs_predictions = tail_probability(standardize_with(test_psabs, train_dataset).data, v);
variationaltail_test_scores = performance_scores(variationaltail_test_pres_predictions, variationaltail_test_psabs_predictions);
variationaltail_test_scores_atbestJ = performance_scores(variationaltail_test_pres_predictions, variationaltail_test_psabs_predictions, threshold=best_variationaltail_thresh);
map(x -> round(x, digits=2), variationaltail_test_scores)
map(x -> round(x, digits=2), variationaltail_test_scores_atbestJ)


variational_northam_tail = tail_probability(northam_standardized_predictors, v)

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Variational tail probability")
variational_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, variational_northam_tail, shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true), 
                  strokewidth=0, markersize=6, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                  strokewidth=0, markersize=6, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], variational_surf)

ga = gat(fig[2, 1], "Variational tail threshold")
variational_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, 0.1 .+ 0.8 .* (variational_northam_tail .>= best_variationaltail_thresh), shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true),
                  strokewidth=0, markersize=6, color=:grey95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                  strokewidth=0, markersize=6, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[2, 2], map_surf)
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
brt_train_psabs_predictors = collect(reduce(hcat, train_psabs.data)')
# brt_train_psabs_predictors = collect(reduce(hcat, standardize_with(train_psabs, train_dataset).data)')
brt_train_X = vcat(brt_train_pres_predictors, brt_train_psabs_predictors)
brt_train_y = vcat(fill(1.0, size(brt_train_pres_predictors, 1)), fill(0.0, size(brt_train_psabs_predictors, 1)))

brt_validation_pres_predictors = collect(reduce(hcat, validation_dataset.data)')
brt_validation_psabs_predictors = collect(reduce(hcat, validation_psabs.data)')
# brt_validation_psabs_predictors = collect(reduce(hcat, standardize_with(validation_psabs, train_dataset).data)')
brt_validation_X = vcat(brt_validation_pres_predictors, brt_validation_psabs_predictors)
brt_validation_y = vcat(fill(1.0, size(brt_validation_pres_predictors, 1)), fill(0.0, size(brt_validation_psabs_predictors, 1)))

brt_model = fit_evotree(gaussian_tree_parameters; x_train=brt_train_X, y_train=brt_train_y, x_eval=brt_validation_X, y_eval=brt_validation_y)

brt_validation_predictions = getindex.(eachrow(EvoTrees.predict(brt_model, brt_validation_X)), 1);
brt_validation_predictions = (x -> x > 1.0 ? 1.0 : x).(brt_validation_predictions);
brt_validation_predictions = (x -> x < 0.0 ? 0.0 : x).(brt_validation_predictions);
brt_validation_pres_pedictions = brt_validation_predictions[1:size(brt_validation_pres_predictors, 1)]
brt_validation_psabs_predictions = brt_validation_predictions[size(brt_validation_pres_predictors, 1)+1:end]

thresholds = 0.001:0.001:0.999;
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
brt_test_psabs_predictors = collect(reduce(hcat, test_psabs.data)')
# brt_test_psabs_predictors = collect(reduce(hcat, standardize_with(test_psabs, train_dataset).data)')
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
brt_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions, 
                             shading=false, colorrange=(0, 1.5), colormap=:viridis)
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true), 
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], brt_surf)

ga = gat(fig[2, 1], "BRT threshold")
brt_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, brt_northam_predictions .> best_brt_thresh, 
                             shading=false, colorrange=(-0.3, 1.3), colormap=:acton)
GeoMakie.scatter!(ga, longitudes(test_psabs, unique=true, flatten=true), 
                      latitudes(test_psabs, unique=true, flatten=true),
                      strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), 
                      latitudes(test_dataset, unique=true, flatten=true),
                      strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
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




c = initiate_chain(dataset, chain_samples=100)
MultivariateNormalCRP.append_tail_probabilities!(
    c.clusters, c.hyperparams, optimize=true, 
    rejection_samples=10000)
c.map_logprob = -Inf
attempt_map!(c)

for i in 1:10
    MultivariateNormalCRP.update_tail_probabilities!(
        c.clusters, c.hyperparams,
        rejection_samples=10000)

    advance_chain!(c, 1; nb_splitmerge=200, 
                mh_stepscale=[1.4, 1.5, 1.3, 1.2, 1.4], 
                sample_every=10, attempt_map=true, pretty_progress=false)


    
    # c.hyperparams.diagnostics.step_scale = optimal_step_scale_local(
    #     c.clusters, c.hyperparams,
    #     optimize=true
    # )
end


for cluster in c.map_clusters
    _, _, psi_c, _ = MultivariateNormalCRP.updated_niw_hyperparams(cluster, c.map_hyperparams.mu, c.map_hyperparams.lambda, c.map_hyperparams.psi, c.map_hyperparams.nu)
    MultivariateNormalCRP.logdetpsd(psi_c)
end



d = 6
mu0, lambda0, L0, nu0 = rand(d), 2.0, LowerTriangular(rand(d, d)), 10.0
psi0 = L0 * L0'
niw_draws = [drawNIW(mu0, lambda0, psi0, nu0) for _ in 1:1000000];

mean([logdet(ms[2]) for ms in niw_draws])
# -2 * finite_difference_gradient(nu -> MultivariateNormalCRP.log_Zniw(nothing, mu0, lambda0, psi0, nu[1]), [nu0])[1]
-d * log(2) + logdet(psi0) - sum(polygamma.(0, nu0/2 + (1 - j)/2) for j in 1:d)

mean([ms[2] for ms in niw_draws])
psi0 / (nu0 - d - 1)

mean([inv(ms[2]) for ms in niw_draws])
nu0 * inv(psi0)

mean([ms[2] \ ms[1] for ms in niw_draws])
nu0 * (psi0 \ mu0)

mean([ms[1]' * (ms[2] \ ms[1]) for ms in niw_draws])
d/lambda0 + nu0 * mu0' * (psi0 \ mu0)

mean([-1/2 * (ms[1] - mu0)' * inv(ms[2]) * (ms[1] - mu0) for ms in niw_draws])
-d/2/lambda0


randpsd = (x -> x * x')(rand(d, d))
mean([tr(psi0 * inv(ms[2])) for ms in niw_draws])
tr(psi0 * nu0 * inv(psi0)), nu0 * d


mean([(
    -(nu0 + d + 2)/2 * logdet(sig)
    - 1/2 * tr(psi0 / sig)
    - lambda0/2 * (mu - mu0)' * (sig \ (mu - mu0))
    )
    for (mu, sig) in niw_draws])
(
    (nu0 + d + 2)*(d/2 * log(2) - 1/2 * logdet(psi0) + 1/2 * sum([polygamma(0, nu0/2 + (1-j)/2) for j in 1:d]))
    - nu0 * d/2 - lambda0/2 * (d/lambda0 + nu0 * mu0' * (psi0 \ mu0))
    + lambda0/2 * nu0 * mu0' * (psi0 \ mu0)
)

v = MNDPVariational(shuffle(dataset.data), 50);

v.hyperparams.alpha = c.map_hyperparams.alpha;
v.hyperparams.mu = c.map_hyperparams.mu;
v.hyperparams.lambda = c.map_hyperparams.lambda;
v.hyperparams.psi = c.map_hyperparams.psi;
v.hyperparams.nu = c.map_hyperparams.nu;

# randomize_phi!(v);

randomize_eta!(v);
advance_phi!(v);

randomize_phi!(v);
# advance_var!(v);
gamma1s = [copy(v.gamma1_k)];
gamma2s = [copy(v.gamma2_k)];
sch = [3, 3]
for step in 1:100
    print("\r$step")

    # if step >= 8 && floor(log2(step)) == log2(step)
    #     v.hyperparams.diagnostics.step_scale = optimal_step_scale_local(
    #        draw_partition(v, map=true), v.hyperparams,
    #        optimize=true, jacobian=false, verbose=true
    #    )
    # end
    advance_var!(v, Nk=nothing, Nn=nothing, sch=sch)
    # if step >= 8
    #     sample_partition = draw_partition(v)
    #     for i in 1:10
    #         advance_hyperparams!(sample_partition, v.hyperparams, v.hyperparams.diagnostics.step_scale)
    #     end
    # end
    if mod(step, 1) == 0
        push!(gamma1s, copy(v.gamma1_k))
        push!(gamma2s, copy(v.gamma2_k))
    end
end

_gamma1s = copy(reduce(hcat, gamma1s)');
_gamma2s = copy(reduce(hcat, gamma2s)');
elg = _gamma1s ./ (_gamma1s .+ _gamma2s);
log_Pgenerative(draw_partition(v, map=true), v.hyperparams)
plot(elg, legend=false)

Set(getindex.(findmax.(eachrow(v.phi_nk)), 2))

as = LinRange(0.01, 50, 100)
w1 = 1.0
w2 = 1.0
N, K = 100, 10



(log_Pgenerative(c.clusters, unpack(c.hyperparams, transform=false) .+ 0.000000001 * (1:30 .== 8), backtransform=false, jacobian=false, hyperpriors=true) 
- log_Pgenerative(c.clusters, unpack(c.hyperparams, transform=false), backtransform=false, jacobian=false, hyperpriors=true))/0.000000001


function comm_mat(m, n)
    A = reshape(1:m*n, m, n)
    v = reshape(A', m * n)

    return I(m*n)[v, :]
end

function epsilonij(i, j, d)
    return 1.0 * (1:d .== i) * (1:d .== j)'
end

function L_epsilon(psi_epsilon::Matrix{Float64}, L::LowerTriangular{Float64})
    d = size(L, 1)

    psi_epsilon = (psi_epsilon + psi_epsilon')/2
    
    vec_L_epsilon = (kron(I(d), L) * comm_mat(d, d) + kron(L', I(d))) \ reshape(psi_epsilon, d*d)

    return reshape(vec_L_epsilon, d, d)

end


h = c.hyperparams
heps = deepcopy(c.hyperparams)
psieps = 0.000001 * (epsilonij(1, 1, 6) + epsilonij(1, 1, 6)) / 2
MultivariateNormalCRP.psi!(heps, heps.psi + psieps)

(log_Pgenerative(c.clusters, heps, hyperpriors=false) - log_Pgenerative(c.clusters, h, hyperpriors=false))/0.000001

grad_log_Pgenerative(c.clusters, c.hyperparams, hyperpriors=false)

delta = 0.000001
(log_Pgenerative(c.clusters, 
    unpack(c.hyperparams, transform=false) .+ delta * (1:192 .== 192), 
    backtransform=false, jacobian=false, hyperpriors=true) 

- log_Pgenerative(c.clusters, 
    unpack(c.hyperparams, transform=false), 
    backtransform=false, jacobian=false, hyperpriors=true))/delta

function nf(w, u, b, h)
    m(x) = -1.0 + log(1 + exp(x))
    ubar = u #.+ w ./ norm(w)^2 .* (m.(w.*u) .- w.*u) 
    function nf(z)
        return z .+ ubar.*h(w' * z + b)
    end
    return nf
end

foo = rand(MvNormal(zeros(2), I(2)), 30000);

 

histogram2d(Tuple.(nf(10*[1.0, 0.0], (theta->[cos(theta), sin(theta)])(0.53*pi), 0.0, tanh).(eachcol(foo))), bins=100)

w = [1.0, 1.0]
u = [10.0, 10.0]
b = -1.0
plot3d(LinRange(0, 1, 30), LinRange(0, 1, 30), 
(z...)->let u=u, w=w, b=b
z = collect(z)
abs(1 + u' * w * sech(w'*z + b)^2)
end,
st=:surface)


alpha_samples = Float64[]
a, s = 1.0, 1.0
for i in 1:10000
    a, s = MultivariateNormalCRP.sample_alpha(100, a, s, 10.0)
    push!(alpha_samples, a)
end
histogram(alpha_samples, normalize=true)
plot!(a->MultivariateNormalCRP.jeffreys_alpha(a, 100)/32.8289, 10, 250)


d = 10
foo = MNCRPHyperparams(pack(rand(3 + d + div(d * (d + 1), 2)))...);

for _ in 1:100000
    set_theta!(
        foo, 
        4*(rand() - 0.5), 
        rand(2+d+1:2+d+div(d*(d+1),2)), backtransform=false
        );
    @assert all(isapprox.(foo.L * foo.L', foo.psi))        
end

d = length(c2.hyperparams.mu)
delta = 1e-6
Lidx = 6
k = 2 + d + Lidx
foo = (log_Pgenerative(c2.clusters, delta .* (1:192 .== k) + unpack(c2.hyperparams, transform=true), backtransform=true, jacobian=true) - log_Pgenerative(c2.clusters, unpack(c2.hyperparams, transform=true), backtransform=true, jacobian=true)) / delta;
bar = grad_log_Pgenerative(c2.clusters, unpack(c2.hyperparams, transform=true), backtransform=true, jacobian=true);
foo, bar[4][Lidx]


seed!(2)
v = MNDPVariational(data3d, 150);
advance_variational!(v, 3000; minibatch_size=30, kappa=0.8, tau=1.0);
log_Pgenerative(draw_partition(v, map=true), v.hyperparams)


foo = zeros(v.T)
for k in 1:v.T-1
    for n in 1:v.N
        foo[k] += exp(logsumexp([v.logphi_nk[n, j] for j in k+1:v.T]))
        # foo[k] += sum([exp(v.logphi_nk[n, j]) for j in k+1:v.T])
    end
end

foosigma_k = permutedims(cat((L -> LowerTriangular(L) * LowerTriangular(L)').(Matrix{Float64}.(eachslice(rand(150, 6, 6), dims=1)))..., dims=3), [3, 1, 2])
MultivariateNormalCRP.log_Q(v.gamma1_k ./ (v.gamma1_k + v.gamma2_k), exp.(v.logphi_nk), rand(150, 6), foosigma_k, v.gamma1_k, v.gamma2_k, v.eta1_k, v.eta2_k, v.eta3_k, v.eta4_k, v.logphi_nk)

foosigma_k = permutedims(cat((L -> LowerTriangular(L) * LowerTriangular(L)').(Matrix{Float64}.(eachslice(rand(150, 6, 6), dims=1)))..., dims=3), [3, 1, 2])


v = MNDPVariational(data6d, 70);
v.hyperparams = deepcopy(c6.hyperparams);
advance_variational_stochastic!(v, 5000; 
minibatch_size=10, kappa=0.7, tau=100.0);
log_Pgenerative(draw_partition(v, map=true), v.hyperparams)

v = MNDPVariational(train_dataset.data, 100);
# v.hyperparams = deepcopy(c6.hyperparams);
advance_variational_stochastic!(v, 1000, minibatch_size=100, kappa=0.5, tau=10.0);
log_Pgenerative(draw_partition(v, map=true), v.hyperparams)

####### North-America variational figure

var_tail = tail_probability(v).(northam_standardized_predictors)
# map_pclass = presence_probability(northam_standardized_predictors, presence_chain.map_clusters, presence_chain.map_hyperparams, absence_chain.map_clusters, absence_chain.map_hyperparams)

fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = gat(fig[1, 1], "Variational tail probability")
var_surf = GeoMakie.surface!(ga, northam_longs, northam_lats, var_tail, shading=false, colorrange=(0, 1))
# GeoMakie.scatter!(ga, train_psabs.longitudes, train_psabs.latitudes, strokewidth=0, markersize=3, color=:gray95);
# GeoMakie.scatter!(ga, longitudes(test_dataset, unique=true, flatten=true), latitudes(test_dataset, unique=true, flatten=true), strokewidth=0, markersize=3, color=GeoMakie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], var_surf)
display(fig)



fig = GeoMakie.Figure(resolution=(1400, 2000), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
            dest="+proj=wintri +lon_0=$((minimum(ebird_df.lon) + maximum(ebird_df.lon))/2) +lat_0=$((minimum(ebird_df.lat) + maximum(ebird_df.lat))/2)", 
            coastlines=true,
            lonlims=(minimum(ebird_df.lon), maximum(ebird_df.lon)),
            latlims=(minimum(ebird_df.lat), maximum(ebird_df.lat))
        );



GeoMakie.scatter!(ga, subset(eb, :sp6 => x -> x .== 1.0).lon, subset(eb, :sp6 => x -> x .== 1.0).lat, strokewidth=0, markersize=3);
GeoMakie.scatter!(ga, subset(eb, :sp6 => x -> x .== 0.0).lon, subset(eb, :sp6 => x -> x .== 0.0).lat, strokewidth=0, markersize=3);
# GeoMakie.Colorbar(fig[1, 2], var_surf)
display(fig)


