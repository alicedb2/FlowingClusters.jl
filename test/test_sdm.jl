using Pkg
Pkg.activate(".")
using Revise
using SimpleSDMLayers
using GBIF
import GBIF: GBIFRecord, GBIFRecords, GBIFTaxon
using CSV
using DataFrames
using StatsBase
using LinearAlgebra
using ColorSchemes
using Random
# using StatsPlots
using Plots
using MultivariateNormalCRP
import GeoMakie
import CairoMakie

predictors = SimpleSDMPredictor(WorldClim, BioClim, [1, 3, 12])
dataset = load_dataset("data/Urocyon_cinereoargenteus.csv", predictors)
# dataset = load_dataset("data/Vulpes_velox.csv", predictors)


train_dataset, validation_dataset, test_dataset = split(dataset, 3)

train_df = unique(train_dataset.dataframe, train_dataset.predictornames)
validation_df = unique(validation_dataset.dataframe, validation_dataset.predictornames)
test_df = unique(test_dataset.dataframe, test_dataset.predictornames)


chain = initiate_chain(train_dataset, nb_samples=100)

left, right, bottom, top = -140.0, -51.0, -4.0, 60.0

p = clip(predictors[1], left=left, right=right, bottom=bottom, top=top)

layer_x, layer_y = 1.0 .* eachrow(reduce(hcat, keys(p)))
all_preds = [[p[lo, la] for p in predictors] for (lo, la) in zip(layer_x, layer_y)]
all_preds = [(p .- train_dataset.data_mean) ./ train_dataset.data_scale for p in all_preds]


#### Generate pseudo-absences
# Filter that one European occurrence
# filter!(:decimalLongitude => x -> x < -50.0, train_df)
function pseudoabsences(dataframe::DataFrame, layers::Vector{<:SimpleSDMLayer}; method=WithinRadius, distance=1.0)
    records = GBIFRecords(dataframe)
    layers = clip.(predictors, Ref(records))
    presences = mask(layers[1], records, Bool)
    wr_1deg_1 = rand(method, presences, distance=distance)
    pas = keys(wr_1deg_1)[collect(wr_1deg_1)]
    # pa_long, pa_lat = 1.0.*eachrow(reduce(hcat, pseudoabsences))
    preds = [Float64[layer[lonlat] for layer in layers] for lonlat in pas]
    return Dict(pas .=> preds)
    # return pas
end


train_pas = pseudoabsences(train_df, train_dataset.predictors, distance=2.0) 
train_pas_longs, train_pas_lats = 1.0 .* eachrow(reduce(hcat, Vector{Float64}.(keys(train_pas))))

validation_pas = pseudoabsences(validation_df, validation_dataset.predictors) 
validation_pas_longs, validation_pas_lats = 1.0 .* eachrow(reduce(hcat, Vector{Float64}.(keys(validation_pas))))

test_pas = pseudoabsences(test_df, test_dataset.predictors) 
test_pas_longs, test_pas_lats = 1.0 .* eachrow(reduce(hcat, Vector{Float64}.(keys(test_pas))))




map_tps = tail_probability(all_preds, chain.map_clusters, chain.map_hyperparams)

fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
            dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
            lonlims=(left, right), latlims=(bottom, top),
            title="Presence probability for MAP state"
);
map_surf = GeoMakie.surface!(ga, layer_x, layer_y, map_tps, shading=false, colorrange=(0, 1))
GeoMakie.scatter!(ga, train_pas_longs, train_pas_lats, strokewidth=0, markersize=3, color=:gray95);
GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=Makie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], map_surf)
display(fig)



summ_tps = tail_probability_summary(all_preds, chain)
summ_tps_weird = tail_probability_summary(all_preds, chain)

fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
            dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
            lonlims=(-140, -51), latlims=(-4, 60),
            title="Presence probability for median of the chain");
summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, summ_tps_weird.median, shading=false, colorrange=(0, 1))
# summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, (x -> x >= 0.2 ? x : 0.0).(summ_tps.median), shading=false, colorrange=(0, 1), colormap=:BuPu_4)
# GeoMakie.scatter!(ga, train_pas_longs, train_pas_lats, strokewidth=0, markersize=3, color=:gray95);
# GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=Makie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], summ_surf)
display(fig)


fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
            dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
            lonlims=(-140, -51), latlims=(-4, 60),
            title="Presence probability for median of the chain");
# summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, summ_tps.median, shading=false, colorrange=(0, 1))
summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, summ_tps.iqr, shading=false, colormap=:viridis)
# GeoMakie.scatter!(ga, train_pas_longs, train_pas_lats, strokewidth=0, markersize=3, color=:gray95);
# GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=Makie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], summ_surf)
display(fig)

fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
            dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
            lonlims=(-140, -51), latlims=(-4, 60),
            title="Presence probability for median of the chain");
# summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, summ_tps.median, shading=false, colorrange=(0, 1))
summ_surf = GeoMakie.surface!(ga, layer_x, layer_y, summ_tps.q95 .- summ_tps.q5, shading=false, colormap=:viridis)
# GeoMakie.scatter!(ga, train_pas_longs, train_pas_lats, strokewidth=0, markersize=3, color=:gray95);
# GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3, color=Makie.Cycled(6));
GeoMakie.Colorbar(fig[1, 2], summ_surf)
display(fig)



bioclim_sdm = bioclim_predictor(train_df, predictors)
bioclim_score = bioclim_sdm.(layer_x, layer_y)

fig = GeoMakie.Figure(resolution=(1500, 900), fontsize=20);
ga = GeoMakie.GeoAxis(fig[1, 1]; 
dest = "+proj=wintri +lon_0=$((left + right)/2) +lat_0=$((top + bottom)/2)", 
lonlims=(-140, -51), latlims=(-4, 60),
title="BIOCLIM at threshold=0.2"
);
bioclim_surf = GeoMakie.surface!(ga, layer_x, layer_y, bioclim_score .>= 0.05, shading=false, colorrange=(0, 1));
GeoMakie.scatter!(ga, train_pas_longs, train_pas_lats, strokewidth=0, markersize=3, color=:gray5);
GeoMakie.scatter!(ga, longitudes(dataset, flatten=true), latitudes(dataset, flatten=true), strokewidth=0, markersize=3);
GeoMakie.Colorbar(fig[1, 2], bioclim_surf)
display(fig)




bcp = bioclim_predictor(longitudes(train_dataset, unique=true, flatten=true), 
                        latitudes(train_dataset, unique=true, flatten=true), 
                        predictors)

bc_validation_p_predictions = bcp(longitudes(validation_dataset, unique=true), latitudes(validation_dataset, unique=true))
bc_validation_pa_predictions = bcp.(collect.(keys(validation_pas)))
bc_test_p_predictions = bcp(longitudes(test_dataset, unique=true), latitudes(test_dataset, unique=true))
bc_test_pa_predictions = bcp.(collect.(keys(test_pas)))

function performance_scores(scores_at_presences, scores_at_absences; threshold=nothing)
    if threshold !== nothing
        @assert 0.0 <= threshold <= 1.0
        scores_at_presences = scores_at_presences .>= threshold
        scores_at_absences = scores_at_absences .>= threshold
    end

    tp = sum(scores_at_presences)
    fn = sum(1 .- scores_at_presences)
    tn = sum(1 .- scores_at_absences)
    fp = sum(scores_at_absences)
    
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    J = sensitivity + specificity - 1
    kappa = 2 * (tp * tn - fn * fp)/((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))

    return (J=J, kappa=kappa, sensitivity=sensitivity, specificity=specificity, tp=tp, fn=fn, tn=tn, fp=fp)

end


thresholds = 0.05:0.05:0.95
#### BIOCLIM
bc_validation_p_predictions = bcp(longitudes(validation_dataset, unique=true), latitudes(validation_dataset, unique=true))
bc_validation_pa_predictions = bcp.(collect.(keys(validation_pas)))
bc_test_p_predictions = bcp(longitudes(test_dataset, unique=true), latitudes(test_dataset, unique=true))
bc_test_pa_predictions = bcp.(collect.(keys(test_pas)))


bc_validation_scores = Dict(threshold => performance_scores(bc_validation_p_predictions, bc_validation_pa_predictions, threshold=threshold) for threshold in thresholds);
best_bc_threshold, best_bc_scores = first(sort(collect(bc_validation_scores), by=x->x[2].J, rev=true));

bc_test_scores_withthreshold = performance_scores(bc_test_p_predictions, bc_test_pa_predictions, threshold=best_bc_threshold)
bc_test_score_nothreshold = performance_scores(bc_test_p_predictions, bc_test_pa_predictions, threshold=nothing)


#### MNCRP
mncrp_validation_p_predictions = tail_probability_summary(validation_dataset.data, chain);
mncrp_validation_pa_predictions = tail_probability_summary(MultivariateNormalCRP.rescale.(1.0 .* values(validation_pas), Ref(validation_dataset)), chain);
mncrp_test_p_predictions = tail_probability_summary(test_dataset.data, chain);
mncrp_test_pa_predictions = tail_probability_summary(MultivariateNormalCRP.rescale.(1.0 .* values(test_pas), Ref(test_dataset)), chain);

mncrp_validation_scores = Dict(threshold => performance_scores(mncrp_validation_p_predictions.median, mncrp_validation_pa_predictions.median, threshold=threshold) for threshold in thresholds);
best_mncrp_threshold, best_mncrp_validation_scores = first(sort(collect(mncrp_validation_scores), by=x->x[2].J, rev=true));


mncrp_test_score_withthreshold = performance_scores(mncrp_test_p_predictions.median, mncrp_test_pa_predictions.median, threshold=best_mncrp_threshold)
mncrp_test_score_nothreshold = performance_scores(mncrp_test_p_predictions.median, mncrp_test_pa_predictions.median)

