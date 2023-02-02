using Pkg
Pkg.activate(".")
using Revise
using SimpleSDMLayers
using GBIF
using CSV
using DataFrames
using StatsBase
using LinearAlgebra
using ColorSchemes
using Random
# using StatsPlots
using Plots
using MultivariateNormalCRP

predictors = SimpleSDMPredictor(WorldClim, BioClim, [1, 3, 12])
obs = DataFrame(CSV.File("data/Urocyon_cinereoargenteus.csv", delim="\t"))

bioclim_temperature, bioclim_isothermality, bioclim_precipitation = predictors
# obs = DataFrame(CSV.File("data/polyommatus_icarus.csv", delim="\t"))
# obs = DataFrame(CSV.File("data/Vulpes_lagopus.csv", delim="\t"))
filter!(row -> !ismissing(row.decimalLatitude) && !ismissing(row.decimalLongitude), obs)

obs.temperature = bioclim_temperature[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
obs.isothermality = bioclim_isothermality[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
obs.precipitation = bioclim_precipitation[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
filter!(row -> !isnothing(row.temperature) && !isnothing(row.precipitation) && !isnothing(row.isothermality), obs)
obs.temperature = Vector{Float64}(obs.temperature)
obs.precipitation = Vector{Float64}(obs.precipitation)
obs.isothermality = Vector{Float64}(obs.isothermality)

shuffled_obs = shuffle(obs)
train_obs = shuffled_obs[1:2:size(shuffled_obs, 1), :]
test_obs = shuffled_obs[2:2:size(shuffled_obs, 1), :]

# obs.standardized_temperature = (obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature)
# obs.standardized_precipitation = (obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation)

# dataset = shuffle(collect(Set((collect.(collect(zip(obs.temperature, obs.isothermality, obs.precipitation)))))))
train_dataset = collect(Set((collect.(collect(zip(train_obs.temperature, train_obs.isothermality, train_obs.precipitation))))))
test_dataset = collect(Set((collect.(collect(zip(test_obs.temperature, test_obs.isothermality, train_obs.precipitation))))))

chain = initiate_chain(train_dataset, nb_samples=200);

advance_chain!(chain, 1000; nb_splitmerge=100, mh_stepscale=1.0)

coordinates = elements(chain);
tailprobs = tail_probability(coordinates, chain, 10000)
tp_summaries = tail_probability_summary(coordinates, chain, 10000)

# surface(100 * getindex.(coordinates, 1), 100 * getindex.(coordinates, 2), tailprobs.median, c=cgrad(:curl, rev=true), camera=(0, 90))

scatter(Tuple.(coordinates), marker_z=tailprobs.median, 
        c=cgrad(:curl, rev=true), clims=(0.0, 1.0), msw=0, ms=2)


histogram2d([Tuple(x[[1, 2]]) for x in foo], bins=60);
scatter!([Tuple(xmax_median[[1, 2]])], msw=0, ms=5, mc=:darkgreen);
xlims!(-3, 3);
ylims!(-3, 3)
histogram2d([Tuple(x[[3, 2]]) for x in foo], bins=60);
scatter!([Tuple(xmax_median[[3, 2]])], msw=0, ms=5, mc=:darkgreen);
xlims!(-3, 3);
ylims!(-3, 3)
histogram2d([Tuple(x[[3, 1]]) for x in foo], bins=60);
scatter!([Tuple(xmax_median[[3, 1]])], msw=0, ms=5, mc=:darkgreen);
xlims!(-3, 3);
ylims!(-3, 3)

# train_dataset_std = collect(1.0 .* eachcol(inv(chain.data_scalematrix) * reduce(hcat, [d .- chain.data_mean for d in test_dataset])))

# xmaxes = Vector{Float64}[]
# for (clusters, hyperparams) in zip(chain.clusters_samples, chain.hyperparams_samples)
#     lp = predictive_logpdf(clusters, hyperparams)
#     objfun = x -> -lp(x)
#     x0 = train_dataset_std[last(findmax(lp.(train_dataset_std)))]
#     optres = optimize(objfun, x0, NelderMead())
#     xmax = Optim.minimizer(optres)
#     push!(xmaxes, xmax)
# end

# xmax_median = median(reduce(hcat, xmaxes), dims=2)





############## BIOCLIM ##############
observations = occurrences(
    GBIF.taxon("Urocyon cinereoargenteus"; strict=true),
    "hasCoordinate" => "true",
    "limit" => 10,
)

left, right = extrema([o.longitude for o in observations]) .+ (-5,5)
bottom, top = extrema([o.latitude for o in observations]) .+ (-5,5)
predictors = SimpleSDMPredictor(WorldClim, BioClim, [1, 3, 12])
# predictors = SimpleSDMPredictor(WorldClim, BioClim, [1])
# predictors = SimpleSDMPredictor(WorldClim, BioClim, [3])
# predictors = SimpleSDMPredictor(WorldClim, BioClim, [12])

_pixel_score(x) = 2.0(x > 0.5 ? 1.0-x : x)

# Returns a model for one layer
function SDM(layer::T, observations::GBIFRecords) where {T <: SimpleSDMLayer}
    qf = ecdf(layer[observations]) # We only want the observed values
    return (_pixel_score∘qf)
end

# Returns layer with lowest score
function SDM(predictors::Vector{T}, models) where {T <: SimpleSDMLayer}
    @assert length(models) == length(predictors)
    return minimum([broadcast(models[i], predictors[i]) for i in 1:length(predictors)])
end

models = [SDM(predictor, observations) for predictor in predictors]
predictions = SDM(predictors, models)

rescale!(predictions, collect(0.0:0.01:1.0))

cutoff = broadcast(x -> x > 0.05, predictions)
cutoff_predictions = broadcast(x -> x > 0.05 ? x : 0.0, predictions)

plot(predictions, frame=:box, c=:lightgrey);
plot!(cutoff_predictions, clim=(0, 1))
plot(mask(cutoff, predictions), clim=(0, 1), c=:bamako)
# scatter!([(o.longitude, o.latitude) for o in observations], ms=0.1, msw=0, c=:orange, lab="");
# scatter!(Tuple.(zip(obs.decimalLongitude, obs.decimalLatitude)), ms=0.3, msw=0, label=nothing);
xaxis!("Longitude");
yaxis!("Latitude")





function bioclim_predictor(training_observations::DataFrame, layers::Vector{T}; longlat=(:decimalLongitude, :decimalLatitude) ) where {T <: SimpleSDMLayer}
    
    maprange(x, mininput, maxinput, minoutput, maxoutput) = minoutput + (x - mininput) * (maxoutput - minoutput) / (maxinput - mininput)
    rankscore(x) = 2.0 * (x > 0.5 ? 1.0-x : x)
    
    long, lat = longlat
    
    training_observations = filter(row -> !ismissing(row[long])&& !isnothing(row[long]) && !ismissing(row[lat]) && !isnothing(row[lat]), obs)
    
    # Vector of vectors V such that
    # V[k][l] gives value of layer l for observation k
    obs_layer_vals = [
        [predictor[o[long], o[lat]] 
        for predictor in layers] 
        for o in eachrow(training_observations)
            ]
            
    filter!(x -> all(!isnothing, x), obs_layer_vals)
    obs_layer_vals = Vector{Float64}.(obs_layer_vals)
    # Transpose, get distribution of 
    # training observation values
    # for each layers
    obs_layer_vals = collect.(eachrow(reduce(hcat, obs_layer_vals)))
    println(typeof.(obs_layer_vals))
    
    # Rank each layer value 
    
    function score_new_observation(longitude::T, latitude::T) where {T <: Real}
        
        if ismissing(longitude) || isnothing(longitude) || ismissing(latitude) || isnothing(latitude)
            return nothing
        end
        
        if any(isnothing.([predictor[longitude, latitude] for predictor in layers]))
            return 0.0
        end
        
        obs_layer_scores = [rankscore(quantilerank(lvals, predictor[longitude, latitude])) 
                            for (predictor, lvals) in zip(predictors, obs_layer_vals) 
                                    if !isnothing(predictor[longitude, latitude])]
            
        obs_score = minimum(obs_layer_scores)
        
        obs_score = obs_score > 0.05 ? obs_score : 0.0
        
        return obs_score
        
    end
        
    function score_new_observation(observations::DataFrame; longlat=(:decimalLongitude, :decimalLatitude))
        
        long, lat = longlat
        
        return score_new_observation.(observations[!, long], observations[!, lat])
    end
        
    return score_new_observation

end
            
# scatter!(Tuple.(zip(obs.decimalLongitude, obs.decimalLatitude)), marker_z=obs_layer_minranks .> 0.5, msw=0, ms=0.2)
# scatter!(Tuple.(zip(obs.decimalLongitude, obs.decimalLatitude)), marker_z=obs_layer_minranks .> 0.05, msw=0, ms=0.2)
predictors = SimpleSDMPredictor(WorldClim, BioClim, [1, 3, 12])

obs = DataFrame(CSV.File("data/Urocyon_cinereoargenteus.csv", delim="\t"))


filter!(row -> !ismissing(row.decimalLatitude) && !ismissing(row.decimalLongitude), obs)
# unique(obs, [:standardized_temperature, :standardized_precipitation])
# obs = DataFrame(CSV.File("data/Vulpes_lagopus.csv", delim="\t"))

shuffled_obs = shuffle(obs)
train_obs = shuffled_obs[1:2:size(shuffled_obs, 1), :]
test_obs = shuffled_obs[2:2:size(shuffled_obs, 1), :]


bp = bioclim_predictor(train_obs, predictors)
test_obs_scores = bp(test_obs)

plot(predictions, frame=:box, c=:lightgrey);
scatter!(Tuple.(zip(test_obs.decimalLongitude, test_obs.decimalLatitude)), marker_z=test_obs_scores .>= 0.05, msw=0, ms=0.2)

# tp = sum(test_obs_scores .>= 0.05)
tp = sum(test_obs_scores)
fn = sum(test_obs_scores .< 0.05) # should all be == 0.0

J = tp / (tp + fn) - 1





fig = GeoMakie.Figure(resolution=(1500, 900));
ga = GeoMakie.GeoAxis(
    fig[1, 1];
    dest = "+proj=wintri",
    coastlines = true,
    lonlims=(-150, -40),
    latlims=(-10, 80)
    );

for cluster in sort(chain.map_clusters, by=length, rev=true)[1:2]
    lls = longlats(cluster, dataset)
    longs, lats = 1.0 .* eachrow(reduce(hcat, lls))
    GeoMakie.scatter!(ga, longs, lats, c=:black)
end

# x = [x[1] for x in canada_coordinates];
# y = [y[2] for y in canada_coordinates];
# sp = GeoMakie.surface!(ga, x, y, map_presence_canada, shading=false)
GeoMakie.Colorbar(fig[1, 2], sp)
display(fig)