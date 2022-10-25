using Revise
using MultivariateNormalCRP
using SimpleSDMLayers
using CSV
using DataFrames
using Random
using StatsBase
using Ripserer
using LinearAlgebra
using ColorSchemes
using Serialization
using StatsPlots
# using Plots
using Random: seed!
using GeoMakie
using CairoMakie
using JLD2

bioclim_temperature, bioclim_precipitation = SimpleSDMPredictor(WorldClim, BioClim, [1, 12])
# obs = DataFrame(CSV.File("data/polyommatus_icarus.csv", delim="\t"))
obs = DataFrame(CSV.File("data/aedes_albopictus.csv", delim="\t"))

# obs = deserialize("data/aedes_albopictus_tempprec.dataframe")

# Remove rows with missing latitude and/or longitude
# and then with missing temperature and/or precipitation
filter!(row -> !ismissing(row.decimalLatitude) && !ismissing(row.decimalLongitude), obs)
obs.temperature = bioclim_temperature[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
obs.precipitation = bioclim_precipitation[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
filter!(row -> !isnothing(row.temperature) && !isnothing(row.precipitation), obs)

obs.temperature = Vector{Float64}(obs.temperature)
obs.precipitation = Vector{Float64}(obs.precipitation)
obs.standardized_temperature = (obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature)
obs.standardized_precipitation = (obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation)

offset = [mean(obs.temperature), mean(obs.precipitation)]
scale_matrix = [1/std(obs.temperature) 0.0; 0.0 1/std(obs.precipitation)]

# filter!(row -> row.basisOfRecord == "HUMAN_OBSERVATION" || row.basisOfRecord == "OCCURRENCE" || obs.basisOfRecord == "MATERIAL_SAMPLE", obs)

# Keep records with unique temps/precs
# (although they'll naturally be dropped
# because clusters are Sets)

unique_obs = unique(obs, [:standardized_temperature, :standardized_precipitation])
unique_clusters = groupby(obs, [:standardized_temperature, :standardized_precipitation])
unique2nonunique = Dict([Vector{Float64}(group[1, [:standardized_temperature, :standardized_precipitation]]) => group for group in unique_clusters])

# Prepare dataset for MultivariateNormalCRP
# dataset = 1.0 * collect(eachrow(hcat(unique_obs.standardized_temperature, unique_obs.standardized_precipitation)))
dataset = collect.(zip(unique_obs.standardized_temperature, unique_obs.standardized_precipitation))



#######################################

# aedes_albopictus_dataset = deserialize("data/aedes_albopictus_dataset.ser")
# polyammus_icarus_dataset = deserialize("data/polyammus_icarus_dataset.ser")

chain_input = aedes_albopictus_dataset["chain_input"]

chain_state = initiate_chain(chain_input)
advance_chain!(chain_state, nb_steps=1000, nb_splitmerge=50, splitmerge_t=2, nb_gibbs=1)

#######################################

obs_clusters = [reduce(vcat, [unique2nonunique[point] for point in cluster]) for cluster in chain_state.map_clusters]
sorted_clusters = sort([(i, length(c)) for (i, c) in enumerate(chain_state.map_clusters)], by=x -> x[2], rev=true)

fig = Figure(resolution=(2000, 1200));
ga = GeoAxis(
    fig[1, 1]; # any cell of the figure's layout
    dest = "+proj=wintri", # the CRS in which you want to plot
    coastlines = true # plot coastlines from Natural Earth, as a reference.
);

for (ci, sizeci) in sorted_clusters
    _cluster = obs_clusters[ci]
    GeoMakie.scatter!(ga, _cluster.decimalLongitude, _cluster.decimalLatitude, markersize=4, label="$sizeci")
end

display(fig)

#######################################

obs_clusters = [reduce(vcat, [unique2nonunique[point] for point in cluster]) for cluster in chain.map_clusters]
sort!(obs_clusters, by=c -> nrow(c), rev=true)

# latlims = (minimum(cluster.decimalLatitude), maximum(cluster.decimalLatitude))
# lonlims = (minimum(cluster.decimalLongitude), maximum(cluster.decimalLongitude))

fig = Figure(resolution=(2400, 1600));
ga = GeoAxis(
    fig[1, 1]; # any cell of the figure's layout
    dest = "+proj=wintri", # the CRS in which you want to plot
    coastlines = true, # plot coastlines from Natural Earth, as a reference.
    # lonlims=lonlims, latlims=latlims
);
for (i, cluster, ms) in zip(1:7, obs_clusters[1:7],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    GeoMakie.scatter!(ga, cluster.decimalLongitude, cluster.decimalLatitude, 
    markersize=5)
end
display(fig)

#######################################

sorted_clusters = sort(collect(chain.map_clusters), by=c -> length(c), rev=true)

for cluster in sorted_clusters[1:7]
    singleton_cluster = MultivariateNormalCRP.Cluster[cluster]
    p = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
    fmt=:png, bins=60, legend=:false, axis=:false);

    covellipses!(singleton_cluster, chain.map_hyperparams, 
        lowest_weight=0, n_std=2,
        scalematrix=scale_matrix, offset=offset,
        legend=:false, fillcolor=:false, fillalpha=0.0, 
        linewidth=4, linealpha=1.0, linecolor=:darkturquoise);
    StatsPlots.xlabel!("Temperature (ᵒC)", fontsize=100);
    StatsPlots.ylabel!("Precipitation (mm/year)");
    display(p)
    # save("cluster_ellipse_size$(length(cluster)).png", p)
end

#######################################


sorted_clusters = sort(collect(chain.map_clusters), by=c -> length(c), rev=true)

cht = [tp[1] for c in chain.clusters for tp in c]
chp = [tp[2] for c in chain.clusters for tp in c]
p = histogram2d(cht, chp, 
    fmt=:png, bins=60, legend=:false, axis=:false);
# p = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
#     fmt=:png, bins=60, legend=:false, axis=:false);

covellipses!(sorted_clusters[1:end], chain.map_hyperparams, 
    lowest_weight=0, n_std=2.0,
    scalematrix=nothing, offset=nothing,
    # scalematrix=scale_matrix, offset=offset,
    legend=:false, fillcolor=:false, fillalpha=0.0, 
    linewidth=2, linealpha=1.0, linecolor=:darkturquoise);
StatsPlots.xlabel!("Temperature (ᵒC)", fontsize=100);
StatsPlots.ylabel!("Precipitation (mm/year)");
display(p)
# save("histogram_map_ellipses.png", p)

#######################################

acosh
obs_clusters = [reduce(vcat, [unique2nonunique[point] for point in cluster]) for cluster in chain_state.map_clusters]
sorted_clusters = sort([(i, length(c)) for (i, c) in enumerate(chain_state.map_clusters)], by=x -> x[2])

fig = plot(legend=:true)
for (ci, sizeci) in sorted_clusters[end:end]
    _cluster = obs_clusters[ci]
    StatsPlots.scatter!(_cluster.temperature, _cluster.precipitation, 
    markersize=4, markerstrokewidth=0, label="$sizeci")
end
xflip!(true)
Plots.xlims!(-10, 35)
display(fig)


#######################################

p = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
                fmt=:png, bins=60, legend=:false, axis=:false, 
                size=(1200, 800), thickness_scaling=2.0);
covellipses!(chain.map_clusters, chain.map_hyperparams, 
       lowest_weight=10, n_std=2, 
       scalematrix=scale_matrix, offset=offset,
       legend=:false, fillcolor=:false, fillalpha=0.0, 
       linewidth=2, linealpha=1.0, linecolor=:darkturquoise);
StatsPlots.xlabel!("Temperature (ᵒC)");
StatsPlots.ylabel!("Precipitation (mm/year)");
display(p)

#######################################

_clusters = chain_state.map_clusters
_hyperparams = chain_state.map_hyperparams

_mu0, _lambda0, _psi0, _nu0 = _hyperparams.mu, _hyperparams.lambda, _hyperparams.psi, _hyperparams.nu
_d = size(_mu0, 1)

updated_cluster_mus = []
updated_cluster_psis = []
updated_cluster_sigma_modes = []
cluster_weights = []
total_weight = sum(length(c) for c in _clusters)

for c in _clusters
    mu_c, lambda_c, psi_c, nu_c = MultivariateNormalCRP.updated_niw_hyperparams(c, _mu0, _lambda0, _psi0, _nu0)
    push!(updated_cluster_mus, mu_c)
    push!(updated_cluster_psis, psi_c)
    push!(cluster_weights, length(c) / total_weight)
    sigma_c = psi_c / (nu_c + _d + 1)
    push!(updated_cluster_sigma_modes, sigma_c)
    
end
n = length(updated_cluster_mus)
distance_matrix = zeros(n, n);
for i in 1:n
    distance_matrix[i, i] = 0.0
    for j in 1:i-1
        distance_matrix[i, j] = wasserstein1_distance_bound(updated_cluster_mus[i], updated_cluster_sigma_modes[i], updated_cluster_mus[j], updated_cluster_sigma_modes[j])
        # distance_matrix[i, j] = wasserstein2_distance(updated_cluster_mus[i], updated_cluster_sigma_modes[i], updated_cluster_mus[j], updated_cluster_sigma_modes[j])
        # distance_matrix[i, j] = symm_kl_divergence(updated_cluster_mus[i], updated_cluster_sigma_modes[i], updated_cluster_mus[j], updated_cluster_sigma_modes[j])        
        # distance_matrix[i, j] = bhattacharyya_distance(updated_cluster_mus[i], updated_cluster_sigma_modes[i], updated_cluster_mus[j], updated_cluster_sigma_modes[j])        
        # distance_matrix[i, j] = hellinger_distance(updated_cluster_mus[i], updated_cluster_sigma_modes[i], updated_cluster_mus[j], updated_cluster_sigma_modes[j])
        # distance_matrix[i, j] = hellinger_distance_weighted(cluster_weights[i], updated_cluster_mus[i], updated_cluster_sigma_modes[i], cluster_weights[j], updated_cluster_mus[j], updated_cluster_sigma_modes[j])

        distance_matrix[j, i] = distance_matrix[i, j]
    end
end
distance_matrix

# p = plot_pi_state_2d(_clusters, axis=:none)
# p = histogram2d((obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature), (obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation), fmt=:png, bins=100, legend=:false, axis=:false)
# x = [m[1] for m in updated_cluster_mus]
# y = [m[2] for m in updated_cluster_mus]
# # scatter!(x, y, markershape=:star, markersize=5)
# weight_lower_tresh = 0
# for (mu, sigma, weight) in zip(updated_cluster_mus, updated_cluster_sigma_modes, cluster_weights)
#     if weight >= weight_lower_tresh
#         StatsPlots.covellipse!(mu, sigma, n_std=2, 
#         legend=:false, fillcolor=:false, fillalpha=0.0, 
#         linewidth=2, linealpha=1.0, linecolor=:darkturquoise)
#     end
# end
# display(p)

# mupsis = [([x.re, x.im], local_average_covariance([x.re, x.im], _clusters, _hyperparams)) for x in exp.(1im * range(0.0, 2pi, 20)[1:end-1])]
# for (mu, psi) in mupsis
#     covellipse!(mu, psi, fillcolor=:false, fillalpha=0.0, legend=:false, linecolor=:white, linewidth=2, linealpha=0.5)
# end
# display(p)

# text = [i for (i, _) in enumerate(_clusters)]
# text = ["$(round(cluster_weights[i]; digits=2, base=10))" for (i, _) in enumerate(_clusters)]
# annotate!(x, y, text, annotationfontsize=8)
# tresh = 1.5
# for i in 1:n
#     for j in 1:i-1
#         if distance_matrix[i, j] < tresh
#             plot!([x[i], x[j]], [y[i], y[j]], 
#             linewidth=2, linecolor=:goldenrod2)
#         end
#     end
# end

#######################################

pds = ripserer(distance_matrix, alg=:homology)
plot(pds)
p = histogram2d(unique_obs.standardized_temperature, unique_obs.standardized_precipitation, bins=60, legend=:false, axis=:false)
# p = plot_clusters_2d(_clusters, legend=false)

c = ColorSchemes.magma[200];
birth_threshold = 4.2
death_threshold = 5
# plot_cluster_covellipses!(_clusters, _hyperparams,
# lowest_weight=10,
# linewidth=2, linecolor=c, fillalpha=0.0, linealpha=1.0)
persistence_intervals = pds[2]
for h in persistence_intervals
    if death(h) <= death_threshold
        StatsPlots.plot!(p, death_simplex(h), 
        [Tuple(v) for v in updated_cluster_mus], 
        linewidth=3, linealpha=1.0, linecolor=c)
    end
end
for h in persistence_intervals
    if birth(h) <= birth_threshold
        StatsPlots.plot!(birth_simplex(h), 
        [Tuple(v) for v in updated_cluster_mus], 
        linewidth=2, linealpha=1.0, 
        markercolor=:black, markersize=2,
        markerstrokewidth=2, markerstrokecolor=c)
    end
end
display(p)

#######################################

_clusters = chain_state.map_clusters
_hyperparams = chain_state.map_hyperparams

p = histogram2d(unique_obs.standardized_temperature, unique_obs.standardized_precipitation,
 fmt=:png, bins=100, legend=:false, axis=:false)

covellipses!(_clusters, _hyperparams, 
lowest_weight=10, n_std=2,
legend=:false, fillcolor=:false, fillalpha=0.0, 
linewidth=1, linealpha=1.0, linecolor=:yellowgreen)

display(p)

#######################################

mupsis = [
    ([x.re, x.im], 
    local_average_covariance([x.re, x.im],
    _clusters, _hyperparams)) 
    for x in exp.(1im * range(0.0, 2pi, 20)[1:end-1])
    ]

for (mu, psi) in mupsis
    covellipse!(mu, psi, fillcolor=:false, fillalpha=0.0, legend=:false, linecolor=c, linewidth=2, linealpha=0.5)
end

display(p)

#########################################

rts = round.(unique_obs.standardized_temperature, digits=1)
rps = round.(unique_obs.standardized_precipitation, digits=1)
rtps = sort(collect(Set(collect.(zip(rts, rps)))))

covprec_averages = [local_average_covprec(tp, chain.map_clusters, chain.map_hyperparams) for tp in rtps]
covs = [cpa.average_covariance for cpa in covprec_averages]
precs = [cpa.average_precision for cpa in covprec_averages]

p = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
           fmt=:png, bins=60, legend=:false, axis=:false, 
           thickness_scaling=2.0, size=(1200, 800));

StatsPlots.scatter!([t / scale_matrix[1, 1] + offset[1] for (t, p) in rtps], 
                    [p / scale_matrix[2, 2] + offset[2] for (t, p) in rtps], 
markersize=[15 * (p[1, 2] > 0 ? -p[1, 2]/sqrt(p[1, 1] * p[2, 2]) : 0.0) for p in precs],
markerstrokewidth=0, markeralpha=0.6);

StatsPlots.scatter!([t / scale_matrix[1, 1] + offset[1] for (t, p) in rtps], 
                    [p / scale_matrix[2, 2] + offset[2] for (t, p) in rtps], 
markersize=[15 * (p[1, 2] <= 0 ? p[1, 2]/sqrt(p[1, 1] * p[2, 2]) : 0.0) for p in precs],
markerstrokewidth=0, markeralpha=0.6);

StatsPlots.xlabel!("Temperature (ᵒC)", fontsize=100);
StatsPlots.ylabel!("Precipitation (mm/year)");

display(p)

####################################

xls, yls = xlims(p), ylims(p);

h = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
                  fmt=:png, bins=60, legend=:false, 
                  axis=:false, 
                  thickness_scaling=2.0,
                  size=(1200, 800));
StatsPlots.xlabel!("Temperature (ᵒC)", fontsize=100);
StatsPlots.ylabel!("Precipitation (mm/year)");
StatsPlots.xlims!(xls);
StatsPlots.ylims!(yls);
display(h)

for (xy, cov) in zip(xs, covs)
    covellipse!(xy, cov, n_std=0.5, 
    fillcolor=:false, fillalpha=0.0, 
    legend=:false, linecolor=:darkturquoise, 
    linewidth=1, linealpha=1.0)
end

#####################

# rts = round.(unique_obs.standardized_temperature, base=20, digits=0)
# rps = round.(unique_obs.standardized_precipitation, base=20, digits=0)
# rtps = sort(collect(Set(collect.(zip(rts, rps)))))

# covprec_averages = [local_average_covprec(tp, chain_state.map_clusters, chain_state.map_hyperparams) for tp in rtps]
# covs = [cpa.average_covariance for cpa in covprec_averages]
# precs = [cpa.average_precision for cpa in covprec_averages]

# p = histogram2d(unique_obs.temperature, unique_obs.precipitation, 
#             fmt=:png, bins=60, legend=:false, axis=:false, 
#             thickness_scaling=2.0, size=(1200, 800));

# for (tp, cov) in zip(rtps, covs)
#     covellipse!(
#     inv(scale_matrix) * tp .+ offset, 
#     inv(scale_matrix) * cov * inv(scale_matrix)',
#     n_std=0.7, 
#     fillcolor=:false, fillalpha=0.0, 
#     legend=:false, linecolor=:darkturquoise, 
#     linewidth=1, linealpha=1.0)
# end

# StatsPlots.xlabel!("Temperature (ᵒC)", fontsize=100);
# StatsPlots.ylabel!("Precipitation (mm/year)");
# StatsPlots.xlims!(xls);
# StatsPlots.ylims!(yls);
# display(p)

#############################################

obs_clusters = [reduce(vcat, [unique2nonunique[point] for point in cluster]) for cluster in chain_state.map_clusters]
sorted_clusters = sort([(i, length(c)) for (i, c) in enumerate(chain_state.map_clusters)], by=x -> x[2], rev=true)

fig = Figure(resolution=(2000, 1200));
ga = GeoAxis(
    fig[1, 1]; # any cell of the figure's layout
    dest = "+proj=wintri", # the CRS in which you want to plot
    coastlines = true # plot coastlines from Natural Earth, as a reference.
);

for (ci, sizeci) in sorted_clusters
    _cluster = obs_clusters[ci]
    GeoMakie.scatter!(ga, _cluster.decimalLongitude, _cluster.decimalLatitude, markersize=4, label="$sizeci")
end

display(fig)