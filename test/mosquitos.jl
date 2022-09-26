using Revise
using MultivariateNormalCRP
using SimpleSDMLayers
using CSV
using DataFrames
using Random
using StatsBase
using StatsPlots
using Ripserer
using LinearAlgebra
using ColorSchemes
using Serialization

temperature, precipitation = SimpleSDMPredictor(WorldClim, BioClim, [1, 12])

obs = DataFrame(CSV.File("/Users/alice/Documents/Postdoc/MultivariateNormalCRP/test/aedes_albopictus.csv", delim="\t"))

# Remove rows with missing latitude and/or longitude
# and then with missing temperature and/or precipitation
filter!(row -> !ismissing(row.decimalLatitude) && !ismissing(row.decimalLongitude), obs)
obs.temperature = temperature[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
obs.precipitation = precipitation[obs, latitude=:decimalLatitude, longitude=:decimalLongitude]
filter!(row -> !isnothing(row.temperature) && !isnothing(row.precipitation), obs)
obs.temperature = Vector{Float64}(obs.temperature)
obs.precipitation = Vector{Float64}(obs.precipitation)

obs.normalized_temperature = (obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature)
obs.normalized_precipitation = (obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation)

# Keep records with unique temps/precs
# (although they'll naturally be dropped
# because clusters are Sets)

unique_obs = unique(obs, [:normalized_temperature, :normalized_precipitation])
unique_clusters = groupby(obs, [:normalized_temperature, :normalized_precipitation])
unique2nonunique = Dict([Vector{Float64}(group[1, [:normalized_temperature, :normalized_precipitation]]) => group for group in unique_clusters])

# Prepare dataset for MultivariateNormalCRP
dataset = 1.0 * collect(eachrow(hcat(unique_obs.normalized_temperature, unique_obs.normalized_precipitation)))
shuffled_dataset = dataset[randperm(length(dataset))]

# draw without replacement
nb_subsamples = 1000
subsampled_dataset = shuffled_dataset[1:nb_subsamples]

#######################################

chain_state = initiate_chain(subsampled_dataset)
advance_chain!(chain_state, nb_steps=100, nb_splitmerge=50, splitmerge_t=2, nb_gibbs=1)

#######################################

obs_clusters = [reduce(vcat, [unique2nonunique[point] for point in cluster]) for cluster in chain_state.map_clusters]
sorted_clusters = sort([(i, length(c)) for (i, c) in enumerate(chain_state.map_clusters)], by=x -> x[2])

fig = Figure(resolution=(2000, 1500));
ga = GeoAxis(
    fig[1, 1]; # any cell of the figure's layout
    dest = "+proj=wintri", # the CRS in which you want to plot
    coastlines = true # plot coastlines from Natural Earth, as a reference.
);

# _cluster = obs_clusters[sorted_clusters[end-4][1]]
# GeoMakie.scatter!(ga, _cluster.decimalLongitude, _cluster.decimalLatitude, markersize=4)

for (ci, _) in sorted_clusters[end:-1:end-10]
    _cluster = obs_clusters[ci]
    GeoMakie.scatter!(ga, _cluster.decimalLongitude, _cluster.decimalLatitude, markersize=4)
end
display(fig)
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
p = histogram2d((obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature), (obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation), fmt=:png, bins=100, legend=:false, axis=:false)
# p = plot_clusters_2d(_clusters, legend=false)

c = ColorSchemes.magma[200];
birth_threshold = 2
death_threshold = 0.0
# plot_cluster_covellipses!(_clusters, _hyperparams,
# lowest_weight=10,
# linewidth=2, linecolor=c, fillalpha=0.0, linealpha=1.0)
persistence_intervals = pds[2]
for h in persistence_intervals
    if death(h) <= death_threshold
        plot!(death_simplex(h), 
        [Tuple(v) for v in updated_cluster_mus], 
        linewidth=3, linealpha=1.0, linecolor=c)
    end
end
for h in persistence_intervals
    if birth(h) <= birth_threshold
        plot!(birth_simplex(h), 
        [Tuple(v) for v in updated_cluster_mus], 
        linewidth=2, linealpha=1.0, 
        markercolor=:black, markersize=2,
        markerstrokewidth=2, markerstrokecolor=c)
    end
end
display(p)

#######################################

p = histogram2d((obs.temperature .- mean(obs.temperature)) ./ std(obs.temperature), 
(obs.precipitation .- mean(obs.precipitation)) ./ std(obs.precipitation),
 fmt=:png, bins=100, legend=:false, axis=:false)

plot_clusters_covellipses!(_clusters, _hyperparams, 
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
