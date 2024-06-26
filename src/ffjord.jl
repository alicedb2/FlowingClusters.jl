function prepare_ffjord_training_data(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams, data2orig::Dict{Vector{Float64}, Vector{Float64}}; training_fraction=0.5)

    training_clusters = deepcopy(clusters)

    training_data = Vector{Float64}[]

    for cluster in training_clusters, el in cluster
        if rand() < training_fraction
            push!(training_data, pop!(cluster, el))
        end
    end
    filter!(!isempty, training_clusters)

    training_data = reduce(hcat, [data2orig[el] for el in training_data],
                           init=zeros(Float64, dimension(hyperparams), 0))

    return training_data, training_clusters

end

Cluster(cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}}) = Cluster([base2original[el] for el in cluster])

function realspace_cluster(::Type{Cluster}, cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}})
    return Cluster([base2original[el] for el in cluster])
end

function realspace_cluster(::Type{Matrix}, cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}})
    return reduce(hcat, [base2original[el] for el in cluster])
end

function realspace_cluster(cluster::Cluster, hyperparams::MNCRPHyperparams; ffjord_model=nothing)
    if hyperparams.nn !== nothing
        if ffjord_model === nothing
            ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
        end
        # Transport training data from base space to real/environmental space
        clustermat = Matrix{Float64}(DiffEqFlux.__backward_ffjord(ffjord_model, Matrix(cluster), hyperparams.nn_params, hyperparams.nn_state))
        return Cluster(clustermat)
    else
        return deepcopy(cluster)
    end
end

function realspace_clusters(T::Type, clusters::Vector{Cluster}, base2original::Dict{Vector{Float64}, Vector{Float64}})
    return [realspace_cluster(T, cluster, base2original) for cluster in clusters]
end

function realspace_clusters(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
    if hyperparams.nn !== nothing
        ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
        return Cluster[realspace_cluster(cluster, hyperparams, ffjord_model=ffjord_model) for cluster in clusters]
    else
        return deepcopy(clusters)
    end
end