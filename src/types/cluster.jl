mutable struct Cluster
    elements::Set{Vector{Float64}}
    sum_x::Vector{Float64}
    sum_xx::Matrix{Float64}

    # This is not thread-safe
    mu_c_volatile::Vector{Float64}
    psi_c_volatile::Matrix{Float64}
end

Base.show(io::IO, cluster::Cluster) = print("Cluster($(length(cluster)))")
Base.show(io::IO, ::MIME"text/plain", cluster::Cluster) = print("Cluster($(length(cluster)))")
Base.show(io::IO, ::MIME"text/plain", arr::AbstractArray{Cluster}) = print(io, arr)

function Cluster(d::Int64)
    return Cluster(Set{Vector{Float64}}(), zeros(Float64, d), zeros(Float64, d, d), Array{Float64}(undef, d), Array{Float64}(undef, d, d))
end

function Cluster(elements::Set{Vector{Float64}})
    return Cluster(collect(elements))
end

function Cluster(matrix::Matrix{Float64})
    return Cluster(collect.(eachcol(matrix)))
end

function Cluster(elements::Vector{Vector{Float64}})
    
    # We need at least one element
    # to know the dimension so that
    # we can initialize sum_x and sum_xx
    @assert !isempty(elements)
    
    # d = size(first(elements), 1)
    # @assert all([size(element, 1) == d for element in elements])
    
    # sum_x = sum(elements)
    # sum_xx = sum([x * x' for x in elements])

    d = length(first(elements))
    @assert all(length.(elements) .== d) "All elements must be of dimension $d"

    sum_x, sum_xx = calculate_sums(elements)

    elements = Set{Vector{Float64}}(elements)

    return Cluster(elements, sum_x, sum_xx, Array{Float64}(undef, d), Array{Float64}(undef, d, d))

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


Base.Matrix(cluster::Cluster) = reduce(hcat, elements(cluster), init=zeros(Float64, size(cluster.sum_x, 1), 0))
Base.Matrix(cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}}) = reduce(hcat, [base2original[el] for el in cluster], init=zeros(Float64, size(cluster.sum_x, 1), 0))

function pop!(cluster::Cluster, x::Vector{Float64})
    x = pop!(cluster.elements, x)

    d = length(x)
    @inbounds for i in 1:d
        cluster.sum_x[i] -= x[i]
    end
    
    @inbounds for j in 1:d
        @inbounds for i in 1:j
            cluster.sum_xx[i, j] -= x[i] * x[j]
            cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
        end
    end


    return x
end

function pop!(cluster::Cluster)
    x = pop!(cluster.elements)

    d = length(x)

    @inbounds for i in 1:d
        cluster.sum_x[i] -= x[i]
    end
    
    @inbounds for j in 1:d
        @inbounds for i in 1:j
            cluster.sum_xx[i, j] -= x[i] * x[j]
            cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
        end
    end


    return x
end

function pop!(clusters::Vector{Cluster}, x::Vector{Float64}; delete_empty::Bool=true)
    for (ci, cluster) in enumerate(clusters)
        if x in cluster
            x = pop!(cluster, x)
            if delete_empty && isempty(cluster)
                deleteat!(clusters, ci)
            end
    
            return x
        end
    end
    error(KeyError, ": key $x not found")
end

function push!(cluster::Cluster, x::Vector{Float64})
    if !(x in cluster.elements)

        d = length(x)
        @inbounds for i in 1:d
            cluster.sum_x[i] += x[i]
        end

        @inbounds for j in 1:d
            @inbounds for i in 1:j
                cluster.sum_xx[i, j] += x[i] * x[j]
                cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
            end
        end
        push!(cluster.elements, x)
    end


    return cluster
end

function delete!(cluster::Cluster, x::Vector{Float64})
    if x in cluster.elements
        pop!(cluster, x)

    end
    return cluster
end

function delete!(clusters::Vector{Cluster}, x::Vector{Float64}; delete_empty::Bool=true)
    for (ci, cluster) in enumerate(clusters)        
        delete!(cluster, x)
        if delete_empty && isempty(cluster)
            deleteat!(clusters, ci)
        end
    end
    return clusters
end

function empty!(cluster::Cluster)
    empty!(cluster.elements)
    cluster.sum_x = zeros(size(cluster.sum_x))
    cluster.sum_xx = zeros(size(cluster.sum_xx))
    return cluster
end    

function union(cluster1::Cluster, cluster2::Cluster)

    @assert length(cluster1.sum_x) == length(cluster2.sum_x)
    
    d = length(cluster1.sum_x)

    elements = union(cluster1.elements, cluster2.elements)

    new_sum_x, new_sum_xx = calculate_sums(elements)
    
    return Cluster(elements, new_sum_x, new_sum_xx, Array{Float64}(undef, d), Array{Float64}(undef, d, d))
end

function isempty(cluster::Cluster)
    return isempty(cluster.elements)
end

function length(cluster::Cluster)
    return length(cluster.elements)
end

function in(element::Vector{Float64}, cluster::Cluster)
    return in(element, cluster.elements)
end

function in(element::Vector{Float64}, clusters::Vector{Cluster})
    return any([element in cluster for cluster in clusters])
end

function find(element::Vector{Float64}, clusters::Vector{Cluster})
    for (i, cl) in enumerate(clusters)
        if in(element, cl)
            return (cl, i)
        end
    end
    error(KeyError, ": key $element not found")
end

function iterate(cluster::Cluster)
    return iterate(cluster.elements)
end

function iterate(cluster::Cluster, state)
    return iterate(cluster.elements, state)
end


function deepcopy(cluster::Cluster)
    return Cluster(deepcopy(cluster.elements), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
end

function copy(cluster::Cluster)
    return Cluster(copy(cluster.elements), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
end

function copy(clusters::Vector{Cluster})
    return Cluster[copy(cl) for cl in clusters]
end

function dims_to_proj(dims::Vector{Int64}, d::Int64)
    return diagm(ones(d))[dims, :]
end

function project_cluster(cluster::Cluster, proj::Matrix{Float64})
    return Cluster([proj * x for x in cluster])
end

function project_clusters(clusters::Vector{Cluster}, proj::Matrix{Float64})
    return Cluster[project_cluster(cluster, proj) for cluster in clusters]
end

function project_clusters(clusters::Vector{Cluster}, dims::Vector{Int64})
    el = pop!(first(clusters))
    d = length(el)
    push!(first(clusters), el)
    return project_clusters(clusters, dims_to_proj(dims, d))
end

function elements(clusters::Vector{Cluster})
    return Vector{Float64}[x for cluster in clusters for x in cluster]
end

function elements(cluster::Cluster)
    return collect(cluster.elements)
end

function first(cluster::Cluster)
    return first(cluster.elements)
end

function calculate_sums(cluster::Union{Cluster, Set{Vector{Float64}}, Vector{Vector{Float64}}})
    d = length(first(cluster))
    @assert all(length.(cluster) .== d) "All elements must be of dimension $d"

    sum_x = zeros(Float64, d)
    @inbounds for i in 1:d
        for x in cluster
            sum_x[i] += x[i]
        end
    end

    sum_xx = zeros(Float64, d, d)
    @inbounds for j in 1:d
        @inbounds for i in 1:j
            for x in cluster
                sum_xx[i, j] += x[i] * x[j]
            end
            sum_xx[j, i] = sum_xx[i, j]
        end
    end

    return sum_x, sum_xx
end

function recalculate_sums!(cluster::Cluster)
    cluster.sum_x, cluster.sum_xx = calculate_sums(cluster)
    return cluster
end