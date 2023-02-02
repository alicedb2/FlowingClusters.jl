mutable struct Cluster
    elements::Set{Vector{Float64}}
    sum_x::Vector{Float64}
    sum_xx::Matrix{Float64}

    # This is not thread-safe
    mu_c_volatile::Vector{Float64}
    psi_c_volatile::Matrix{Float64}
end

function Cluster(d::Int64)
    return Cluster(Set{Vector{Float64}}(), zeros(Float64, d), zeros(Float64, d, d), Array{Float64}(undef, d), Array{Float64}(undef, d, d))
end

function Cluster(elements::Set{Vector{Float64}})
    return Cluster(collect(elements))
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

    sum_x = zeros(Float64, d)
    @inbounds for i in 1:d
        for x in elements
            sum_x[i] += x[i]
        end
    end

    sum_xx = zeros(Float64, d, d)
    @inbounds for j in 1:d
        @inbounds for i in 1:j
            for x in elements
                sum_xx[i, j] += x[i] * x[j]
            end
            sum_xx[j, i] = sum_xx[i, j]
        end
    end

    elements = Set{Vector{Float64}}(elements)

    return Cluster(elements, sum_x, sum_xx, Array{Float64}(undef, d), Array{Float64}(undef, d, d))

end

function pop!(cluster::Cluster, x::Vector{Float64})
    x = pop!(cluster.elements, x)
    # cluster.sum_xx -= x * x'
    # cluster.sum_x -= x
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
    # cluster.sum_xx -= x * x'
    # cluster.sum_x -= x
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
        # cluster.sum_xx += x * x'
        # cluster.sum_x += x
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

    # we don't know if they have elements in common
    # so we need to recompute sum_x and sum_xx
    # maybe we'll accelerate this by
    # tracking the intersection

    # new_sum_xx = sum([x * x' for x in elements])
    # new_sum_x = sum(elements)

    new_sum_x = zeros(Float64, d)
    @inbounds for i in 1:d
        @inbounds for x in elements
            new_sum_x[i] += x[i]
        end
    end

    new_sum_xx = zeros(Float64, d, d)
    @inbounds for j in 1:d
        @inbounds for i in 1:j
            @inbounds for x in elements
                new_sum_xx[i, j] += x[i] * x[j]
            end
            new_sum_xx[j, i] = new_sum_xx[i, j]
        end
    end
    
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

function first(cluster::Cluster)
    return first(cluster.elements)
end