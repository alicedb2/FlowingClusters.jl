abstract type AbstractCluster{D, T} end

struct SetCluster{D, T} <: AbstractCluster{D, T}
    elements::Set{Vector{T}}
    b2o::Dict{Vector{T}, Vector{T}}
    sum_x::Vector{T}
    sum_xx::Matrix{T}

    # Scratch arrays to avoid allocations
    mu_c_volatile::Vector{T}
    psi_c_volatile::Matrix{T}

    SetCluster{D, T}(elements::Set{Vector{T}}, b2o::AbstractDict{Vector{T}, Vector{T}}, sum_x::Vector{T}, sum_xx::Matrix{T}, mu_c_volatile::Vector{T}, psi_c_volatile::Matrix{T}) where {D, T} = new{D, T}(elements, b2o, sum_x, sum_xx, mu_c_volatile, psi_c_volatile)

    function SetCluster(elements::AbstractVector{Vector{T}}, b2o::AbstractDict{Vector{T}, Vector{T}}) where {T}
        D = size(first(elements), 1)
        @assert D == size(first(keys(b2o)), 1) == size(first(values(b2o)), 1)
        sum_x = sum(elements)
        sum_xx = sum([x * x' for x in elements])
        mucvol = Vector{T}(undef, D)
        psicvol = Matrix{T}(undef, D, D)
        return new{D, T}(Set{Vector{T}}(elements), b2o, sum_x, sum_xx, mucvol, psicvol)
    end

    function SetCluster(elements::AbstractMatrix{T}, b2o::AbstractDict{Vector{T}, Vector{T}}) where {T}
        return SetCluster(collect.(eachcol(elements)), b2o)
    end

    function SetCluster(b2o::AbstractDict{Vector{T}, Vector{T}}) where {T}
        D = size(first(keys(b2o)), 1)
        return new{D, T}(Set{Vector{T}}(), b2o, zeros(T, D), zeros(T, D, D), Array{T}(undef, D), Array{T}(undef, D, D))
    end

    function SetCluster(D::Int, b2o::AbstractDict{Vector{T}, Vector{T}}) where {T}
        return new{D, T}(Set{Vector{T}}(), b2o, zeros(T, D), zeros(T, D, D), Array{T}(undef, D), Array{T}(undef, D, D))
    end

end

const B = 1
const O = 2

struct BitCluster{D, T} <: AbstractCluster{D, T}
    mask::BitVector
    b2o::Array{T, 3}
    sum_x::Vector{T}
    sum_xx::Matrix{T}

    # Scratch arrays to avoid allocations
    mu_c_volatile::Vector{T}
    psi_c_volatile::Matrix{T}

    function BitCluster(bitarray::BitVector, b2o::AbstractArray{T, 3}) where {T}
        @assert ndims(b2o) == 3
        d = size(b2o, 1)
        sum_x = dropdims(sum(b2o[:, bitarray, B], dims=2), dims=2)
        sum_xx = sum([x * x' for x in eachcol(b2o[:, bitarray, B])])
        mucvol = Vector{T}(undef, d)
        psicvol = Matrix{T}(undef, d, d)
        return new{d, T}(bitarray, b2o, sum_x, sum_xx, mucvol, psicvol)
    end

    function BitCluster(b2o::AbstractArray{T, 3}) where {T}
        D, N = size(b2o, 1), size(b2o, 2)
        return new{D, T}(falses(N), b2o, zeros(T, D), zeros(T, D, D), Array{T}(undef, D), Array{T}(undef, D, D))
    end
end

function Base.BitVector(N::Int, v::Vector{Int})::BitVector
    bv = falses(N)
    bv[v] .= true
    return bv
end

isvalid(clusters::Vector{BitCluster}) = isvalid([bc.mask for bc in clusters])
isvalid(bitvectors::Vector{BitVector}) = !any(reduce(((av, ov), x) -> (av .| (ov .& x), ov .| x), bitvectors, init=(zeros(Bool, length(first(bitvectors))), zeros(Bool, length(first(bitvectors)))))[1])
iscomplete(clusters::Vector{BitCluster}) = iscomplete([bc.mask for bc in clusters])
iscomplete(bitvectors::Vector{BitVector}) = all(reduce((ov, x) -> ov .| x, bitvectors))

Base.show(io::IO, cluster::AbstractCluster) = print(io, "$(typeof(cluster).name.wrapper){$(typeof(cluster).parameters[1]), $(typeof(cluster).parameters[2])}($(length(cluster)))")
# function realspace_cluster(::Type{Cluster}, cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}})
#     return Cluster([base2original[el] for el in cluster])
# end

# function realspace_cluster(::Type{Matrix}, cluster::Cluster, base2original::Dict{Vector{Float64}, Vector{Float64}})
#     return reduce(hcat, [base2original[el] for el in cluster])
# end

# function realspace_cluster(::Type{Cluster}, cluster::Cluster, hyperparams::FCHyperparamsFFJORD; ffjord_model=nothing)
    
# end

# function realspace_cluster(cluster::Cluster, hyperparams::FCHyperparamsFFJORD; ffjord_model=nothing) where T
#     if hyperparams.nn !== nothing
#         if ffjord_model === nothing
#             ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
#         end
#         # Transport training data from base space to real/environmental space
#         clustermat = Matrix{Float64}(DiffEqFlux.__backward_ffjord(ffjord_model, Matrix(cluster), hyperparams.nn_params, hyperparams.nn_state))
#         return Cluster(clustermat)
#     else
#         return deepcopy(cluster)
#     end
# end

# function realspace_clusters(T::Type, clusters::Vector{Cluster}, base2original::Dict{Vector{Float64}, Vector{Float64}})
#     return [realspace_cluster(T, cluster, base2original) for cluster in clusters]
# end

# function realspace_clusters(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams)
#     if hyperparams.nn !== nothing
#         ffjord_model = FFJORD(hyperparams.nn, (0.0f0, 1.0f0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
#         return Cluster[realspace_cluster(cluster, hyperparams, ffjord_model=ffjord_model) for cluster in clusters]
#     else
#         return deepcopy(clusters)
#     end
# end


# function pop!(cluster::Cluster, x::Vector{Float64})
#     x = pop!(cluster.elements, x)

#     d = length(x)
#     @inbounds for i in 1:d
#         cluster.sum_x[i] -= x[i]
#     end
    
#     @inbounds for j in 1:d
#         @inbounds for i in 1:j
#             cluster.sum_xx[i, j] -= x[i] * x[j]
#             cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
#         end
#     end


#     return x
# end

# function pop!(cluster::Cluster)
#     x = pop!(cluster.elements)

#     d = length(x)

#     @inbounds for i in 1:d
#         cluster.sum_x[i] -= x[i]
#     end
    
#     @inbounds for j in 1:d
#         @inbounds for i in 1:j
#             cluster.sum_xx[i, j] -= x[i] * x[j]
#             cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
#         end
#     end


#     return x
# end

# function pop!(clusters::Vector{Cluster}, x::Vector{Float64}; delete_empty::Bool=true)
#     for (ci, cluster) in enumerate(clusters)
#         if x in cluster
#             x = pop!(cluster, x)
#             if delete_empty && isempty(cluster)
#                 deleteat!(clusters, ci)
#             end
    
#             return x
#         end
#     end
#     error(KeyError, ": key $x not found")
# end

# function push!(cluster::Cluster, x::Vector{Float64})
#     if !(x in cluster.elements)

#         d = length(x)
#         @inbounds for i in 1:d
#             cluster.sum_x[i] += x[i]
#         end

#         @inbounds for j in 1:d
#             @inbounds for i in 1:j
#                 cluster.sum_xx[i, j] += x[i] * x[j]
#                 cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
#             end
#         end
#         push!(cluster.elements, x)
#     end


#     return cluster
# end

# function delete!(cluster::Cluster, x::Vector{Float64})
#     if x in cluster.elements
#         pop!(cluster, x)

#     end
#     return cluster
# end

# function delete!(clusters::Vector{Cluster}, x::Vector{Float64}; delete_empty::Bool=true)
#     for (ci, cluster) in enumerate(clusters)        
#         delete!(cluster, x)
#         if delete_empty && isempty(cluster)
#             deleteat!(clusters, ci)
#         end
#     end
#     return clusters
# end

Base.empty!(cluster::AbstractCluster{D, T}) where {D, T} = Base.empty!(cluster)
function Base.empty!(cluster::SetCluster{D, T}) where {D, T}
    empty!(cluster.elements)
    cluster.sum_x .= zero(T)
    cluster.sum_xx .= zero(T)
    return cluster
end
function Base.empty!(cluster.BitCluster{D, T}) where {D, T}
    cluster.mask .= false
    cluster.sum_x .= zero(T)
    cluster.xum_xx .= zero(T)
    return cluster
end


# function union(cluster1::SetCluster{D, T, V, S}, cluster2::SetCluster{D, T, V, S}) where {D, T, V, S}
#     elements = union(cluster1.elements, cluster2.elements)
#     new_sum_x, new_sum_xx = calculate_sums(elements)
#     return Cluster(elements, new_sum_x, new_sum_xx, Array{Float64}(undef, d), Array{Float64}(undef, d, d))
# end

Base.isempty(cluster::AbstractCluster) = isempty(cluster)
Base.isempty(cluster::SetCluster) = isempty(cluster.elements)
Base.isempty(cluster::BitCluster) = !any(cluster.mask)

Base.eltype(cluster::AbstractCluster) = Base.eltype(cluster)
Base.eltype(cluster::BitCluster{D, T}) where {D, T} = Vector{T}
Base.eltype(cluster::SetCluster{D, T}) where {D, T} = V

Base.iterate(cluster::SetCluster{D, T}) where {D, T} = iterate(cluster.elements)
Base.iterate(cluster::SetCluster{D, T}, state) where {D, T} = iterate(cluster.elements, state)

Base.first(cluster::SetCluster; orig=false) = !orig ? first(cluster.elements) : cluster.b2o[first(cluster.elements)]
Base.first(cluster::BitCluster; orig=false) = cluster.b2o[:, findfirst(cluster.mask), orig ? O : B]

Base.length(cluster::AbstractCluster) = length(cluster)
Base.length(cluster::BitCluster) = sum(cluster.mask)
Base.length(cluster::SetCluster) = length(cluster.elements)

Base.in(element, cluster::AbstractCluster) = in(element, cluster)
Base.in(element::AbstractVector{T}, cluster::SetCluster) where T = in(element, cluster.elements)
Base.in(element::Int, cluster::BitCluster) = cluster.mask[element]
Base.in(elements::BitVector, cluster::BitCluster) = all(cluster.mask[elements])
Base.in(element, clusters::AbstractVector{<:AbstractCluster}) = any([(element in cluster) for cluster in clusters])

Base.copy(cluster::SetCluster{D, T}) where {D, T} = SetCluster{D, T}(copy(cluster.elements), copy(cluster.b2o), copy(cluster.sum_x), copy(cluster.sum_xx), copy(cluster.mu_c_volatile), copy(cluster.psi_c_volatile))
Base.copy(cluster::BitCluster{D, T}) where {D, T} = BitCluster{D, T}(copy(cluster.mask), copy(cluster.b2o), copy(cluster.sum_x), copy(cluster.sum_xx), copy(cluster.mu_c_volatile), copy(cluster.psi_c_volatile))
function Base.copy(clusters::Vector{SetCluster{D, T}}) where {D, T}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = copy(b2o)
    return SetCluster{D, T}[SetCluster{D, T}(copy(cl.elements), _b2o, copy(cl.sum_x), copy(cl.sum_xx), copy(cl.mu_c_volatile), copy(cl.psi_c_volatile)) for cl in clusters]
end
function Base.copy(clusters::Vector{BitCluster{D, T}}) where {D, T}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = copy(b2o)
    return BitCluster{D, T}[BitCluster{D, T}(copy(cl.mask), _b2o, copy(cl.sum_x), copy(cl.sum_xx), copy(cl.mu_c_volatile), copy(cl.psi_c_volatile)) for cl in clusters]
end

Base.deepcopy(cluster::SetCluster{D, T}) where {D, T} = SetCluster{D, T}(deepcopy(cluster.elements), deepcopy(cluster.b2o), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
Base.deepcopy(cluster::BitCluster{D, T}) where {D, T} = BitCluster{D, T}(deepcopy(cluster.mask), deepcopy(cluster.b2o), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
function Base.deepcopy(clusters::Vector{SetCluster{D, T}}) where {D, T}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = deepcopy(b2o)
    return SetCluster{D, T}[SetCluster{D, T}(deepcopy(cl.elements), _b2o, deepcopy(cl.sum_x), deepcopy(cl.sum_xx), deepcopy(cl.mu_c_volatile), deepcopy(cl.psi_c_volatile)) for cl in clusters]
end
function Base.deepcopy(clusters::Vector{BitCluster{D, T}}) where {D, T}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = deepcopy(b2o)
    return BitCluster{D, T}[BitCluster{D, T}(deepcopy(cl.mask), _b2o, deepcopy(cl.sum_x), deepcopy(cl.sum_xx), deepcopy(cl.mu_c_volatile), deepcopy(cl.psi_c_volatile)) for cl in clusters]
end

function find(element::Vector{T}, clusters::AbstractVector{SetCluster{D, T}}) where {D, T}
    for (i, cl) in enumerate(clusters)
        if in(element, cl)
            return (cl, i)
        end
    end
    error(KeyError, ": key $element not found")
end

function find(i::Int, clusters::AbstractVector{BitCluster{D, T}}) where {D, T}
    1 <= i <= lastindex(first(clusters).mask) || throw(BoundsError(first(clusters).mask, i))
    for (ci, cl) in enumerate(clusters)
        if in(i, cl)
            return (cl, ci)
        end
    end
    error(KeyError, ": element $i not found")
end

function find(bitvector::BitVector, clusters::AbstractVector{BitCluster{D, T}}) where {D, T}
    length(bitvector) == length(first(clusters).mask) || error("BitVector must have the same length as the dataset")
    sum(bitvector) == 1 || error("BitVector must have exactly one true value/element")
    for (ci, cl) in enumerate(clusters)
        if in(bitvector, cl)
            return (cl, ci)
        end
    end
    error(KeyError, ": element $(findfirst(bitvector)) not found")
end


dims_to_proj(dims::Vector{Int}, d::Int) = diagm(ones(d))[dims, :]


# function project_cluster(cluster::Cluster, proj::Matrix{Float64})
#     return Cluster([proj * x for x in cluster])
# end

# function project_clusters(clusters::Vector{Cluster}, proj::Matrix{Float64})
#     return Cluster[project_cluster(cluster, proj) for cluster in clusters]
# end

# function project_clusters(clusters::Vector{Cluster}, dims::Vector{Int64})
#     el = pop!(first(clusters))
#     d = length(el)
#     push!(first(clusters), el)
#     return project_clusters(clusters, dims_to_proj(dims, d))
# end

function Base.Int(cluster::BitCluster)::Vector{Int}
    return findall(cluster.mask)
end

function Base.Vector(cluster::SetCluster{D, T}; orig=false)::Vector{Vector{T}} where {D, T} 
    if orig
        return [cluster.b2o[el] for el in cluster.elements]
    else
        return collect(cluster.elements)
    end
end
function Base.Vector(cluster::BitCluster{D, T}; orig=false)::Vector{Vector{T}} where {D, T} 
    return [cluster.b2o[:, i, orig ? O : B] for i in findall(cluster.mask)]
end


function Base.Matrix(cluster::SetCluster{D, T}; orig=false)::Matrix{T} where {D, T}
    if orig
        return reduce(hcat, [cluster.b2o[el] for el in cluster.elements], init=zeros(T, D, 0))
    else
        return reduce(hcat, collect(cluster.elements), init=zeros(T, D, 0))
    end
end
function Base.Matrix(cluster::BitCluster{D, T}; orig=false)::Matrix{T} where {D, T}
    return cluster.b2o[:, cluster.mask, orig ? O : B]
end

# function calculate_sums(cluster::Union{Cluster, Set{Vector{Float64}}, Vector{Vector{Float64}}})
#     d = length(first(cluster))
#     @assert all(length.(cluster) .== d) "All elements must be of dimension $d"

#     sum_x = zeros(Float64, d)
#     @inbounds for i in 1:d
#         for x in cluster
#             sum_x[i] += x[i]
#         end
#     end

#     sum_xx = zeros(Float64, d, d)
#     @inbounds for j in 1:d
#         @inbounds for i in 1:j
#             for x in cluster
#                 sum_xx[i, j] += x[i] * x[j]
#             end
#             sum_xx[j, i] = sum_xx[i, j]
#         end
#     end

#     return sum_x, sum_xx
# end

# function recalculate_sums!(cluster::Cluster)
#     cluster.sum_x, cluster.sum_xx = calculate_sums(cluster)
#     return cluster
# end