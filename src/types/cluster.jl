abstract type AbstractCluster{T, D, E} end

struct EmptyCluster{T, D, E} <: AbstractCluster{T, D, E}
end

struct SetCluster{T, D, E <: SVector{D, T}} <: AbstractCluster{T, D, E}
    elements::Set{E}
    b2o::Dict{E, E}
    sum_x::Vector{T}
    sum_xx::Matrix{T}

    # Scratch arrays to avoid allocations
    mu_c_volatile::Vector{T}
    psi_c_volatile::Matrix{T}

end

function SetCluster(elements::AbstractVector{SVector{D, T}}, b2o::AbstractDict{SVector{D, T}, SVector{D, T}}; check=false) where {D, T}

    # We don't check by default because we assume
    # that the same checks have been done during the
    # creation of the chain, but we can use that
    # during unit testing
    if check
        # Do some dimension checking, make sure everyone is the same size
        all(size.(keys(b2o), 1) .== size.(values(b2o), 1)) || throw(DimensionMismatch("All elements in dataset must have the same dimension"))
        all(size.(elements, 1) .== size(first(keys(b2o)), 1)) || throw(DimensionMismatch("Dimension of elements must be the same as the dimension of the dataset"))
        # and that everyone we try to put in this cluster is in the dataset
        all([element in keys(b2o) for element in elements]) || throw(ArgumentError("Element not in b2o"))
    end

    # D = size(first(elements), 1)
    @assert D == size(first(keys(b2o)), 1) == size(first(values(b2o)), 1)
    sum_x = sum(elements)
    sum_xx = sum([x * x' for x in elements])
    mucvol = Vector{T}(undef, D)
    psicvol = Matrix{T}(undef, D, D)
    return SetCluster{T, D, SVector{D, T}}(Set{SVector{D, T}}([SVector{D, T}(el) for el in elements]), b2o, sum_x, sum_xx, mucvol, psicvol)
end

function SetCluster(element::AbstractVector{T}, b2o::AbstractDict{SVector{D, T}, SVector{D, T}}; check=false) where {D, T}
    return SetCluster([SVector{size(element, 1), T}(element)], b2o, check=check)
end

function SetCluster(elements::AbstractMatrix{T}, b2o::Dict{SVector{D, T}, SVector{D, T}}; check=false) where {D, T}
    size(elements, 1) == D || throw(DimensionMismatch("All elements in dataset must have the same dimension"))
    if size(elements, 2) == 0
        return SetCluster(b2o, check=check)
    else
        elements = SVector{D, T}.(eachcol(elements))
        return SetCluster(elements, b2o)
    end
end

function SetCluster(b2o::AbstractDict{SVector{D, T}, SVector{D, T}}; check=false) where{D, T}
    if check
        all(size.(keys(b2o), 1) .== size.(values(b2o), 1)) || throw(DimensionMismatch("All elements in dataset must have the same dimension"))
    end
    return SetCluster{T, D, SVector{D, T}}(Set{SVector{D, T}}(), b2o, zeros(T, D), zeros(T, D, D), Array{T}(undef, D), Array{T}(undef, D, D))
end

const B = 1 # Base space elements in BitCluster b2o
const O = 2 # Original space elements in BitCluster b2o

struct BitCluster{T, D, E <: Int} <: AbstractCluster{T, D, E}
    mask::BitVector
    b2o::Array{T, 3}
    sum_x::Vector{T}
    sum_xx::Matrix{T}

    # Scratch arrays to avoid allocations
    mu_c_volatile::Vector{T}
    psi_c_volatile::Matrix{T}
end

function BitCluster(mask::BitVector, b2o::AbstractArray{T, 3}; check=false) where T
    if check
        size(mask, 1) == size(b2o, 2) || throw(DimensionMismatch("Mask must have the same size as the dataset"))
    end
    D = size(b2o, 1)
    sum_x = dropdims(sum(b2o[:, mask, B], dims=2), dims=2)
    sum_xx = sum([x * x' for x in eachcol(b2o[:, mask, B])])
    mucvol = Vector{T}(undef, D)
    psicvol = Matrix{T}(undef, D, D)
    return BitCluster{T, D, Int}(mask, b2o, sum_x, sum_xx, mucvol, psicvol)
end

function BitCluster(idx::Int, b2o::AbstractArray{T, 3}; check=false) where T
    if check
        1 <= idx <= size(b2o, 2) || throw(ArgumentError("Index out of bounds"))
    end
    mask = falses(size(b2o, 2))
    mask[idx] = true
    return BitCluster(mask, b2o)
end

function BitCluster(idxarray::AbstractVector{Int}, b2o::AbstractArray{T, 3}; check=false) where T
    if check
        all(1 .<= idxarray .<= size(b2o, 2)) || throw(ArgumentError("One or more index out of bounds"))
    end

    mask = falses(size(b2o, 2))
    mask[idxarray] .= true
    return BitCluster(mask, b2o)
end

function BitCluster(b2o::AbstractArray{T, 3}) where T
    D, N = size(b2o, 1), size(b2o, 2)
    return BitCluster{T, D, Int}(falses(N), b2o, zeros(T, D), zeros(T, D, D), Array{T}(undef, D), Array{T}(undef, D, D))
end

function SetCluster(cluster::BitCluster{T, D, E}; check=false) where {T, D, E}
    b2o = Dict{SVector{D, T}, SVector{D, T}}(collect.(eachcol(cluster.b2o[:, :, B])) .=> collect.(eachcol(cluster.b2o[:, :, O])))
    return SetCluster(SVector{D, T}.(Vector(cluster, orig=false)), b2o, check=check)
end

function SetCluster(clusters::AbstractVector{BitCluster{T, D, E}}; check=false)::Vector{SetCluster{T, D, SVector{D, T}}} where {T, D, E}
    b2o = first(clusters).b2o
    b2o = Dict{SVector{D, T}, SVector{D, T}}(collect.(eachcol(b2o[:, :, B])) .=> collect.(eachcol(b2o[:, :, O])))
    return [SetCluster(SVector{D, T}.(Vector(cluster, orig=false)), b2o, check=check) for cluster in clusters]
end

# No real way to do that efficiently
# other than by repeatedly traversing 
# b2o and testing vector equalities.
# Unfortunately this means the masks
# are not guaranteed to be the same between
# BitCluster(clusters) and BitCluster.(clusters)
function BitCluster(cluster::SetCluster{T, D, E}) where {T, D, E}
    b2o = Array(cluster.b2o)
    mask = BitVector(zeros(Bool, size(b2o, 2)))
    for v in cluster
        idx = findall(eachcol(b2o[:, :, B]) .== [v])
        length(idx) == 1 || throw("Duplicates in base2original map")
        mask[first(idx)] = true
    end
    return BitCluster(mask, b2o)
end

function BitCluster(clusters::AbstractVector{SetCluster{T, D, E}}; check=false)::Vector{BitCluster{T, D, Int}} where {T, D, E}
    basedata = Matrix(clusters, orig=false)
    origdata = Matrix(clusters, orig=true)
    b2o = cat(basedata, origdata, dims=3)
    slices = chunkslices(length.(clusters))
    nbelements = size(b2o, 2)
    masks = [BitVector(zeros(Bool, nbelements)) for _ in slices]
    for (mask, slice) in zip(masks, slices)
        mask[slice] .= true
    end
    return [BitCluster(mask, b2o, check=check) for mask in masks]
end

# Cute, only uses two bitvectors, doesn't need to switch to integers
function isvalidpartition(bitclusters::AbstractVector{BitCluster{T, D, E}}; fullresult=false) where {T, D, E}
    isempty(bitclusters) && return true
    # andvector keeps track of collisions as we iterate through the bitclusters.
    # orvector keeps track of elements seen as we iterate through the bitclusters.

    # When we see an element in the current bitcluster that is already in the orvector
    # we record it as a collision in the andvector.

    # At the end if there are any true values in the andvector then it means
    # that at least one element is shared between any two of the clusters
    # and the partition is invalid. All clusters should be mutually exclusive.

    #                                                      ((    record new collisions   ),  (records new elements))
    #                                                      ((      in the and-vector     ),  (  in the or-vector  ))
    av, ov = fill!(similar(first(bitclusters).mask), false), fill!(similar(first(bitclusters).mask), false)

    result = reduce(((andvector, orvector), x) -> ((andvector .|= (orvector .& x)), (   orvector .|= x   )),
                    [cl.mask for cl in bitclusters],
                    init=(av, ov)
                   ) # reduce() returns (andvector, orvector)
    if fullresult
        return !any(first(result)), av
    else
        return !any(first(result))
    end
end

# Let's do the same thing for SetCluster!
# Just need to replace .& with intersection and .| with union
# and check that the final intersection is empty
# Waaaay slower than using bitclusters even though we do it in-place as well
function isvalidpartition(setclusters::AbstractVector{SetCluster{T, D, E}}; fullresult=false) where {T, D, E}
    rep = all([el in keys(cl.b2o) for cl in setclusters for el in cl.elements])
    ref = all([first(setclusters).b2o === cl.b2o for cl in setclusters])
    result = reduce(((intset, unset), x) -> (union!(intset, intersect(unset, x)), union!(unset, x)),
                    [cl.elements for cl in setclusters],
                    init=(Set{Vector{T}}(),
                          Set{Vector{T}}())
                   )
    if fullresult
        return rep && ref && isempty(first(result)), first(result)
    else
        return rep && ref && isempty(first(result))
    end

end

function iscompletepartition(bitclusters::AbstractVector{BitCluster{T, D, E}}) where {T, D, E}
    isempty(bitclusters) && return false
    return all(reduce((ov, x) -> ov .| x, [cl.mask for cl in  bitclusters], init=fill!(similar(first(bitclusters).mask), false)))
end

function iscompletepartition(setclusters::AbstractVector{SetCluster{T, D, E}}) where {T, D, E}
    isempty(setclusters) && return false
    unionset = reduce((unionset, x) -> union!(unionset, x), [cl.elements for cl in  setclusters], init=Set{Vector{T}}())
    return unionset == keys(first(setclusters).b2o)
end

Base.show(io::IO, cluster::AbstractCluster) = print(io, "$(typeof(cluster).name.wrapper){$(typeof(cluster).parameters[1]), $(typeof(cluster).parameters[2])}($(length(cluster)))")


### Pop/push operations work on elements that are compatible
### with gibbs/split-merge, i.e. integers for BitCluster and
### vectors for SetCluster. If you want the element's vector
### out of a BitCluster you can use the Vector() function

# Pop element from cluster
function Base.pop!(cluster::BitCluster{T, D, E}, i::E)::E where {T, D, E}
    cluster.mask[i] || throw(KeyError("Element $i is not in the cluster"))
    cluster.mask[i] = false
    _pop_update_sums!(cluster, i)
    return i
end
function Base.pop!(cluster::SetCluster{T, D, E}, x::E)::E where {T, D, E}
    x = pop!(cluster.elements, x)
    _pop_update_sums!(cluster, x)
    return x
end

# Pop element from whole partition
# function Base.pop!(bitclusters::AbstractVector{BitCluster{T, D, E}}, i::E)::E where {T, D, E}
#     for cluster in bitclusters
#         if cluster.mask[i]
#             cluster.mask[i] = false
#             _pop_update_sums!(cluster, i)
#             return i
#         end
#     end
#     throw(KeyError("Element $i is not in the partition"))
# end

function Base.pop!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, x::E; deleteifempty=true)::E where {T, D, E}
    for (ci, cluster) in enumerate(clusters)
        if x in cluster
            x = pop!(cluster, x)
            if deleteifempty && isempty(cluster)
                deleteat!(clusters, ci)
            end
            return x
        end
    end
    throw(KeyError("key $x not found"))
end

function Base.push!(cluster::BitCluster{T, D, E}, i::E) where {T, D, E}
    if !cluster.mask[i]
        cluster.mask[i] = true
        _push_update_sums!(cluster, i)
    end
    return cluster
end

function Base.push!(cluster::SetCluster{T, D, E}, x::E) where {T, D, E}
    if !(x in cluster.elements)
        push!(cluster.elements, x)
        _push_update_sums!(cluster, x)
    end
    return cluster
end

function Base.delete!(cluster::SetCluster{T, D, E}, x::E) where {T, D, E}
    if x in cluster.elements
        pop!(cluster, x)
    end
    return cluster
end

function Base.delete!(cluster::BitCluster{T, D, E}, x::E) where {T, D, E}
    if cluster.mask[x]
        pop!(cluster, x)
    end
    return cluster
end

function Base.delete!(clusters::AbstractVector{<:AbstractCluster{T, D, E}}, x::E; deleteifempty=true) where {T, D, E}
    for (ci, cluster) in enumerate(clusters)
        delete!(cluster, x)
        if deleteifempty && isempty(cluster)
            deleteat!(clusters, ci)
            break
        end
    end
    return clusters
end

function Base.union(cluster1::SetCluster{T, D, E}, cluster2::SetCluster{T, D, E}) where {T, D, E}
    elements = union(cluster1.elements, cluster2.elements)
    new_sum_x, new_sum_xx = calculate_sums(elements)
    return SetCluster{T, D, E}(elements, cluster1.b2o, new_sum_x, new_sum_xx, Array{T}(undef, D), Array{T}(undef, D, D))
end

function Base.union(cluster1::BitCluster{T, D, E}, cluster2::BitCluster{T, D, E}) where {T, D, E}
    cluster1.b2o === cluster2.b2o || throw(ArgumentError("Clusters aren't from the same dataset"))
    mask = cluster1.mask .| cluster2.mask
    new_sum_x, new_sum_xx = calculate_sums(cluster1.b2o[:, mask, B])
    return BitCluster{T, D, E}(mask, cluster1.b2o, new_sum_x, new_sum_xx, Array{T}(undef, D), Array{T}(undef, D, D))
end

function Base.empty!(cluster::SetCluster{T, D, E}) where {T, D, E}
    empty!(cluster.elements)
    cluster.sum_x .= zero(T)
    cluster.sum_xx .= zero(T)
    return cluster
end
function Base.empty!(cluster::BitCluster{T, D, E}) where {T, D, E}
    cluster.mask .= false
    cluster.sum_x .= zero(T)
    cluster.xum_xx .= zero(T)
    return cluster
end

Base.isempty(cluster::AbstractCluster) = isempty(cluster)
Base.isempty(cluster::SetCluster) = isempty(cluster.elements)
Base.isempty(cluster::BitCluster) = !any(cluster.mask)

Base.eltype(::AbstractCluster{T, D, E}) where {T, D, E} = E

Base.iterate(cluster::SetCluster) = iterate(cluster.elements)
Base.iterate(cluster::SetCluster, state) = iterate(cluster.elements, state)

function Base.iterate(cluster::BitCluster{T, D, E}) where {T, D, E}
    idx = findall(cluster.mask)
    return isempty(idx) ? nothing : (idx[1], (idx, 2))
end

function Base.iterate(::BitCluster{T, D, E}, state::Tuple{Vector{Int}, E}) where {T, D, E}
    idx, i = state
    return i > lastindex(idx) ? nothing : (idx[i], (idx, i + 1))
end

Base.first(cluster::SetCluster) = first(cluster.elements)
Base.first(cluster::BitCluster) = findfirst(cluster.mask)

Base.length(cluster::AbstractCluster) = length(cluster)
Base.length(cluster::EmptyCluster) = 0
Base.length(cluster::BitCluster) = sum(cluster.mask)
Base.length(cluster::SetCluster) = length(cluster.elements)

Base.in(element, cluster::AbstractCluster) = in(element, cluster)
Base.in(element, cluster::EmptyCluster) = false
Base.in(element::AbstractVector{T}, cluster::SetCluster{T, D, E}) where {T, D, E} = in(element, cluster.elements)
Base.in(element::Int, cluster::BitCluster) = cluster.mask[element]
Base.in(elements::BitVector, cluster::BitCluster) = all(cluster.mask[elements])
Base.in(element, clusters::AbstractVector{<:AbstractCluster}) = any([(element in cluster) for cluster in clusters])

Base.copy(::EmptyCluster{T, D, E}) where {T, D, E} = EmptyCluster{T, D, E}()
Base.copy(cluster::SetCluster{T, D, E}) where {T, D, E} = SetCluster{T, D, E}(copy(cluster.elements), copy(cluster.b2o), copy(cluster.sum_x), copy(cluster.sum_xx), copy(cluster.mu_c_volatile), copy(cluster.psi_c_volatile))
Base.copy(cluster::BitCluster{T, D, E}) where {T, D, E} = BitCluster{T, D, E}(copy(cluster.mask), copy(cluster.b2o), copy(cluster.sum_x), copy(cluster.sum_xx), copy(cluster.mu_c_volatile), copy(cluster.psi_c_volatile))
function Base.copy(clusters::AbstractVector{SetCluster{T, D, E}}) where {T, D, E}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = copy(b2o)
    return SetCluster{T, D, E}[SetCluster{T, D, E}(copy(cl.elements), _b2o, copy(cl.sum_x), copy(cl.sum_xx), copy(cl.mu_c_volatile), copy(cl.psi_c_volatile)) for cl in clusters]
end
function Base.copy(clusters::AbstractVector{BitCluster{T, D, E}}) where {T, D, E}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = copy(b2o)
    return BitCluster{T, D, E}[BitCluster{T, D, E}(copy(cl.mask), _b2o, copy(cl.sum_x), copy(cl.sum_xx), copy(cl.mu_c_volatile), copy(cl.psi_c_volatile)) for cl in clusters]
end

Base.deepcopy(::EmptyCluster{T, D, E}) where {T, D, E} = EmptyCluster{T, D, E}()
Base.deepcopy(cluster::SetCluster{T, D, E}) where {T, D, E} = SetCluster{T, D, E}(deepcopy(cluster.elements), deepcopy(cluster.b2o), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
Base.deepcopy(cluster::BitCluster{T, D, E}) where {T, D, E} = BitCluster{T, D, E}(deepcopy(cluster.mask), deepcopy(cluster.b2o), deepcopy(cluster.sum_x), deepcopy(cluster.sum_xx), deepcopy(cluster.mu_c_volatile), deepcopy(cluster.psi_c_volatile))
function Base.deepcopy(clusters::AbstractVector{SetCluster{T, D, E}}) where {T, D, E}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = deepcopy(b2o)
    return SetCluster{T, D, E}[SetCluster{T, D, E}(deepcopy(cl.elements), _b2o, deepcopy(cl.sum_x), deepcopy(cl.sum_xx), deepcopy(cl.mu_c_volatile), deepcopy(cl.psi_c_volatile)) for cl in clusters]
end
function Base.deepcopy(clusters::AbstractVector{BitCluster{T, D, E}}) where {T, D, E}
    b2o = first(clusters).b2o
    @assert all([b2o === cl.b2o for cl in clusters])
    _b2o = deepcopy(b2o)
    return BitCluster{T, D, E}[BitCluster{T, D, E}(deepcopy(cl.mask), _b2o, deepcopy(cl.sum_x), deepcopy(cl.sum_xx), deepcopy(cl.mu_c_volatile), deepcopy(cl.psi_c_volatile)) for cl in clusters]
end

function Base.Vector(cluster::BitCluster{T, D, E}, i::E; orig=false, checkpresence=true)::Vector{T} where {T, D, E}
    1 <= i <= lastindex(cluster.mask) || throw(BoundsError(cluster.mask, i))
    if checkpresence
        cluster.mask[i] || throw(KeyError("Element $i is not in the cluster"))
    end
    return cluster.b2o[:, i, orig ? O : B]
end

function Base.Vector(clusters::AbstractVector{BitCluster{T, D, E}}, i::Int; orig=false, checkpresence=false)::Vector{T} where {T, D, E}
    if checkpresence
        find(i, clusters) # Let find fail if i is not in clusters
    end
    return first(clusters).b2o[:, i, orig ? O : B]
end

function Base.Vector(cluster::SetCluster{T, D, E}; orig=false)::Vector{Vector{T}} where {T, D, E}
    if orig
        return [cluster.b2o[el] for el in cluster.elements]
    else
        return collect(cluster.elements)
    end
end

function Base.Vector(cluster::BitCluster{T, D, E}; orig=false)::Vector{Vector{T}} where {T, D, E}
    return [cluster.b2o[:, i, orig ? O : B] for i in cluster]
end


function Base.Matrix(cluster::SetCluster{T, D, E}; orig=false)::Matrix{T} where {T, D, E}
    if orig
        return reduce(hcat, [cluster.b2o[el] for el in cluster.elements], init=zeros(T, D, 0))
    else
        return reduce(hcat, collect(cluster.elements), init=zeros(T, D, 0))
    end
end

function Base.Matrix(cluster::BitCluster{T, D, E}; orig=false)::Matrix{T} where {T, D, E}
    return cluster.b2o[:, cluster.mask, orig ? O : B]
end

function Base.Matrix(clusters::AbstractVector{<:AbstractCluster{T, D, E}}; orig=false)::Matrix{T} where {T, D, E}
    # Could grab b2o from the first cluster but then you'd be assuming
    # no element has been popped from any cluster.
    return reduce(hcat, [Matrix(cl, orig=orig) for cl in clusters], init=zeros(T, D, 0))
end

function availableelements(clusters::AbstractVector{<:AbstractCluster{T, D, E}})::Vector{E} where {T, D, E}
    return [el for cl in clusters for el in cl]
end

function allelements(clusters::AbstractVector{<:AbstractCluster{T, D, E}})::Vector{E} where {T, D, E}
    length(clusters) > 0 || throw(ArgumentError("Empty partition"))
    return allelements(clusters)
end

function allelements(clusters::AbstractVector{BitCluster{T, D, E}})::Vector{E} where {T, D, E}
    length(clusters) > 0 || throw(ArgumentError("Empty partition"))
    return collect(1:size(first(clusters).mask, 2))
end

function allelements(clusters::AbstractVector{SetCluster{T, D, E}})::Vector{E} where {T, D, E}
    length(clusters) > 0 || throw(ArgumentError("Empty partition"))
    return collect(keys(first(clusters).b2o))
end



# function elements(clusters::AbstractVector{SetCluster{T, D, E}})::Vector{Vector{T}} where {T, D, E}
#     return [el for cl in clusters for el in elements(cl)]
# end

# function elements(clusters::AbstractVector{BitCluster{T, D, E}})::Vector{Int} where {T, D, E}
#     return [el for cl in clusters for el in elements(cl)]
# end


function find(element::E, clusters::AbstractVector{SetCluster{T, D, E}}) where {T, D, E}
    for (i, cl) in enumerate(clusters)
        if in(element, cl)
            return (cl, i)
        end
    end
    error(KeyError, ": key $element not found")
end

function find(i::E, clusters::AbstractVector{BitCluster{T, D, E}}) where {T, D, E}
    1 <= i <= lastindex(first(clusters).mask) || throw(BoundsError(first(clusters).mask, i))
    for (ci, cl) in enumerate(clusters)
        if in(i, cl)
            return (cl, ci)
        end
    end
    error(KeyError, ": element $i not found")
end

# function find(bitvector::BitVector, clusters::AbstractVector{BitCluster{T, D, E}}) where {T, D, E}
#     length(bitvector) == length(first(clusters).mask) || error("BitVector must have the same length as the dataset")
#     sum(bitvector) == 1 || error("BitVector must have exactly one true value/element")
#     for (ci, cl) in enumerate(clusters)
#         if in(bitvector, cl)
#             return (cl, ci)
#         end
#     end
#     error(KeyError, ": element $(findfirst(bitvector)) not found")
# end


function project_cluster(cluster::SetCluster{T, D, E}, proj=[1, 2]; orig=false) where {T, D, E}
    if orig
        return project_vec.([cluster.b2o[el] for el in cluster], Ref(proj))
    else
        return project_vec.(cluster, Ref(proj))
    end
end

function project_cluster(cluster::BitCluster{T, D, E}, proj=[1, 2]; orig=false) where {T, D, E}
    return project_vec.(eachcol(cluster.b2o[:, cluster.mask, orig ? O : B]), Ref(proj))
end

function project_clusters(clusters::AbstractVector{<:AbstractCluster}, proj=[1, 2]; orig=false)
    return project_cluster.(clusters, Ref(proj), orig=orig)
end

function calculate_sums(cluster::SetCluster{T, D, E}) where {T, D, E}

    sum_x = zeros(T, D)
    sum_xx = zeros(T, D, D)

    if isempty(cluster)
        return sum_x, sum_xx
    end

    @inbounds for i in 1:D
        for x in cluster
            sum_x[i] += x[i]
        end
    end

    for x in cluster
    @inbounds for j in 1:D
        @inbounds for i in 1:j
                sum_xx[i, j] += x[i] * x[j]
                sum_xx[j, i] = sum_xx[i, j]
            end
        end
    end

    return sum_x, sum_xx
end


function calculate_sums(cluster::BitCluster{T, D, E}) where {T, D, E}

    sum_x = zeros(T, D)
    sum_xx = zeros(T, D, D)

    if isempty(cluster)
        return sum_x, sum_xx
    end

    @inbounds for idx in cluster
        @inbounds for i in 1:D
            sum_x[i] += cluster.b2o[i, idx, B]
        end
    end

    @inbounds for idx in cluster
        @inbounds for j in 1:D
            @inbounds for i in 1:j
                sum_xx[i, j] += cluster.b2o[i, idx, B] * cluster.b2o[j, idx, B]
                sum_xx[j, i] = sum_xx[i, j]
            end
        end
    end

    return sum_x, sum_xx
end

# Could be done in place but I'm not reall using
# this function anywhere tight at the moment.
# If I were to it would be to get rid of
# the potential accumulation of float underflows
# after a large number of iterations involving
# Gibbs sampling and low-precision.
# edit: well now I'm using it in generate_data()
#       to deal with collisions in low-precision
function _recalculate_sums!(cluster::AbstractCluster)
    new_sum_x, new_sum_xx = calculate_sums(cluster)
    cluster.sum_x .= new_sum_x
    cluster.sum_xx .= new_sum_xx
    return cluster.sum_x, cluster.sum_xx
end
function _recalculate_sums!(clusters::AbstractVector{<:AbstractCluster})
    _recalculate_sums!.(clusters)
    return cluster.sum_x, cluster.sum_xx
end

## _pop/_push functions do not perform any check

function _push_update_sums!(cluster::BitCluster{T, D, E}, i::Int) where {T, D, E}
    return _push_update_sums!(cluster, cluster.b2o[:, i, B])
end

function _push_update_sums!(cluster::AbstractCluster{T, D, E}, x::AbstractVector{T}) where {T, D, E}

    @inbounds for i in 1:D
        cluster.sum_x[i] += x[i]
    end

    @inbounds for j in 1:D
        @inbounds for i in 1:j
            cluster.sum_xx[i, j] += x[i] * x[j]
            cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
        end
    end

    return cluster.sum_x, cluster.sum_xx

end

function _pop_update_sums!(cluster::BitCluster{T, D, E}, i::Int) where {T, D, E}
    return _pop_update_sums!(cluster, cluster.b2o[:, i, B])
end

function _pop_update_sums!(cluster::AbstractCluster{T, D, E}, x::AbstractVector{T}) where {T, D, E}
    @inbounds for i in 1:D
        cluster.sum_x[i] -= x[i]
    end

    @inbounds for j in 1:D
        @inbounds for i in 1:j
            cluster.sum_xx[i, j] -= x[i] * x[j]
            cluster.sum_xx[j, i] = cluster.sum_xx[i, j]
        end
    end
    return cluster.sum_x, cluster.sum_xx
end