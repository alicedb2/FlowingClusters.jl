### We have to deal with vector input explicitly in
### forwardffjord so that we can specify the output types
### of named tuple in a type stable way. The logpxs/deltalogpxs
### are are a single number when a vector is passed, so you can't
### specify the ouput as Array{T, N-1} when N=1.
### Honestly this is a bit overkill but it's a good exercise

## forwardffjord pushes the data in the original space
## through the neural network and returns the data in the
## base space where the clustering happens
function forwardffjord(rng::AbstractRNG, x::AbstractVector{T}, hyperparams::AbstractFCHyperparams{T, D})::@NamedTuple{logpx::T, deltalogpxs::T, z::Vector{T}} where {T, D}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    hasnn(hyperparams) || return (;logpx=zero(T), deltalogpxs=zero(T), z=x)
    return forwardffjord(rng, x, hyperparams._, hyperparams.ffjord)
end

function forwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparams::AbstractFCHyperparams{T, D})::@NamedTuple{logpx::Array{T, N-1}, deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, D, N}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    hasnn(hyperparams) || return (;logpx=zeros(T, size(x)[2:end]), deltalogpxs=zeros(T, size(x)[2:end]), z=x)
    return forwardffjord(rng, x, hyperparams._, hyperparams.ffjord)
end

function forwardffjord(rng::AbstractRNG, x::AbstractVector{T}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::@NamedTuple{logpx::T, deltalogpxs::T, z::Vector{T}} where {T}
    ret = forwardffjord(rng, reshape(x, :, 1), hyperparamsarray, ffjord)
    return (; logpx=first(ret.logpx), deltalogpxs=first(ret.deltalogpxs), z=reshape(ret.z, :))
end

function forwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::@NamedTuple{logpx::Array{T, N-1}, deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, N}
    hasnn(hyperparamsarray) || return (;logpx=zeros(T, size(x)[2:end]), deltalogpxs=zeros(T, size(x)[2:end]), z=x)
    D = size(x, 1)
    D == first(ffjord.nn.layers).in_dims == last(ffjord.nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

    # FFJORD expect a matrix with each data point as a column
    if ndims(x) > 2
        _x = reshape(x, D, :)
    else
        _x = x
    end

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), T(1)), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = first(ffjord_mdl(_x, hyperparamsarray.nn.params, ffjord.nns, rng))

    return (; logpx=reshape(ret.logpx, size(x)[2:end]...), deltalogpxs=reshape(ret.delta_logp, size(x)[2:end]...), z=reshape(ret.z, size(x)...))
end


## backwardffjord is easier in terms of type stability
## because the input is always an array with the
## same dimension as the output

## backwardffjord pulls the data from the base space
## through the neural network and returns the data in
## the original space.
function backwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::Array{T, N} where {T, N}
    hasnn(hyperparamsarray) || return x
    D = size(x, 1)
    D == first(ffjord.nn.layers).in_dims == last(ffjord.nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

    if ndims(x) == 1
        _x = reshape(x, D, 1)
    elseif ndims(x) == 2
        _x = x
    else
        _x = reshape(x, D, :)
    end

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), T(1)), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = __backward_ffjord(ffjord_mdl, _x, hyperparamsarray.nn.params, ffjord.nns, rng)
    return reshape(ret, size(x)...)

end

function backwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparams::AbstractFCHyperparams{T, D})::Array{T, N} where {T, D, N}
    hasnn(hyperparams) || return x
    return backwardffjord(rng, x, hyperparams._, hyperparams.ffjord)
end

function reflow(rng::AbstractRNG, clusters::AbstractVector{SetCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple) where {T, D, E}

    cluster_sizes = length.(clusters)
    base2original = first(clusters).b2o
    @assert all([base2original === c.b2o for c in clusters])
    # origdata = reduce(hcat, [base2original[el] for c in clusters for el in c])
    origdata = Matrix(clusters, orig=true)

    _, deltalogpxs, new_basedata = forwardffjord(rng, origdata, hyperparamsarray, ffjord)

    new_base2original = Dict{SVector{D, T}, SVector{D, T}}(collect.(eachcol(new_basedata)) .=> collect.(eachcol(origdata)))
    new_clusters = [SetCluster(basedata, new_base2original) for basedata in chunk(new_basedata, cluster_sizes)]

    return (clusters=new_clusters, deltalogpxs=deltalogpxs)
end
function reflow(rng::AbstractRNG, clusters::AbstractVector{BitCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple) where {T, D, E}
    base2original = first(clusters).b2o

    _, deltalogpxs, new_basedata = forwardffjord(rng, base2original[:, :, O], hyperparamsarray, ffjord)

    new_base2original = cat(new_basedata, base2original[:, :, O], dims=3)

    return (clusters=[BitCluster(cluster.mask, new_base2original) for cluster in clusters], deltalogpxs=deltalogpxs)
end
function reflow(rng::AbstractRNG, clusters::AbstractVector{IndexCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple) where {T, D, E}
    base2original = first(clusters).b2o

    _, deltalogpxs, new_basedata = forwardffjord(rng, base2original[:, :, O], hyperparamsarray, ffjord)

    new_base2original = cat(new_basedata, base2original[:, :, O], dims=3)

    return (clusters=[BitCluster(cluster.mask, new_base2original) for cluster in clusters], deltalogpxs=deltalogpxs)
end