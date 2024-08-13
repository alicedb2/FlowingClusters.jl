### We have to deal with vector input explicitly in
### forwardffjord so that we can specify the output types
### of named tuple in a type stable way. The logpxs/deltalogpxs
### are are a single number when a vector is passed, so you can't
### specify the ouput as Array{T, N-1} when N=1.
### Honestly this is a bit overkill but it's a good exercise

function forwardffjord(x::AbstractVector{T}, hyperparams::AbstractFCHyperparams{T, D})::@NamedTuple{logpx::T, deltalogpxs::T, z::Vector{T}} where {T, D}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    hasnn(hyperparams) || return (;logpx=zero(T), deltalogpxs=zero(T), z=x)
    return forwardffjord(x, hyperparams._, hyperparams.nn)
end

function forwardffjord(x::AbstractArray{T, N}, hyperparams::AbstractFCHyperparams{T, D})::@NamedTuple{logpx::Array{T, N-1}, deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, D, N}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    hasnn(hyperparams) || return (;logpx=zeros(T, size(x)[2:end]), deltalogpxs=zeros(T, size(x)[2:end]), z=x)
    return forwardffjord(x, hyperparams._, hyperparams.nn)
end

function forwardffjord(x::AbstractVector{T}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::@NamedTuple{logpx::T, deltalogpxs::T, z::Vector{T}} where {T}
    ret = forwardffjord(reshape(x, :, 1), hyperparamsarray, ffjord)
    return (;logpx=first(ret.logpx), deltalogpxs=first(ret.deltalogpxs), z=reshape(ret.z, :))
end

function forwardffjord(x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::@NamedTuple{logpx::Array{T, N-1}, deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, N}
    hasnn(hyperparamsarray) || return (;logpx=zeros(T, size(x)[2:end]), deltalogpxs=zeros(T, size(x)[2:end]), z=x)
    D = size(x, 1)
    D == first(ffjord.nn.layers).in_dims == last(ffjord.nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

    # FFJORD expect a matrix with each data point as a column
    if ndims(x) > 2
        _x = reshape(x, D, :)
    else
        _x = x
    end

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), one(T)), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = first(ffjord_mdl(_x, hyperparamsarray.nn.params, ffjord.nns))

    return (;logpx=reshape(ret.logpx, size(x)[2:end]...), deltalogpxs=reshape(ret.delta_logp, size(x)[2:end]...), z=reshape(ret.z, size(x)...))
end

## backwardffjord is easier in terms of type stability
## because the input is always an array with the
## same dimension as the output
function backwardffjord(x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple)::Array{T, N} where {T, N}
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

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), one(T)), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = __backward_ffjord(ffjord_mdl, _x, hyperparamsarray.nn.params, ffjord.nns)
    return reshape(ret, size(x)...)

end

function reflow!(clusters::AbstractVector{SetCluster{T, D, E}}, nnparams::ComponentArray{T}, nns::NamedTuple) where {T, D, E}
    
    
    for cluster in clusters
        reflow!(cluster, nnparams, nns)
    end
end

function reflow!(clusters::AbstractVector{BitCluster{T, D, E}}, nnparams::ComponentArray{T}, nns::NamedTuple) where {T, D, E}
end


reflow!(clusters::AbstractClusters{TD, E}, hyperparams::AbstractFCHyperparams{T, D}) where {T, D, E, TD} = reflow!(clusters, hyperparams.nn.nn, hyperparams.nn.nns)