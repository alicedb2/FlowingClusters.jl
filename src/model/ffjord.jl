### We have to deal with vector input explicitly in
### forwardffjord so that we can specify the output types
### of named tuple in a type stable way. The logpxs/deltalogpxs
### are are a single number when a vector is passed, so you can't
### specify the ouput as Array{T, N-1} when N=1.
### Honestly this is a bit overkill but it's a good exercise

## forwardffjord pushes the data in the original space
## through the neural network and returns the data in the
## base space where the clustering happens
function forwardffjord(rng::AbstractRNG, x::AbstractVector{T}, hyperparams::AbstractFCHyperparams{T, D}; t=T(1))::@NamedTuple{deltalogpxs::T, z::Vector{T}} where {T, D}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    if hasnn(hyperparams)
        return forwardffjord(rng, x, hyperparams._, hyperparams.ffjord, t=t)
    else
        return (; deltalogpxs=zero(T), z=x)
    end
end

function forwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparams::AbstractFCHyperparams{T, D}; t=T(1))::@NamedTuple{deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, D, N}
    D === size(x, 1) || throw(ArgumentError("The dimension of the data must be the same as the dimension of the hyperparameters"))
    if hasnn(hyperparams)
        return forwardffjord(rng, x, hyperparams._, hyperparams.ffjord, t=t)
    else
        return (; deltalogpxs=zeros(T, size(x)[2:end]),
                  z=x)
    end
end

function forwardffjord(rng::AbstractRNG, x::AbstractVector{T}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1))::@NamedTuple{deltalogpxs::T, z::Vector{T}} where {T}
    ret = forwardffjord(rng, reshape(x, :, 1), hyperparamsarray, ffjord, t=t)
    return (; deltalogpxs=first(ret.deltalogpxs),
              z=reshape(ret.z, :))
end

function forwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1))::@NamedTuple{deltalogpxs::Array{T, N-1}, z::Array{T, N}} where {T, N}
    hasnn(hyperparamsarray) || return (;logpx=zeros(T, size(x)[2:end]), deltalogpxs=zeros(T, size(x)[2:end]), z=x)
    D = size(x, 1)
    # D == first(ffjord.nn.layers).in_dims == last(ffjord.nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

    # FFJORD expect a matrix with each data point as a column
    if ndims(x) > 2
        _x = reshape(x, D, :)
    else
        _x = x
    end

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), t), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = first(ffjord_mdl(_x, hyperparamsarray.nn.params, ffjord.nns, rng))

    return (; deltalogpxs=reshape(ret.delta_logp, size(x)[2:end]...),
              z=reshape(ret.z, size(x)...))
end


## backwardffjord is easier in terms of type stability
## because the input is always an array with the
## same dimension as the output

## backwardffjord pulls the data from the base space
## through the neural network and returns the data in
## the original space.
function backwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1))::Array{T, N} where {T, N}
    hasnn(hyperparamsarray) || return x
    D = size(x, 1)
    D == first(ffjord.nn.layers).in_dims == last(ffjord.nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

    if N === 1
        _x = reshape(x, D, 1)
    elseif N === 2
        _x = x
    else
        _x = reshape(x, D, :)
    end

    ffjord_mdl = FFJORD(ffjord.nn, (zero(T), t), (D,), Tsit5(), ad=AutoForwardDiff(), basedist=nothing)
    ret = __backward_ffjord(ffjord_mdl, _x, hyperparamsarray.nn.params, ffjord.nns, rng)
    return reshape(ret, size(x)...)

end

function backwardffjord(rng::AbstractRNG, x::AbstractArray{T, N}, hyperparams::AbstractFCHyperparams{T, D}; t=T(1))::Array{T, N} where {T, D, N}
    hasnn(hyperparams) || return x
    return backwardffjord(rng, x, hyperparams._, hyperparams.ffjord, t=t)
end

function reflow(rng::AbstractRNG, clusters::AbstractVector{SetCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1)) where {T, D, E}

    cluster_sizes = length.(clusters)
    base2original = first(clusters).b2o
    @assert all([base2original === c.b2o for c in clusters])
    # origdata = reduce(hcat, [base2original[el] for c in clusters for el in c])
    origdata = Matrix(clusters, orig=true)

    deltalogpxs, new_basedata = forwardffjord(rng, origdata, hyperparamsarray, ffjord, t=t)

    new_base2original = Dict{SVector{D, T}, SVector{D, T}}(collect.(eachcol(new_basedata)) .=> collect.(eachcol(origdata)))
    new_clusters = [SetCluster(basedata, new_base2original) for basedata in chunk(new_basedata, cluster_sizes)]

    return (clusters=new_clusters, deltalogpxs=deltalogpxs)
end
function reflow(rng::AbstractRNG, clusters::AbstractVector{BitCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1)) where {T, D, E}
    base2original = first(clusters).b2o

    deltalogpxs, new_basedata = forwardffjord(rng, base2original[:, :, O], hyperparamsarray, ffjord, t=t)

    new_base2original = cat(new_basedata, base2original[:, :, O], dims=3)

    return (clusters=[BitCluster(cluster.mask, new_base2original) for cluster in clusters], deltalogpxs=deltalogpxs)
end
function reflow(rng::AbstractRNG, clusters::AbstractVector{IndexCluster{T, D, E}}, hyperparamsarray::ComponentArray{T}, ffjord::NamedTuple; t=T(1)) where {T, D, E}
    base2original = first(clusters).b2o

    deltalogpxs, new_basedata = forwardffjord(rng, base2original[:, :, O], hyperparamsarray, ffjord, t=t)

    new_base2original = cat(new_basedata, base2original[:, :, O], dims=3)

    return (clusters=[IndexCluster(cluster.elements, new_base2original) for cluster in clusters], deltalogpxs=deltalogpxs)
end

function dense_nn(nbnodes::Int...; act=tanh_fast, gain=0.2, uw=2.0)

    if act isa Tuple || act isa AbstractVector
        @assert length(act) == length(nbnodes)-1
    elseif act isa Function
        act = fill(act, length(nbnodes)-1)
    end

    layers = [Dense(nbnodes[i], nbnodes[i+1], act[i],
                    # init_bias=(i == length(nbnodes)-1) ? ((x...) -> uw*(rand64(x...) .- 1/2)) : zeros64,
                    init_bias=zeros64,
                    init_weight=zeros64,
                    # init_weight=orthogonal(Float64)
                    # init_weight=glorot_uniform(Float64, gain=gain)
                    # init_weight=kaiming_uniform(gain=gain)
                    ) for i in 1:length(nbnodes)-1]

    nn = Chain(layers...)

    return nn
end

function canonicalize(nn_params::ComponentArray{T}) where T

    nn_params = deepcopy(nn_params)

    # signs = sign.(nn_params.layer_1.weight[:, 1])
    signs = T[sign(ws[findlast(x -> abs(x) == maximum(abs.(ws)), ws)]) for ws in eachrow(nn_params.layer_1.weight)]
    signs[signs .== 0] .= one(T)
    nn_params.layer_1.weight .*= signs
    nn_params.layer_1.bias .*= signs
    nn_params.layer_2.weight .*= signs'

    # Append w for norm(w) in case of a
    # cosmically improbable tie.
    permidx = sortperm(eachrow(nn_params.layer_1.weight), by=w -> (norm(w), w), rev=true)
    # permidx = sortperm(eachrow(nn_params.layer_1.weight), rev=true)

    nn_params.layer_1.weight .= nn_params.layer_1.weight[permidx, :]
    nn_params.layer_1.bias .= nn_params.layer_1.bias[permidx]
    nn_params.layer_2.weight .= nn_params.layer_2.weight[:, permidx]

    fullperm = collect(1:length(nn_params))

    lm1w_idx = label2index(nn_params, "layer_1.weight")
    lm1w_perm = reshape(reshape(lm1w_idx, size(nn_params.layer_1.weight))[permidx, :], :)
    fullperm[lm1w_idx] .= fullperm[lm1w_perm]

    lm1b_idx = label2index(nn_params, "layer_1.bias")
    fullperm[lm1b_idx] .= fullperm[lm1b_idx][permidx]

    l2w_idx = label2index(nn_params, "layer_2.weight")
    l2w_perm = reshape(reshape(l2w_idx, size(nn_params.layer_2.weight)...)[:, permidx], :)
    fullperm[l2w_idx] .= fullperm[l2w_perm]

    signs = signs[permidx]
    fullsigns = ComponentArray(ones(T, length(nn_params)), first(getaxes(nn_params)))
    fullsigns.layer_1.weight .*= signs
    fullsigns.layer_1.bias .*= signs
    fullsigns.layer_2.weight .*= signs'

    return (nn_params=nn_params, permutation=fullperm, signs=fullsigns)

end
