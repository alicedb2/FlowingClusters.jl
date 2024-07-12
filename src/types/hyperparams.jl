abstract type AbstractFCHyperparams{T, D} end

struct FCHyperparams{T, D} <: AbstractFCHyperparams{T, D}
    _::ComponentArray{T}
end

struct FCHyperparamsFFJORD{T, D} <: AbstractFCHyperparams{T, D}
    _::ComponentArray{T}
    nn::NamedTuple
    # neural network state, which I still don't understand what it does
    # everything is already in the parameters.
end

function FCHyperparams(D::Int, ::Type{T}, nn::Union{Nothing, Chain}=nothing) where {T <: AbstractFloat}
    
    if isnothing(nn)
        return FCHyperparams{T, D}(
            ComponentArray{T}(
                        pyp=(alpha=T(7.77),),# sigma=0.0),
                        niw=(mu=zeros(T, D), 
                             lambda=one(T), 
                             flatL=unfold(LowerTriangular{T}(I(D))),
                             nu=T(D + 1))
                )
            )
    else

        D == first(nn.layers).in_dims == last(nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

        nn_params, nn_state = Lux.setup(Xoshiro(), nn)
        nn_params = ComponentArray{T}(nn_params)
        nn_state = (model=nn_state, regularize=false, monte_carlo=false)
        
        return FCHyperparamsFFJORD{T, D}(ComponentArray{T}(
                        pyp=(alpha=T(7.77),),# sigma=0.0),
                        niw=(mu=zeros(T, D), 
                            lambda=one(T), 
                            flatL=unfold(LowerTriangular{T}(I(D))),
                            nu=T(D + 1)), 
                        nn=(params=nn_params, 
                            t=(alpha=T(1), scale=T(1))
                            )
                        ),
                    (nn=nn, nns=nn_state)
                )
    end
end

function forwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T, D}, ffjord::FFJORD) where {T, D}
    ret, _ = ffjord(x, hyperparams._.nn.params, hyperparams.nn.nns)
    return (; logps=ret.logp, delta_logps=ret.delta_logps, z=z)
end

function forwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T, D}) where {T, D}
    ffjord_model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (D,), Tsit5(), ad=AutoForwardDiff())
    return forwardffjord(x, hyperparams, ffjord_model)
end

backwardffjord(x::AbstractVector{T}, hyperparams::FCHyperparamsFFJORD{T, D}, ffjord::FFJORD) where {T, D} = reshape(backwardffjord(ffjord, reshape(x, D, 1), hyperparams), D)
backwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T, D}, ffjord::FFJORD) where {T, D} = Matrix{T}(__backward_ffjord(ffjord, x, hyperparams.nn_params, hyperparams.nn_state))

function backwardffjord(x::AbstractVector{T}, hyperparams::FCHyperparamsFFJORD{T, D}) where {T, D}
    ffjord_model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (D,), Tsit5(), ad=AutoForwardDiff())
    return return backwardffjord(x, hyperparams, ffjord_model)
end

function backwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T, D}) where {T, D}
    model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (D,), Tsit5(), ad=AutoForwardDiff())
    return backwardffjord(x, hyperparams, model)
end

dimension(::AbstractFCHyperparams{T, D}) where {T, D} = D

function modeldimension(hyperparams::FCHyperparamsFFJORD; include_nn=true) 
    dim = size(hyperparams._, 1)
    dim -= !include_nn ? 0 : size(hyperparams._.nn, 1)
    return dim
end

modeldimension(hyperparams::FCHyperparams) = size(hyperparams._, 1)

hasnn(hyperparams::AbstractFCHyperparams) = hyperparams isa FCHyperparamsFFJORD
hasnn(hyperparamsarray::ComponentArray) = :nn in keys(hyperparamsarray)

transform(hparray::ComponentArray) = backtransform!(similar(hparray))
backtransform(transformedhparray::ComponentArray) = backtransform!(similar(transformedhparray))

function transform!(hparray::ComponentArray)
    hparray.alpha .= log(hparray.alpha)
    hparray.lambda .= log(hparray.lambda)
    hparray.nu .= log(hparray.nu - d + 1)
    if hasnn(hparray)
        hparray.nn.t.alpha .= log(hparray.nn.t.alpha)
        hparray.nn.t.scale .= log(hparray.nn.t.scale)
    end
    return hparray
end

function backtransform!(transformedhparray::ComponentArray)
    transformedhparray.alpha .= exp(transformedhparray.alpha)
    transformedhparray.lambda .= exp(transformedhparray.lambda)
    transformedhparray.nu .= exp(transformedhparray.nu) + d - 1
    if hasnn(transformedhparray)
        transformedhparray.nn.t.alpha .= exp(transformedhparray.nn.t.alpha)
        transformedhparray.nn.t.scale .= exp(transformedhparray.nn.t.scale)
    end
    return transformedhparray
end

function ij(flatk::Int)
    i = ceil(Int, (sqrt(1 + 8 * flatk) - 1) / 2)
    j = flatk - div(i * (i - 1), 2)
    return i, j
end

function flatk(i::Int, j::Int)
    return div(i * (i - 1), 2) + j
end

function squaredim(flatL::AbstractVector)
    D = size(flatL, 1)
    return div(Int(sqrt(1 + 8 * D)) - 1, 2)
end

# This eventually hits BLAS, no need to try to optimize it with for loops
function foldpsi(flatL::AbstractVector)
    L = foldL(flatL)
    return L * L'
end

function unfold(L::LowerTriangular{T}) where T
    d = size(L, 1)
    flatL = Array{T}(undef, div(d * (d + 1), 2))
    idx = 1
    @inbounds for i in 1:d
        @inbounds for j in 1:i
            flatL[idx] = L[i, j]
            idx += 1
        end
    end
    return flatL
end

function LinearAlgebra.LowerTriangular(flatL::AbstractVector{T}) where T
    d = squaredim(flatL)
    L = LowerTriangular{T}(Array{T}(undef, d, d))
    
    # The order is row major. This is important because
    # it allows the conversion k -> (i, j)
    # and (i, j) -> k without neither a for-loop
    # nor a passing dimension d as argument
    idx = 1
    for i in 1:d
        for j in 1:i
            L[i, j] = flatL[idx]
            idx += 1
        end
    end
    return L
end

function Base.show(io::IO, hyperparams::AbstractFCHyperparams)
    println(io, "$(typeof(hyperparams))(model dimension: $(modeldimension(hyperparams)))")
    println(io, "  pyp")
    println(io, "    alpha: $(round(hyperparams._.pyp.alpha, digits=3))")
    # println(io, "    sigma: $(round(hyperparams._.pyp.sigma, digits=3))")
    println(io, "  niw")
    println(io, "    mu: $(round.(hyperparams._.niw.mu, digits=3))")
    println(io, "    lambda: $(round(hyperparams._.niw.lambda, digits=3))")
    println(io, "    flatL: $(round.(hyperparams._.niw.flatL, digits=3))")
    println(io, "    psi: $(round.(foldpsi(hyperparams._.niw.flatL), digits=3))")
    print(io,   "    nu: $(round(hyperparams._.niw.nu, digits=3))")
    if hasnn(hyperparams)
        println(io)
        println(io, "  nn")
        println(io, "    params: $(map(x->round(x, digits=3), hyperparams._.nn.params))")
        println(io, "    t")
        println(io, "      alpha: $(round(hyperparams._.nn.t.alpha, digits=3))")
        print(io,   "      scale: $(round(hyperparams._.nn.t.scale, digits=3))")
    end
end