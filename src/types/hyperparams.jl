abstract type AbstractFCHyperparams{T} end

struct FCHyperparams{T} <: AbstractFCHyperparams{T}
    _::ComponentArray{T}
end

struct FCHyperparamsFFJORD{T} <: AbstractFCHyperparams{T}
    _::ComponentArray{T}
    nn::NamedTuple
    # neural network state, which I still don't understand what it does
    # everything is already in the parameters.
end

function FCHyperparams{T}(d::Int) where T
    
    flatL_d = div(d * (d + 1), 2)

    return FCHyperparams(ComponentArray{T}(
                    pyp=(alpha=T(7.77),),# sigma=0.0),
                    niw=(mu=zeros(T, d), 
                        lambda=one(T), 
                        flatL=flatten(LowerTriangular{T}(diagm(ones(T, d)))),
                        nu=T(d + 1))
                    )
                )

end

function FCHyperparamsFFJORD{T}(d::Int, nn::Chain) where T
    nn_params, nn_state = Lux.setup(Xoshiro(), nn)
    nn_params = ComponentArray{T}(nn_params)    
    nn_state = (model=nn_state, regularize=false, monte_carlo=false)
    
    ffjord_model = FFJORD(nn, (zero(T), one(T)), (d,), Tsit5(), ad=AutoForwardDiff())

    return FCHyperparamsFFJORD{T}(ComponentArray{T}(
                    pyp=(alpha=T(7.77),),# sigma=0.0),
                    niw=(mu=zeros(T, d), 
                        lambda=one(T), 
                        flatL=flatten(LowerTriangular{T}(diagm(ones(T, d)))),
                        nu=T(d + 1)), 
                    nn=(params=nn_params, 
                        t=(alpha=T(1.0), scale=T(1.0))
                        )
                    ),
                (nn=nn, nns=nn_state)
            )
end

function forwardffjord(x::AbstractArray{T}, hyperparams::FCHyperparamsFFJORD{T}, ffjord::FFJORD) where T
    ret, _ = model(x, hyperparams._.nn.params, hyperparams.nn.nns)
    return (; logps=ret.logp, delta_logp=ret.delta_logps, z=z)
end

function forwardffjord(x::AbstractArray{T}, hyperparams::FCHyperparamsFFJORD{T}) where T
    ffjord_model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
    return forwardffjord(x, hyperparams, ffjord_model)
end

backwardffjord(x::AbstractVector, hyperparams::FCHyperparamsFFJORD, ffjord::FFJORD) = reshape(backwardffjord(reshape(x, :, 1), hyperparams), :)
backwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T}, ffjord::FFJORD) where T = Matrix{T}(__backward_ffjord(model, x, hyperparams.nn_params, hyperparams.nn_state))

function backwardffjord(x::AbstractVector{T}, hyperparams::FCHyperparamsFFJORD{T}) where T
    ffjord_model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
    return return backwardffjord(x, hyperparams, ffjord_model)
end

function backwardffjord(x::AbstractMatrix{T}, hyperparams::FCHyperparamsFFJORD{T}) where T
    model = FFJORD(hyperparams.nn.nn, (zero(T), one(T)), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
    return backwardffjord(x, hyperparams, model)
end


dimension(hyperparams::AbstractFCHyperparams) = size(hyperparams._.niw.mu, 1)

function modeldimension(hyperparams::FCHyperparamsFFJORD; include_nn=true) 
    dim = size(hyperparams._, 1)
    dim -= !include_NN ? 0 : size(hyperparams._.nn, 1)
    return dim
end

modeldimension(hyperparams::FCHyperparams) = size(hyperparams._, 1)

hasnn(hyperparams::AbstractFCHyperparams{T}) where T = typeof(hyperparams) === FCHyperparamsFFJORD{T}

function transform(hparray::ComponentArray{T})::ComponentArray{T} where T
    thp = similar(hparray)
    thp.alpha = log(thp.alpha)
    thp.lambda = log(thp.lambda)
    thp.nu = log(thp.nu - d + 1)
    if :nn in keys(hparray)
        thp.nn.t.alpha = log(thp.nn.t.alpha)
        thp.nn.t.scale = log(thp.nn.t.scale)
    end
    return thp
end

function backtransform(transformedhparray::ComponentArray{T})::ComponentArray{T} where T
    hp = similar(transformedhparray)
    hp.alpha = exp(hp.alpha)
    hp.lambda = exp(hp.lambda)
    hp.nu = exp(hp.nu) + d - 1
    if :nn in keys(transformedhparray)
        hp.nn.t.alpha = exp(hp.nn.t.alpha)
        hp.nn.t.scale = exp(hp.nn.t.scale)
    end
    return hp
end

function ij(flatk::Int)
    # Int64 will fail if flat_k is not a triangular number
    i = Int64(ceil(Int64, 1/2 * (sqrt(1 + 8 * flatk) - 1)))
    j = Int64(flatk - i * (i - 1)/2)
    return (i, j)
end

function flatk(i::Int, j::Int)
    return div(i * (i - 1), 2) + j
end

function squaredim(flatL::AbstractVector{T})::Int where T
    return Int((sqrt(1 + 8 * size(flatL, 1)) - 1) / 2)
end

function foldpsi(flatL::AbstractVector{T}) where T
    d = squaredim(flatL)
    L = foldL(flatL)
    return L * L'
end

function flatten(L::LowerTriangular{T})::Vector{T} where T

    d = size(L, 1)

    flatL = Array{Float64}(undef, Int64(d * (d + 1) / 2))

    idx = 1
    for i in 1:d
        for j in 1:i
            flatL[idx] = L[i, j]
            idx += 1
        end
    end

    return flatL
end


function foldL(flatL::AbstractVector{T})::LowerTriangular{T} where T

    d = squaredim(flatL)

    L = LowerTriangular(Array{T}(undef, d, d))

    # The order is row major
    idx = 1
    for i in 1:d
        for j in 1:i
            L[i, j] = flatL[idx]
            idx += 1
        end
    end

    return L
end

project_vec(vec::AbstractVector{T}, proj::Matrix{T}) where T = proj * vec
project_vec(vec::AbstractVector{T}, dims::AbstractVector{Int}) where T = dims_to_proj(dims, size(vec, 1)) * vec

project_mat(mat::AbstractMatrix{T}, proj::AbstractMatrix{T}) where T = proj * mat * proj'

function project_mat(mat::AbstractMatrix{T}, dims::AbstractVector{Int}) where T
    d = size(mat, 1)
    proj = dims_to_proj(dims, d)
    return proj * mat * proj'
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