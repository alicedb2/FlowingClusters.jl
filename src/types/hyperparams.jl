abstract type AbstractFCHyperparams{T, D} end

# That janky underscore is really ugly,
# I'm sorry, I'm not clever enough to come up
# with the idiomatic clean way to do this.
struct FCHyperparams{T, D} <: AbstractFCHyperparams{T, D}
    _::ComponentArray{T}
end

struct FCHyperparamsFFJORD{T, D} <: AbstractFCHyperparams{T, D}
    _::ComponentArray{T}
    ffjord::NamedTuple
    # neural network state, which I still don't understand what it does
    # everything is already in the parameters.
end

function FCHyperparams(::Type{T}, D::Int, nn::Union{Nothing, Chain}=nothing; rng=default_rng()) where {T <: AbstractFloat}

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

        nn_params, nn_state = Lux.setup(rng, nn)
        nn_params = ComponentArray{T}(nn_params)
        nn_state = (model=nn_state, regularize=false, monte_carlo=false)

        return FCHyperparamsFFJORD{T, D}(ComponentArray{T}(
                        pyp=(alpha=T(7.77),),# sigma=0.0),
                        niw=(mu=zeros(T, D),
                            lambda=one(T),
                            flatL=unfold(LowerTriangular{T}(I(D))),
                            nu=T(D + 1)),
                        nn=(params=nn_params,
                            prior=(alpha=T(1), scale=T(1))
                            )
                        ),
                    (nn=nn, nns=nn_state)
                )
    end
end

datadimension(::AbstractFCHyperparams{T, D}) where {T, D} = D

function modeldimension(hyperparams::FCHyperparamsFFJORD; include_nn=true)
    dim = size(hyperparams._, 1)
    dim -= !include_nn ? 0 : size(hyperparams._.nn, 1)
    return dim
end

modeldimension(hyperparams::FCHyperparams) = size(hyperparams._, 1)

hasnn(hyperparams::AbstractFCHyperparams) = hyperparams isa FCHyperparamsFFJORD
hasnn(hyperparamsarray::ComponentArray) = :nn in keys(hyperparamsarray)

transform(hparray::ComponentArray) = transform!(copy(hparray))
backtransform(transformedhparray::ComponentArray) = backtransform!(copy(transformedhparray))

function transform!(hparray::ComponentArray)
    hparray.pyp.alpha = log(hparray.pyp.alpha)
    hparray.niw.lambda = log(hparray.niw.lambda)
    hparray.niw.nu = log(hparray.niw.nu - size(hparray.niw.mu, 1) + 1)
    if hasnn(hparray)
        hparray.nn.prior.alpha = log(hparray.nn.prior.alpha)
        hparray.nn.prior.scale = log(hparray.nn.prior.scale)
    end
    return hparray
end

function backtransform!(transformedhparray::ComponentArray)
    transformedhparray.pyp.alpha = exp(transformedhparray.pyp.alpha)
    transformedhparray.niw.lambda = exp(transformedhparray.niw.lambda)
    transformedhparray.niw.nu = exp(transformedhparray.niw.nu) + size(transformedhparray.niw.mu, 1) - 1
    if hasnn(transformedhparray)
        transformedhparray.nn.prior.alpha = exp(transformedhparray.nn.prior.alpha)
        transformedhparray.nn.prior.scale = exp(transformedhparray.nn.prior.scale)
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
    L = LowerTriangular(flatL)
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
        println(io, "      alpha: $(round(hyperparams._.nn.prior.alpha, digits=3))")
        print(io,   "      scale: $(round(hyperparams._.nn.prior.scale, digits=3))")
    end
end

function perturb!(hyperparams::AbstractFCHyperparams{T, D}; logstep::T=one(T)/3) where {T, D}
    return perturb!(default_rng(), hyperparams, logstep=logstep)
end

function perturb!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}; logstep::T=one(T)/3) where {T, D}
    backtransform!(transform!(hyperparams._) .+= logstep * randn(rng, length(hyperparams._)))
    return hyperparams
end

function niwparams(hyperparams::AbstractFCHyperparams{T, D}; psi=true) where {T, D}
    if psi
        return hyperparams._.niw.mu, hyperparams._.niw.lambda, foldpsi(hyperparams._.niw.flatL), hyperparams._.niw.nu
    else
        return hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.flatL, hyperparams._.niw.nu
    end
end

function niwparams(hyperparamsarray::ComponentArray{T}; psi=true) where T
    if psi
        return hyperparamsarray.niw.mu, hyperparamsarray.niw.lambda, foldpsi(hyperparamsarray.niw.flatL), hyperparamsarray.niw.nu
    else
        return hyperparamsarray.niw.mu, hyperparamsarray.niw.lambda, hyperparamsarray.niw.flatL, hyperparamsarray.niw.nu
    end
end