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

function FCHyperparams(::Type{T}, D::Int, nn=nothing; rng::Union{Nothing, AbstractRNG}=nothing) where {T <: AbstractFloat}

    hpparams = ComponentArray{T}(
                    crp=(alpha=one(T),
                         niw=(mu=zeros(T, D),
                             lambda=T(0.1^2),
                             flatL=unfold(LowerTriangular{T}(0.01*I(D))),
                             nu=T(D)
                         )
                    ),
                )

    if isnothing(nn)
        return FCHyperparams{T, D}(hpparams)
    else
        rng isa AbstractRNG || throw(ArgumentError("You must provide a random number generator when using FFJORD"))
        # D == first(nn.layers).in_dims == last(nn.layers).out_dims || throw(ArgumentError("The input and output dimensions of the neural network must be the same as the dimension of the data"))

        nn_params, nn_state = Lux.setup(rng, nn)
        nn_state = (model=nn_state, regularize=false, monte_carlo=false)
        ffjord = (nn=nn, nns=nn_state)
        nn_params = ComponentArray{T}(nn_params)

        hpparams = vcat(hpparams, ComponentArray{T}(nn=(params=nn_params, logt=zero(T))))

        return FCHyperparamsFFJORD{T, D}(hpparams, ffjord)
    end
end

datadimension(::AbstractFCHyperparams{T, D}) where {T, D} = D
datadimension(hyperparamsarray::ComponentArray) = size(hyperparamsarray.crp.niw.mu, 1)

modeldimension(hpparams::ComponentArray; include_nn=true) = size(hpparams.crp, 1) + (include_nn && hasnn(hpparams) ? size(hpparams.nn, 1) : 0)
modeldimension(hyperparams::AbstractFCHyperparams; include_nn=true) = modeldimension(hyperparams._, include_nn=include_nn)

hasnn(hyperparams::AbstractFCHyperparams) = hyperparams isa FCHyperparamsFFJORD
hasnn(hyperparamsarray::ComponentArray) = :nn in keys(hyperparamsarray)

transform(hparray::ComponentArray) = transform!(copy(hparray))
backtransform(transformedhparray::ComponentArray) = backtransform!(copy(transformedhparray))

function transform!(hparray::ComponentArray)
    hparray.crp.alpha = log(hparray.crp.alpha)
    hparray.crp.niw.lambda = log(hparray.crp.niw.lambda)
    hparray.crp.niw.nu = log(hparray.crp.niw.nu - datadimension(hparray) + 1)

    return hparray
end

function backtransform!(transformedhparray::ComponentArray)
    transformedhparray.crp.alpha = exp(transformedhparray.crp.alpha)
    transformedhparray.crp.niw.lambda = exp(transformedhparray.crp.niw.lambda)
    transformedhparray.crp.niw.nu = exp(transformedhparray.crp.niw.nu) + datadimension(transformedhparray) - 1

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
    println(io, "  crp")
    println(io, "    alpha: $(round(hyperparams._.crp.alpha, digits=3))")
    # println(io, "    sigma: $(round(hyperparams._.crp.sigma, digits=3))")
    println(io, "    niw")
    println(io, "      mu: $(round.(hyperparams._.crp.niw.mu, digits=3))")
    println(io, "      lambda: $(round(hyperparams._.crp.niw.lambda, digits=3))")
    println(io, "      flatL: $(round.(hyperparams._.crp.niw.flatL, digits=3))")
    println(io, "      psi: $(round.(foldpsi(hyperparams._.crp.niw.flatL), digits=3))")
    print(io,   "      nu: $(round(hyperparams._.crp.niw.nu, digits=3))")
    if hasnn(hyperparams)
        println(io)
        println(io, "  nn")
        print(io,   "    params: $(map(x->round(x, digits=3), hyperparams._.nn.params))")
        # println(io, "    prior")
        # println(io, "              mu0: $(round(hyperparams._.nn.prior.mu0, digits=3))")
        # println(io, "      log lambda0: $(round(hyperparams._.nn.prior.lambda0, digits=3))")
        # println(io, "       log alpha0: $(round(hyperparams._.nn.prior.alpha0, digits=3))")
        # print(io,   "        log beta0: $(round(hyperparams._.nn.prior.beta0, digits=3))")
    end
end

function perturb!(hyperparams::AbstractFCHyperparams{T, D}; logstep::T=one(T)/3, rng=default_rng()) where {T, D}
    return perturb!(rng, hyperparams, logstep=logstep)
end

function perturb!(rng::AbstractRNG, hyperparams::AbstractFCHyperparams{T, D}; logstep::T=one(T)/3) where {T, D}
    backtransform!(transform!(hyperparams._) .+= logstep * randn(rng, length(hyperparams._)))
    return hyperparams
end

function niwparams(hyperparams::AbstractFCHyperparams{T, D}; psi=true) where {T, D}
    if psi
        return hyperparams._.crp.niw.mu, hyperparams._.crp.niw.lambda, foldpsi(hyperparams._.crp.niw.flatL), hyperparams._.crp.niw.nu
    else
        return hyperparams._.crp.niw.mu, hyperparams._.crp.niw.lambda, hyperparams._.crp.niw.flatL, hyperparams._.crp.niw.nu
    end
end

function niwparams(hyperparamsarray::ComponentArray{T}; psi=true) where T
    if psi
        return hyperparamsarray.crp.niw.mu, hyperparamsarray.crp.niw.lambda, foldpsi(hyperparamsarray.crp.niw.flatL), hyperparamsarray.crp.niw.nu
    else
        return hyperparamsarray.crp.niw.mu, hyperparamsarray.crp.niw.lambda, hyperparamsarray.crp.niw.flatL, hyperparamsarray.crp.niw.nu
    end
end