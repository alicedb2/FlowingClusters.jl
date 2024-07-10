struct FCHyperparams
    _::ComponentArray{Float64}
    nn::Union{Nothing, Chain}
    nns::Union{Nothing, NamedTuple} # neural network state, which I still don't understand what it does
end

function FCHyperparams(d, nn=nothing)
    
    flatL_d = div(d * (d + 1), 2)
    
    nn_params, nn_state = Lux.setup(Xoshiro(), nn)
    nn_params = ComponentArray{Float64}(nn_params)

    if isnothing(nn)
        FCHyperparams(ComponentArray{Float64}(
            pyp=(alpha=7.77,),# sigma=0.0),
            niw=(mu=zeros(d), 
                 lambda=1.0, 
                 flatL=flatten(LowerTriangular(diagm(ones(d)))),
                 nu=d + 1)
            ),
        nothing, nothing)
    else
        nn_state = (model=nn_state, regularize=false, monte_carlo=false)
        return FCHyperparams(ComponentArray{Float64}(
                    pyp=(alpha=7.77,),# sigma=0.0),
                    niw=(mu=zeros(d), 
                        lambda=1.0, 
                        flatL=flatten(LowerTriangular(diagm(ones(d)))),
                        nu=d + 1), 
                    nn=(params=nn_params, 
                        t=(alpha=1.0, scale=1.0)
                        )
                    ),
                nn,
                nn_state)
    end
end

function FFJORD(hyperparams::FCHyperparams)
    return FFJORD(hyperparams.nn, (0.0, 1.0), (dimension(hyperparams),), Tsit5(), ad=AutoForwardDiff())
end

dimension(hyperparams::FCHyperparams) = size(hyperparams._.niw.mu, 1)
function modeldimension(hyperparams::FCHyperparams; include_nn=true) 
    dim = size(hyperparams._, 1)
    if hasnn(hyperparams) && !include_nn
        dim -= size(hyperparams._.nn)
    end
    return dim
end

hasnn(hyperparams::FCHyperparams) = !isnothing(hyperparams.nn)

function transform(hparray::ComponentArray{Float64})
    thp = similar(hparray)
    thp.alpha = log(thp.alpha)
    thp.lambda = log(thp.lambda)
    thp.nu = log(thp.nu - d + 1)
    thp.nn.t.alpha = log(thp.nn.t.alpha)
    thp.nn.t.scale = log(thp.nn.t.scale)
    return thp
end

function backtransform(transformedhparray::ComponentArray{Float64})
    hp = similar(transformedhparray)
    hp.alpha = exp(hp.alpha)
    hp.lambda = exp(hp.lambda)
    hp.nu = exp(hp.nu) + d - 1
    hp.nn.t.alpha = exp(hp.nn.t.alpha)
    hp.nn.t.scale = exp(hp.nn.t.scale)
    return hp
end

function ij(flatk::Int64)
    # Int64 will fail if flat_k is not a triangular number
    i = Int64(ceil(Int64, 1/2 * (sqrt(1 + 8 * flatk) - 1)))
    j = Int64(flatk - i * (i - 1)/2)
    return (i, j)
end

function flatk(i::Int64, j::Int64)
    return div(i * (i - 1), 2) + j
end

function squaredim(flatL::AbstractVector{Float64})
    return Int64((sqrt(1 + 8 * size(flatL, 1)) - 1) / 2)
end

function foldpsi(flatL::AbstractVector{Float64})
    d = squaredim(flatL)
    L = foldL(flatL)
    return L * L'
end

function flatten(L::LowerTriangular{Float64})::Vector{Float64}

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


function foldL(flatL::AbstractVector{Float64})::LowerTriangular

    d = squaredim(flatL)

    L = LowerTriangular(Array{Float64}(undef, d, d))

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

project_vec(vec::Vector{Float64}, proj::Matrix{Float64}) = proj * vec
project_vec(vec::Vector{Float64}, dims::Vector{Int64}) = dims_to_proj(dims, size(vec, 1)) * vec

project_mat(mat::Matrix{Float64}, proj::Matrix{Float64}) = proj * mat * proj'

function project_mat(mat::Matrix{Float64}, dims::Vector{Int64})
    d = size(mat, 1)
    proj = dims_to_proj(dims, d)
    return proj * mat * proj'
end

function Base.show(io::IO, hyperparams::FCHyperparams)
    println(io, "FCHyperparams")
    println(io, "  pyp")
    println(io, "    alpha: $(round(hyperparams._.pyp.alpha, digits=3))")
    # println(io, "    sigma: $(round(hyperparams._.pyp.sigma, digits=3))")
    println(io, "  niw")
    println(io, "    mu: $(round.(hyperparams._.niw.mu, digits=3))")
    println(io, "    lambda: $(round(hyperparams._.niw.lambda, digits=3))")
    println(io, "    flatL: $(round.(hyperparams._.niw.flatL, digits=3))")
    println(io, "    psi: $(round.(foldpsi(hyperparams._.niw.flatL), digits=3))")
    println(io, "    nu: $(round(hyperparams._.niw.nu, digits=3))")
    println(io, "  nn")
    println(io, "    params: $(map(x->round(x, digits=3), hyperparams._.nn.params))")
    println(io, "    t")
    println(io, "      alpha: $(round(hyperparams._.nn.t.alpha, digits=3))")
    print(io,   "      scale: $(round(hyperparams._.nn.t.scale, digits=3))")
end