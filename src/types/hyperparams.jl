mutable struct MNCRPHyperparams
    alpha::Float64
    mu::Vector{Float64}
    lambda::Float64
    flatL::Vector{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    psi::Matrix{Float64}
    nu::Float64

    diagnostics::Diagnostics

    nn::Union{Nothing, Chain}
    nn_params::Union{Nothing, ComponentArray}
    nn_state::Union{Nothing, NamedTuple}

    nn_alpha::Float64
    nn_scale::Float64

end

function Base.show(io::IO, h::MNCRPHyperparams)
    d = dimension(h)
    D = 3 + d + div(d * (d + 1), 2)
    if h.nn !== nothing
        nn_dim = size(h.nn_params, 1)
        D += nn_dim + 2
    end
    str = "MNCRPHyperparams(d=$d, D=$D from sum(1, $d, 1, $(div(d * (d + 1), 2)), 1" * (h.nn !== nothing ? ", $nn_dim, 1, 1" : "") * "))"
    print(io, str)
end

function MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu; ffjord_nn=nothing, ffjord_nn_alpha=nothing, ffjord_nn_scale=nothing)
    d = size(mu, 1)

    if ffjord_nn !== nothing
        @assert d == ffjord_nn[1].in_dims == ffjord_nn[end].out_dims
        nn_params, nn_state = Lux.setup(Xoshiro(), ffjord_nn)
        nn_params = ComponentArray(nn_params)
        nn_state = (model=nn_state, regularize=false, monte_carlo=false)
        nn_alpha = ffjord_nn_alpha
        nn_scale = ffjord_nn_scale
    else
        nn_params = nothing
        nn_state = nothing
        nn_alpha = 1.0
        nn_scale = 1.0
    end

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu, Diagnostics(d, nn_params=nn_params), ffjord_nn, nn_params, nn_state, nn_alpha, nn_scale)
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, flatL::Vector{Float64}, nu::Float64; ffjord_nn=nothing, ffjord_nn_alpha=1.0, ffjord_nn_scale=1.0)
    d = size(mu, 1)
    flatL_d = div(d * (d + 1), 2)
    if size(flatL, 1) != flatL_d
        error("Dimension mismatch, flatL should have length $flatL_d")
    end

    L = foldflat(flatL)
    psi = L * L'

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu, ffjord_nn=ffjord_nn, ffjord_nn_alpha=ffjord_nn_alpha, ffjord_nn_scale=ffjord_nn_scale)
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, L::LowerTriangular{Float64}, nu::Float64; ffjord_nn=nothing, ffjord_nn_alpha=1.0, ffjord_nn_scale=1.0)
    d = size(mu, 1)
    if !(d == size(L, 1))
        error("Dimension mismatch, L should have dimension $d x $d")
    end

    psi = L * L'
    flatL = flatten(L)

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu, ffjord_nn=ffjord_nn, ffjord_nn_alpha=ffjord_nn_alpha, ffjord_nn_scale=ffjord_nn_scale)
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64; ffjord_nn=nothing, ffjord_nn_alpha=1.0, ffjord_nn_scale=1.0)
    d = size(mu, 1)
    if !(d == size(psi, 1) == size(psi, 2))
        error("Dimension mismatch, L should have dimension $d x $d")
    end

    L = cholesky(psi).L
    flatL = flatten(L)

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu, ffjord_nn=ffjord_nn, ffjord_nn_alpha=ffjord_nn_alpha, ffjord_nn_scale=ffjord_nn_scale)
end

function MNCRPHyperparams(d::Int64; ffjord_nn=nothing)
    return MNCRPHyperparams(
        7.77,
        zeros(d),
        1.0,
        LowerTriangular(diagm(fill(0.1, d))),
        d + 1.0,
        ffjord_nn=ffjord_nn, ffjord_nn_alpha=1.0, ffjord_nn_scale=1.0)
end

function clear_diagnostics!(hyperparams::MNCRPHyperparams; clearhyperparams=true, clearsplitmerge=true, clearnn=true, keepstepscale=true)

    clear_diagnostics!(hyperparams.diagnostics, clearhyperparams=clearhyperparams, clearsplitmerge=clearsplitmerge, clearnn=clearnn, keepstepscale=keepstepscale)

    return hyperparams
end

function Base.collect(hyperparams::MNCRPHyperparams)
    return [hyperparams.alpha, hyperparams.mu, hyperparams.lambda, hyperparams.psi, hyperparams.nu]
end

function ij(flat_k::Int64)
    i = Int64(ceil(1/2 * (sqrt(1 + 8 * flat_k) - 1)))
    j = Int64(flat_k - i * (i - 1)/2)
    return (i, j)
end

function flatten(L::LowerTriangular{Float64})

    d = size(L, 1)

    flatL = zeros(Int64(d * (d + 1) / 2))

    # Fortran flashbacks
    idx = 1
    for i in 1:d
        for j in 1:i
            flatL[idx] = L[i, j]
            idx += 1
        end
    end

    return flatL
end



function foldflat(flatL::Vector{Float64})

    n = size(flatL, 1)

    # Recover the dimension of a matrix
    # from the length of the vector
    # containing elements of the diag + lower triangular
    # part of the matrix. The condition is that
    # length of vector == #els diagonal + #els lower triangular part
    # i.e N == d + (d² - d) / 2
    # Will fail at Int64() if this condition
    # cannot be satisfied for N and d integers
    # Basically d is the "triangular root"
    d = Int64((sqrt(1 + 8 * n) - 1) / 2)

    L = LowerTriangular(zeros(d, d))

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


function dimension(hyperparams::MNCRPHyperparams)
    @assert size(hyperparams.mu, 1) == size(hyperparams.psi, 1) == size(hyperparams.psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
    return size(hyperparams.mu, 1)
end

function param_dimension(hyperparams::MNCRPHyperparams; include_nn=true)
    d = dimension(hyperparams)
    D = 3 + d + div(d * (d + 1), 2)
    if hyperparams.nn_params !== nothing && include_nn
        D += size(hyperparams.nn_params, 1) + 2
    end
    return D
end

function flatL!(hyperparams::MNCRPHyperparams, value::Vector{Float64})
    d = dimension(hyperparams)
    if (2 * size(value, 1) != d * (d + 1))
        error("Dimension mismatch, value should have length $(Int64(d*(d+1)/2))")
    end
    hyperparams.flatL = deepcopy(value) # Just making sure?
    hyperparams.L = foldflat(hyperparams.flatL)
    hyperparams.psi = hyperparams.L * hyperparams.L'
end

function L!(hyperparams::MNCRPHyperparams, value::T) where {T <: AbstractMatrix{Float64}}
    d = dimension(hyperparams)
    if !(d == size(value, 1) == size(value, 2))
        error("Dimension mismatch, value shoud have size $d x $d")
    end
    hyperparams.L = LowerTriangular(value)
    hyperparams.flatL = flatten(hyperparams.L)
    hyperparams.psi = hyperparams.L * hyperparams.L'
end

function psi!(hyperparams::MNCRPHyperparams, value::Matrix{Float64})
    d = dimension(hyperparams)
    if !(d == size(value, 1) == size(value, 2))
        error("Dimension mismatch, value should have size $d x $d")
    end
    hyperparams.L = cholesky(value).L
    hyperparams.flatL = flatten(hyperparams.L)
    hyperparams.psi = deepcopy(value)
end

project_vec(vec::Vector{Float64}, proj::Matrix{Float64}) = proj * vec
project_vec(vec::Vector{Float64}, dims::Vector{Int64}) = dims_to_proj(dims, size(vec, 1)) * vec

project_mat(mat::Matrix{Float64}, proj::Matrix{Float64}) = proj * mat * proj'

function project_mat(mat::Matrix{Float64}, dims::Vector{Int64})
    d = size(mat, 1)
    proj = dims_to_proj(dims, d)
    return proj * mat * proj'
end

function project_hyperparams(hyperparams::MNCRPHyperparams, proj::Matrix{Float64})
    proj_hp = deepcopy(hyperparams)
    proj_hp.mu = proj * proj_hp.mu
    proj_hp.psi = proj * proj_hp.psi * proj'
    return proj_hp
end


function project_hyperparams(hyperparams::MNCRPHyperparams, dims::Vector{Int64})
    d = dimension(hyperparams)
    return project_hyperparams(hyperparams, dims_to_proj(dims, d))
end


function pack(theta::Vector{Float64}; backtransform=true)::Tuple{Float64, Vector{Float64}, Float64, Vector{Float64}, LowerTriangular{Float64}, Matrix{Float64}, Float64}
    d = Int64((sqrt(8 * length(theta) - 15) - 3)/2)

    alpha = theta[1]
    mu = theta[2:(2 + d - 1)]
    lambda = theta[2 + d]
    flatL = theta[(2 + d + 1):(2 + d + div(d * (d + 1), 2))]
    nu = theta[end]

    if backtransform
        alpha = exp(alpha)
        lambda = exp(lambda)
        nu = d - 1 + exp(nu)
        # diag_idx = div.((1:d) .* (2:d+1), 2)
        # flatL[diag_idx] = exp.(flatL[diag_idx])
    end

    L = foldflat(flatL)
    psi = L * L'

    return (alpha, mu, lambda, flatL, L, psi, nu)
end


function unpack(hyperparams::MNCRPHyperparams; transform=true)::Vector{Float64}
    d = dimension(hyperparams)
    alpha = hyperparams.alpha
    lambda = hyperparams.lambda
    nu = hyperparams.nu

    if transform
        alpha = log(alpha)
        lambda = log(lambda)
        nu = log(nu - d + 1)
        # diag_idx = div.((1:d) .* (2:d+1), 2)
        # flatL[diag_idx] = log.(flatL[diag_idx])
    end

    return vcat(alpha, hyperparams.mu, lambda, hyperparams.flatL, nu)

end

# We gotta go fast!
# Propagate what theta[i] locally affects
# This is mostly to avoid an explicit matrix multiplication
# for Psi = LL' when one element of L is modified.
function set_theta!(hyperparams::MNCRPHyperparams, val::Float64, i::Int64; backtransform=true)::MNCRPHyperparams
    d = dimension(hyperparams)
    D = 3 + d + div(d * (d + 1), 2)
    @assert 1 <= i <= D
    if i == 1
        if backtransform
            val = exp(val)
        end
        hyperparams.alpha = val
    elseif 1 + 1 <= i <= 1 + d
        k = i - 1
        hyperparams.mu[k] = val
    elseif i == 2 + d
        if backtransform
            val = exp(val)
        end
        hyperparams.lambda = val
    elseif 2 + d + 1 <= i <= 2 + d + div(d * (d + 1), 2)
        k = i - 2 - d
        hyperparams.flatL[k] = val
        i, j = ij(k)
        L = hyperparams.L
        hyperparams.psi[i, i] += val * val - L[i, j] * L[i, j]
        for k in i+1:d
            hyperparams.psi[k, i] += val * L[k, j] - L[i, j] * L[k, j]
            hyperparams.psi[i, k] = hyperparams.psi[k, i]
        end
        for l in j:i-1
            hyperparams.psi[i, l] += L[l, j] * val - L[l, j] * L[i, j]
            hyperparams.psi[l, i] = hyperparams.psi[i, l]
        end
        L[i, j] = val
    elseif i == D
        if backtransform
            val = d - 1 + exp(val)
        end
        hyperparams.nu = val
    end

    return hyperparams
end

function get_theta(hyperparams::MNCRPHyperparams, i::Int64; transform=true)
    d = dimension(hyperparams)
    D = 3 + d + div(d * (d + 1), 2)
    @assert 1 <= i <= D
    if i == 1
        if transform
            return log(hyperparams.alpha)
        else
            return hyperparams.alpha
        end
    elseif 1 + 1 <= i <= 1 + d
        k = i - 1
        return hyperparams.mu[k]
    elseif i == 2 + d
        if transform
            return log(hyperparams.lambda)
        else
            return hyperparams.lambda
        end
    elseif 2 + d + 1 <= i <= 2 + d + div(d * (d + 1), 2)
        k = i - 2 - d
        i, j = ij(k)
        return hyperparams.L[i, j]
    elseif i == D
        if transform
            return log(hyperparams.nu - d + 1)
        else
            return hyperparams.nu
        end
    end
end