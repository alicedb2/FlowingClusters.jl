mutable struct MNCRPHyperparams
    alpha::Float64
    mu::Vector{Float64}
    lambda::Float64
    flatL::Vector{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    psi::Matrix{Float64}
    nu::Float64

    diagnostics::Diagnostics
end

function Base.show(io::IO, h::MNCRPHyperparams)
    d = length(h.mu)
    D = 3 + d + div(d * (d + 1), 2)
    print(io, "MNCRPHyperparams(d=$d, D=$D from sum(1, $d, 1, $(div(d * (d + 1), 2)), 1))")
end

function MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu)
    d = length(mu)

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu, Diagnostics(d))
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, flatL::Vector{Float64}, nu::Float64)
    d = length(mu)
    flatL_d = Int64(d * (d + 1) / 2)
    if size(flatL, 1) != flatL_d
        error("Dimension mismatch, flatL should have length $flatL_d")
    end

    L = foldflat(flatL)
    psi = L * L'

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu)
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, L::LowerTriangular{Float64}, nu::Float64)
    d = length(mu)
    if !(d == size(L, 1))
        error("Dimension mismatch, L should have dimension $d x $d")
    end

    psi = L * L'
    flatL = flatten(L)

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu)
end

function MNCRPHyperparams(alpha::Float64, mu::Vector{Float64}, lambda::Float64, psi::Matrix{Float64}, nu::Float64)
    d = length(mu)
    if !(d == size(psi, 1) == size(psi, 2))
        error("Dimension mismatch, L should have dimension $d x $d")
    end

    L = cholesky(psi).L
    flatL = flatten(L)

    return MNCRPHyperparams(alpha, mu, lambda, flatL, L, psi, nu)
end

function MNCRPHyperparams(d::Int64)
    return MNCRPHyperparams(
        7.77, 
        zeros(d), 
        0.01, 
        LowerTriangular(diagm(fill(0.1, d))), 
        d + 1.0
        )
end

function clear_diagnostics!(hyperparams::MNCRPHyperparams)
    d = length(hyperparams.mu)

    hyperparams.diagnostics = Diagnostics(d)

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
    @assert length(hyperparams.mu) == size(hyperparams.psi, 1) == size(hyperparams.psi, 2) "Dimensions of mu (d) and psi (d x d) do not match"
    return length(hyperparams.mu)
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
    d = length(hyperparams.mu)
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
    d = length(hyperparams.mu)
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

function add_one_dimension!(hyperparams::MNCRPHyperparams)

    d = length(hyperparams.mu)
    hyperparams.mu = vcat(hyperparams.mu, 0.0)
    hyperparams.nu += 1
    psi = hcat(hyperparams.psi, zeros(d))
    psi = vcat(psi, zeros(1, d + 1))
    psi[d+1, d+1] = 0.01
    L = cholesky(psi).L
    flatL = flatten(L)
    hyperparams.flatL, hyperparams.L, hyperparams.psi = flatL, L, psi
    
    add_one_dimension!(hyperparams.diagnostics)

    return hyperparams
end

# We gotta go fast!
# Propagate what theta[i] locally affects
# This is mostly to avoid an explicit matrix multiplication
# for Psi = LL' when one element of L is modified.
function set_theta!(hyperparams::MNCRPHyperparams, val::Float64, i::Int64; backtransform=true)::MNCRPHyperparams
    d = length(hyperparams.mu)
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
    d = length(hyperparams.mu)
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