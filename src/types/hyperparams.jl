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
    return MNCRPHyperparams(1.0, zeros(d), 1.0, LowerTriangular(diagm(fill(sqrt(0.1), d))), 1.0 * d)
end

function clear_diagnostics!(hyperparams::MNCRPHyperparams)
    d = length(hyperparams.mu)

    hyperparams.diagnostics = Diagnostics(d)

    return hyperparams    
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
    hyperparams.flatL = flatten(hyperparams._L)
    hyperparams.psi = value
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
    end

    return vcat(alpha, hyperparams.mu, lambda, hyperparams.flatL, nu)

end