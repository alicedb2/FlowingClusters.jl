function performance_statistics(scores_at_presences, scores_at_absences; threshold=nothing)

    if threshold !== nothing
        scores_at_presences = scores_at_presences .>= threshold
        scores_at_absences = scores_at_absences .>= threshold
    end

    # presence with a score towards 1 is a true positive
    tp = sum(scores_at_presences)
    # absence with score towards 1 is a false positive
    fp = sum(scores_at_absences)

    # absence with a score towards 0 is a true negative
    tn = sum(1 .- scores_at_absences)
    # presence with score towards 0 is a false negative
    fn = sum(1 .- scores_at_presences)


    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    J = sensitivity + specificity - 1
    MCC = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    kappa = 2 * (tp * tn - fn * fp)/((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    return (;
            MCC, J, kappa,
            sensitivity, specificity,
            ppv, npv,
            tp, tn, fp, fn
            )

end

function best_score_threshold(scores_at_presences, scores_at_absences; statistic=:MCC, nbsteps=1000)

    thresholds = LinRange(0.0, 1.0, nbsteps)
    best_score = -Inf
    best_thresh = 0.0

    statistic in [:J, :MCC, :kappa] || throw(ArgumentError("perfscore must be :J (Youden's J), :MCC (Matthew's correlation coefficient), or :kappa (Cohen's kappa)"))

    for thresh in thresholds
        perfstats = performance_statistics(scores_at_presences, scores_at_absences, threshold=thresh)
        if perfstats[statistic] >= best_score
            best_score = perfstats[statistic]
            best_thresh = thresh
        end
    end

    return best_thresh

end


function sqrtsigmoid(x::T; a=1/2) where T
    E = eltype(T)
    return E(1/2) .+ E(a) .* x ./ sqrt.(E(1) .+ E(a).^2 .* x.^2) ./ E(2)
end
function sqrttanh(x::T; a=1) where T
    E = eltype(T)
    return E(2) .* sqrtsigmoid.(x, a=E(a)) .- E(1)
end
sqrttanhgrow(x) = x .+ sqrttanh.(x)

function chunkslices(sizes)
    boundaries = cumsum(vcat(1, sizes))
    return [boundaries[i]:boundaries[i+1]-1 for i in 1:length(sizes)]
end

function chunk(data::T, sizes) where {T}
    sum(sizes) <= size(data, ndims(data)) || throw(ArgumentError("Sum of sizes ($(sum(sizes))) must be less than or equal to last dimension ($(size(data, ndims(data))))"))
    return [T(selectdim(data, ndims(data), slice)) for slice in chunkslices(sizes)]
end

# Quick and dirty but faster logdet
# for (assumed) positive-definite matrix
function logdetpsd(A::AbstractMatrix{T})::T where T
    chol = cholesky(Symmetric(A), check=false)
    if issuccess(chol)
        return logdet(chol)
    else
        return -Inf
    end
end

# function logdetflatLL(flatL::Vector{Float64})
#     acc = 0.0
#     i = 1
#     delta = 2
#     while i <= length(flatL)
#         acc += log(flatL[i])
#         i += delta
#         delta += 1
#     end
#     return 2 * acc
# end


function freedmandiaconis(x::AbstractArray)
    n = length(x)
    return 2 * iqr(x) / length(n)^(1/3)
end

function doane(x::AbstractArray)
    n = length(x)
    skw = skewness(x)
    sg1 = sqrt(6 * (n - 2) / (n + 1) / (n + 3))
    return 1 + log2(n) + log2(1 + abs(skw) / sg1)
end


function drawNIW(
    mu::AbstractVector{Float64},
    lambda::Float64,
    psi::AbstractMatrix{Float64},
    nu::Float64, rng::AbstractRNG=default_rng())::Tuple{Vector{Float64}, Matrix{Float64}}

    invWish = InverseWishart(nu, psi)
    sigma = rand(rng, invWish)

    multNorm = MvNormal(mu, sigma/lambda)
    mu = rand(rng, multNorm)

    return mu, sigma
end


# Used by plot() when nb_clusters given as a float 0 < p < 1.0
# Find the minimum size of clusters to include at least a proportion of the data
function minimum_size(clusters::AbstractVector{<:AbstractCluster}, proportion=0.05)

    N = sum(length.(clusters))
    include_up_to = N * (1 - proportion)

    sorted_sizes = sort(length.(clusters), rev=true)
    cumul_sizes = cumsum(sorted_sizes)

    last_idx = findlast(x -> x <= include_up_to, cumul_sizes)

    # min_cumul_idx = findfirst(x -> x > propN, cumul_sizes)
    # min_size_idx = findfirst(x -> x > sorted_sizes[min_cumul_idx], sorted_sizes)

    minsize = sorted_sizes[last_idx]

    return max(minsize, 0)

end

dims_to_proj(dims::AbstractVector{Int}, d::Int) = I(d)[dims, :]

project_vec(vec::AbstractVector, proj::AbstractMatrix) = proj * vec
project_vec(vec::AbstractVector, dims::AbstractVector{Int}) = dims_to_proj(dims, size(vec, 1)) * vec

project_mat(mat::AbstractMatrix, proj::AbstractMatrix) = proj * mat * proj'

function project_mat(mat::AbstractMatrix, dims::AbstractVector{Int})
    d = size(mat, 1)
    proj = dims_to_proj(dims, d)
    return proj * mat * proj'
end


function generate_data(;D=6, T=Float64, K=10, N=100, muscale=0.5, sigmascale=0.3, seed=default_rng(), test=false)
    if isnothing(seed)
        rng = default_rng()
    elseif seed isa AbstractRNG
        rng = seed
    elseif seed isa Int
        rng = MersenneTwister(seed)
    else
        throw(ArgumentError("seed must be an AbstractRNG or an integer"))
    end


    base_matclusters = [Matrix{T}(rand(rng, MvNormal(T(muscale)*randn(rng, T, D), Diagonal(T(sigmascale)^2*rand(rng, D))), N)) for _ in 1:K]
    orig_matclusters = [Matrix{T}(rand(rng, MvNormal(T(muscale)*randn(rng, T, D), Diagonal(T(sigmascale)^2*rand(rng, D))), N)) for _ in 1:K]

    b2oset = Dict{SVector{D, T}, SVector{D, T}}(vb => vo for (clb, clo) in zip(base_matclusters, orig_matclusters) for (vb, vo) in zip(eachcol(clb), eachcol(clo)))
    setclusters = [SetCluster(cl, b2oset, check=test) for cl in base_matclusters]

    isvalid, offenders = isvalidpartition(setclusters, fullresult=true)
    if !isvalid
        # We are removing the offenders from the base2original dictionary
        filter!(pair -> !(pair[1] in offenders), b2oset) # b2o is common to all
        for cl in setclusters
            filter!(el -> !(el in offenders), cl.elements)
            _recalculate_sums!(cl)
        end
        filter!(!isempty, setclusters)
        # @warn "There were collisions in the SetClusters, discarding offenders.\nYou won't get exactly $(K*N) elements in $K clusters in total, you'll get $(length(b2oset)) elements in $(length(setclusters)) instead.\nBet you were playing in low precision, low dimension, and with large N.\n\033[1;33mThis might fail a few tests in the test suite, that's fine.\033[0m"
    end

    b = reduce((b2o, x) -> hcat(b2o, Matrix(x)), setclusters, init=zeros(T, D, 0))
    o = reduce((b2o, x) -> hcat(b2o, Matrix(x, orig=true)), setclusters, init=zeros(T, D, 0))
    b2obit = cat(b, o, dims=3)
    idxclusters = chunkslices(length.(setclusters))
    bitclusters = [BitCluster(idx, b2obit, check=test) for idx in idxclusters]
    _recalculate_sums!(bitclusters)
    if test
        return true
    else
        return bitclusters, setclusters
    end
end

function Base.Array(b2o::Dict{SVector{D, T}, SVector{D, T}})::Array{T, 3} where {T, D}
    basedata = reduce(hcat, collect.(keys(b2o)))
    origdata = reduce(hcat, collect.(values(b2o)))
    return cat(basedata, origdata, dims=3)
end

function idinit(mult::T, direction=:in) where T <: Real
    function init(rng::AbstractRNG, dims...)
        w = zeros(T, dims...)
        outdim, indim = dims[1], dims[2]
        if direction === :in
            slices = chunkslices(fill(div(outdim, indim), indim))
            for (i, slice) in enumerate(slices)
                w[slice, i] .= T(mult)
            end
        elseif direction === :out
            slices = chunkslices(fill(div(indim, outdim), outdim))
            for (i, slice) in enumerate(slices)
                w[i, slice] .= T(mult)
            end
        end
        return w
    end
end
