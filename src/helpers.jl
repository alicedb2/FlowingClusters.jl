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


sqrtsigmoid(x::T; a=1/2) where {T} = T(1/2) + T(a) * x / sqrt(T(1) + T(a)^2 * x^2) / T(2)
sqrttanh(x::T) where {T} = T(2) * sqrtsigmoid(x, a=1) - T(1)
sqrttanhgrow(x) = x + sqrttanh(x)

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
function logdetpsd(A::AbstractMatrix{Float64})
    chol = cholesky(Symmetric(A), check=false)
    if issuccess(chol)
        return logdet(chol)
    else
        return -Inf
    end
end

function logdetflatLL(flatL::Vector{Float64})
    acc = 0.0
    i = 1
    delta = 2
    while i <= length(flatL)
        acc += log(flatL[i])
        i += delta
        delta += 1
    end
    return 2 * acc
end


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
    nu::Float64)::Tuple{Vector{Float64}, Matrix{Float64}}

    invWish = InverseWishart(nu, psi)
    sigma = rand(invWish)

    multNorm = MvNormal(mu, sigma/lambda)
    mu = rand(multNorm)

    return mu, sigma
end


# Used by plot() when nb_clusters given as a float 0 < p < 1.0
# Find the minimum size of clusters to include at least a proportion of the data
function minimum_size(clusters::Vector{Cluster}, proportion=0.05)

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