using DataFrames: DataFrame, DataFrameRow
# import GBIF: GBIFRecord, GBIFRecords, occurrences
# using CSV

# function GBIFRecord(row::DataFrameRow)
    
#     row = Dict(names(row) .=> values(row))

#     levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
#     for level in levels
#         push!(row, level * "Key" => pop!(row, level))
#     end

#     push!(row, "key" => pop!(row, "gbifID"))

#     issues = pop!(row, "issue")
#     push!(row, "issues" => ismissing(issues) ? String[] : split(issues, ";"))

#     return GBIFRecord(row)
    
# end

# function GBIFRecords(dataframe::DataFrame)
#     query = nothing
#     records = GBIFRecord.(eachrow(dataframe))

#     return GBIFRecords(query, records)
# end

# occurrences(dataframe::DataFrame) = GBIFRecords(dataframe)
# occurrences(filename::AbstractString) = GBIFRecords(DataFrame(File(filename)))


# function generate_pseudoabsences(records::GBIFRecords, layers::Vector{<:SimpleSDMLayer}; n=nothing, method=WithinRadius, distance=1.0)
#     presences_only = mask(clip(first(layers), records), records, Bool)
#     if method == WithinRadius
#         psabs_lonlats = rand(method, presences_only, distance=distance)
#     else
#         psabs_lonlats = rand(method, presences_only)
#     end
#     psabs_lonlats = keys(psabs_lonlats)[collect(psabs_lonlats)]
#     if n !== nothing
#         n = min(n, length(psabs_lonlats))
#         psabs_longlats = sample(psabs_lonlats, n, replace=false)
#     end
#     psabs_predictors = Vector{Float64}[Float64[layer[lonlat] for layer in layers] for lonlat in psabs_lonlats]
#     return (predictors=psabs_predictors, longitudes=getindex.(psabs_lonlats, 1), latitudes=getindex.(psabs_lonlats, 2))
# end

# function generate_pseudoabsences(dataframe::DataFrame, layers::Vector{<:SimpleSDMLayer}; n=nothing, method=WithinRadius, distance=1.0)
#     records = GBIFRecords(dataframe)
#     return generate_pseudoabsences(records, layers, n=n, method=method, distance=distance)
# end


function performance_scores(scores_at_presences, scores_at_absences; threshold=nothing)
    if threshold !== nothing
        @assert 0.0 <= threshold <= 1.0

        # Scores of 1 always mean a presence,
        # so an absence with a score of 1 means a false positive
        scores_at_presences = scores_at_presences .>= threshold
        scores_at_absences = scores_at_absences .>= threshold
    end
    
    tp = sum(scores_at_presences)
    fn = sum(1 .- scores_at_presences)
    
    # absence with a score towards 0, therefore 1 - score -> 1
    # is a true negative
    tn = sum(1 .- scores_at_absences)
    fp = sum(scores_at_absences) # absence with score towards 1 is a false positive
    
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    MCC = (tp * tn - fp * fn)/sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    J = sensitivity + specificity - 1
    kappa = 2 * (tp * tn - fn * fp)/((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    return (; MCC, J, kappa, 
        sensitivity, specificity, 
        ppv, npv, 
        tp, fn, tn, fp,
        )
    
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
        # marginally faster than
        # 2 * sum(log.(diag(chol.U)))
        acc = 0.0
        for i in 1:size(A, 1)
            acc += log(chol.U[i, i])
        end
        return 2 * acc
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

