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
    J = sensitivity + specificity - 1
    kappa = 2 * (tp * tn - fn * fp)/((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    return (
        J=J, 
        kappa=kappa, 
        sensitivity=sensitivity, 
        specificity=specificity, 
        ppv=ppv, 
        npv=npv, 
        tp=tp, 
        fn=fn, 
        tn=tn, 
        fp=fp,
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
