module Dataset
    using DataFrames: DataFrame, groupby
    using CSV: File
    using SimpleSDMLayers: SimpleSDMLayer
    import SimpleSDMLayers: longitudes, latitudes
    using StatsBase: std, mean
    using Random: shuffle!, shuffle
    using MultivariateNormalCRP: Cluster

    import Base: show, split

    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes, rescale, split

    struct MNCRPDataset
        dataframe::DataFrame
        predictors::Vector{<:SimpleSDMLayer}
        predictornames::Vector{String}
        longlatcols::Vector{Union{Symbol, String, Int}}
        data_mean::Vector{Float64}
        data_scale::Vector{Float64}
        unique_map::Dict
        data::Vector{Vector{Float64}}
    end

    function show(io::IO, dataset::MNCRPDataset)
        println(io, "          records: $(size(dataset.dataframe, 1))")
        println(io, "   unique records: $(length(dataset.unique_map))")
        println(io, "       predictors: $(length(dataset.predictors))")
        println(io, "  predictor names: $(dataset.predictornames)")
    end


    """
        load_dataset(dataframe::Union{DataFrame, AbstractString}, predictors::Vector{<:SimpleSDMLayer}; predictornames=nothing, longlatcols=[:decimalLongitude, :decimalLatitude], perturb=false, delim="\\t" )

    Load a dataset from a DataFrame `dataframe` (optionally a CSV filename) containing longitude and latitude columns. Datapoints are generated from SimpleSDMLayer layers `predictors`. Only points with unique values are kept in the final data.

    The optional keywords are:

      •  `predictornames`: list the predictor names that will be added as new column to the dataframe. Defaults to ["predictor1", "predictor2", ...].

      •  `longlatcols`: specify alternative longitude/latitude columns names.

      •  `perturb`: perturb each predictor point by a small value < 10⁻⁵

      •  `delim`: specify alternative column delimiter if loading from a CSV file
    """
    function load_dataset(dataframe::Union{DataFrame, AbstractString}, 
                          predictors::Vector{<:SimpleSDMLayer};
                          predictornames=nothing,
                          longlatcols=[:decimalLongitude, :decimalLatitude],
                          perturb=false,
                          delim="\t"
                          )
    
        @assert length(longlatcols) == 2
        @assert isnothing(predictornames) || length(predictors) == length(predictornames)
        
        if typeof(longlatcols) == Tuple
            longlatcols = collect(longlatcols)
        end

        if typeof(dataframe) <: AbstractString
            dataframe = DataFrame(File(dataframe, delim=delim))
        else
            dataframe = copy(dataframe)
        end

        if isnothing(predictornames)
            predictornames = ["predictor$i" for i in 1:length(predictors)]
        end

        longc, latc = longlatcols

        valid_mask = (.!isnothing.(dataframe[!, longc]) 
                  .&& .!ismissing.(dataframe[!, longc])
                  .&& .!isnothing.(dataframe[!, latc])
                  .&& .!ismissing.(dataframe[!, latc]))

        obspredvals = [valid_mask[k] ? [predictor[obs[longc], obs[latc]] for predictor in predictors] : [nothing for _ in 1:length(predictors)] for (k, obs) in enumerate(eachrow(dataframe))]

        valid_mask = BitVector(valid_mask .&& map(x -> !any(isnothing.(x)), obspredvals))

        predobsvals = Array{Union{Nothing, Float64}}(reduce(hcat, obspredvals))

        dataframe = dataframe[valid_mask, :]
        predobsvals = predobsvals[:, valid_mask]

        for (predname, vals) in zip(predictornames, eachrow(predobsvals))
            vals = collect(vals)
            if perturb
                vals .+= 1e-5 * (rand(length(vals)) .- 0.5)
            end
            dataframe[!, predname] = vals
        end
        
        unique_predclusters = groupby(dataframe, predictornames)
        unique_predvals = [collect(first(group)[predictornames]) for group in unique_predclusters]

        # standardize #
        unique_predvals = reduce(hcat, unique_predvals)
        m = mean(unique_predvals, dims=2)
        s = std(unique_predvals, dims=2)
        unique_standardized_predvals = 1.0 .* eachcol((unique_predvals .- m) ./ s)
        ################

        unique_map = Dict(val => group 
        for (val, group) in zip(unique_standardized_predvals, unique_predclusters))

        # unique_map now contains a map between
        # points in standardized environmental space to
        # groups of original observations.
        # Those points will be loaded onto the chain
        
        return MNCRPDataset(dataframe,
                            predictors[:], 
                            predictornames,
                            longlatcols,
                            m[:, 1], s[:, 1],
                            unique_map,
                            unique_standardized_predvals)
    end


    """
    `split(dataset::MNCRPDataset, n::Int=2)`

    Randomly split a dataset n-ways over its unique predictor values.

    """
    function split(dataset::MNCRPDataset, n::Int=2)
        n == 1 && return dataset

        datasets = MNCRPDataset[]
        data = shuffle(dataset.data)
        uv_splits = [data[i:n:end] for i in 1:n]

        for i in 1:n
            reduced_dataframe = reduce(vcat, (dataset.unique_map[x] for x in uv_splits[i]))
            reduced_unique_map = Dict(x => dataset.unique_map[x] for x in uv_splits[i])
            reduced_dataset = MNCRPDataset(reduced_dataframe, 
                                           dataset.predictors[:], 
                                           dataset.predictornames[:], 
                                           dataset.longlatcols[:],
                                           dataset.data_mean[:],
                                           dataset.data_scale[:],
                                           reduced_unique_map,
                                           uv_splits[i])
            push!(datasets, reduced_dataset)
        end

        return datasets
    end


    function dataframe(clusters::AbstractVector{Cluster}, dataset::MNCRPDataset; repeats=true)
        return DataFrame[dataframe(cluster, dataset, repeats=repeats) for cluster in clusters]
    end

    function dataframe(element::Vector{Float64}, dataset::MNCRPDataset; repeats=true)
        return first(dataframe(Vector{Float64}[element], dataset, repeats=repeats))
    end

    function dataframe(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset; repeats=true)
        if !repeats
            return reduce(vcat, DataFrame.(first(dataset.unique_map[x]) for x in elements))
        else
            return reduce(vcat, (dataset.unique_map[x] for x in elements))
        end
    end

    function original(element::Vector{Float64}, dataset::MNCRPDataset)
        return collect(first(dataset.unique_map[element])[dataset.predictornames])
    end
    
    function original(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset)
        return Vector{Float64}[original(element, dataset) for element in elements]
    end

    function original(clusters::AbstractVector{Cluster}, dataset::MNCRPDataset)
        return [original(cluster, dataset) for cluster in clusters]
    end

    function longitudes(element::Vector{Float64}, dataset::MNCRPDataset; unique=false)
        if unique
            return first(dataset.unique_map[element])[dataset.longlatcols[1]]
        else
            return dataset.unique_map[element][:, dataset.longlatcols[1]]
        end
    end

    function longitudes(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset; unique=false, flatten=false)
        longs = [longitudes(element, dataset, unique=unique) for element in elements]
        if flatten
            longs = reduce(vcat, longs)
        end
        return longs
    end

    function longitudes(dataset::MNCRPDataset; unique=false, flatten=false)
        return longitudes(dataset.data, dataset, unique=unique, flatten=flatten)
    end

    function latitudes(element::Vector{Float64}, dataset::MNCRPDataset; unique=false)
        if unique
            return first(dataset.unique_map[element])[dataset.longlatcols[2]]
        else
            return dataset.unique_map[element][:, dataset.longlatcols[2]]
        end
    end

    function latitudes(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset; unique=false, flatten=false)
        lats = [latitudes(element, dataset, unique=unique) for element in elements]
        if flatten
            lats = reduce(vcat, lats)
        end

        return lats
    end

    function latitudes(dataset::MNCRPDataset; unique=false, flatten=false)
        return latitudes(dataset.data, dataset, unique=unique, flatten=flatten)
    end

    function rescale(element::Vector{Float64}, dataset::MNCRPDataset)
        return (element .- dataset.data_mean) ./ dataset.data_scale
    end

    function rescale(elements::Vector{Vector{Float64}}, dataset::MNCRPDataset)
        return rescale.(elements, Ref(dataset))
    end

end
