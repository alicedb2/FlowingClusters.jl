module Dataset
    using DataFrames: AbstractDataFrame, DataFrame, groupby
    using CSV: File
    using SpeciesDistributionToolkit: SimpleSDMLayer
    import SpeciesDistributionToolkit: longitudes, latitudes
    using StatsBase: std, mean
    import StatsBase: standardize
    using Random: shuffle!, shuffle
    using MultivariateNormalCRP: Cluster

    import Base: show, split

    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes, standardize_with, standardize_with!, split, standardize!

    mutable struct MNCRPDataset
        dataframe::DataFrame
        layers::Vector{<:SimpleSDMLayer}
        layernames::Vector{String}
        longlatcols::Vector{Union{Symbol, String, Int}}
        data_zero::Vector{Float64}
        data_scale::Vector{Float64}
        unique_map::Union{Dict, Nothing}
        data::Vector{Vector{Float64}}
        data_raw::Vector{Vector{Float64}}
    end

    function Base.show(io::IO, dataset::MNCRPDataset)
        println(io, "          records: $(size(dataset.dataframe, 1))")
        println(io, "   unique records: $(length(dataset.unique_map))")
        println(io, "       predictors: $(length(dataset.layers))")
        println(io, "  predictor names: $(dataset.layernames)")
    end


    """
        load_dataset(dataframe::Union{DataFrame, AbstractString}, predictors::Vector{<:SimpleSDMLayer}; layernames=nothing, longlatcols=[:decimalLongitude, :decimalLatitude], perturb=false, delim="\\t", eps=1e-5)

    Load a dataset from a DataFrame `dataframe` (optionally a CSV filename) 
    
    The dataframe only needs a longitude and a latitude column.
    Datapoints are generated from values returned by `SimpleSDMLayer` layers contained in `predictors`.
    Only points with unique predictor values are kept in the final `data` field. Use `pertub=true` to keep repeated/non-unique values.
    Data points with missing longitude and/or latitude or with predictor values containing one or more `nothing` are dropped.

    The optional keywords are:

    * `layernames`: list the predictor names that will be added as new columns to the dataframe saved into the dataset. Defaults to ["predictor1", "predictor2", ...].

    * `longlatcols`: specify longitude/latitude columns names. Can be a mix of symbols, strings, and indices. Defaults to [:decimalLongitude, :decimalLatitude].

    * `delim`: specify alternative column delimiter if loading from a CSV file. Defaults to "`\\t`".

    * `perturb`: perturb each predictor value by a small value < `eps`. This is an approximate way to allow repeats (e.g. when loading in MNCRP chain).

    """
    function MNCRPDataset(dataframe::Union{AbstractDataFrame, AbstractString}, 
                          layers::AbstractVector{<:SimpleSDMLayer};
                          layernames=nothing,
                          longlatcols=[:decimalLongitude, :decimalLatitude],
                          delim="\t",
                          standardize=false,
                          perturb=false,
                          eps=1e-6
                          )
    
        @assert length(longlatcols) == 2
        @assert isnothing(layernames) || length(layers) == length(layernames)
        
        if typeof(longlatcols) == Tuple
            longlatcols = collect(longlatcols)
        end

        if typeof(dataframe) <: AbstractString
            dataframe = DataFrame(File(dataframe, delim=delim))
        else
            dataframe = copy(dataframe)
        end

        if isnothing(layernames)
            layernames = ["predictor$i" for i in 1:length(layers)]
        end

        longc, latc = longlatcols

        valid_mask = (.!isnothing.(dataframe[!, longc]) 
                  .&& .!ismissing.(dataframe[!, longc])
                  .&& .!isnothing.(dataframe[!, latc])
                  .&& .!ismissing.(dataframe[!, latc]))

        obspredvals = [valid_mask[k] ? [predictor[obs[longc], obs[latc]] for predictor in layers] : [nothing for _ in 1:length(layers)] for (k, obs) in enumerate(eachrow(dataframe))]

        valid_mask = BitVector(valid_mask .&& map(x -> !any(isnothing.(x)), obspredvals))

        predobsvals = Array{Union{Nothing, Float64}}(reduce(hcat, obspredvals))

        dataframe = dataframe[valid_mask, :]
        predobsvals = predobsvals[:, valid_mask]

        for (predname, vals) in zip(layernames, eachrow(predobsvals))
            vals = collect(vals)
            if perturb
                vals .+= epsilon * (rand(length(vals)) .- 0.5)
            end
            dataframe[!, predname] = vals
        end
        
        unique_predclusters = groupby(dataframe, layernames)
        unique_predvals = [collect(first(group)[layernames]) for group in unique_predclusters]
        # standardize #
        if standardize
            data_zero = mean(unique_predvals)
            data_scale = std(unique_predvals)
            unique_standardized_predvals = [(x .- data_zero) ./ data_scale for x in unique_predvals]
        else
            data_zero = zeros(length(first(unique_predvals)))
            data_scale = ones(length(data_zero))
            unique_standardized_predvals = unique_predvals[:]
        end

        ################

        unique_map = Dict(val => group 
                          for (val, group) in zip(unique_standardized_predvals, unique_predclusters))

        # unique_map now contains a map between
        # points in standardized environmental space to
        # groups of original observations.
        # Those points will be loaded onto the chain
        
        return MNCRPDataset(dataframe,
                            layers[:], 
                            layernames[:],
                            longlatcols[:],
                            data_zero[:, 1], data_scale[:, 1],
                            unique_map,
                            unique_standardized_predvals,
                            unique_predvals
                            )
    end


    function MNCRPDataset(lonlats::NamedTuple, layers::Vector{<:SimpleSDMLayer}; standardize_with::Union{MNCRPDataset, Nothing}=nothing)
        @assert :longitudes in keys(lonlats) && :latitudes in keys(lonlats)
        
        dataset = MNCRPDataset(
            DataFrame(Dict(:longitude => lonlats.longitudes[:], :latitude => lonlats.latitudes[:])),
            layers[:],
            longlatcols=[:longitude, :latitude]
            )

        if standardize_with !== nothing
            standardize_with!(dataset, standardize_with)
        end
        
        return dataset

    end

    """
    `split(dataset::MNCRPDataset, n::Int=2; rescaletofirst=true)`

    Randomly split a dataset `n`-ways.

    Note: It splits the unique values of the predictors and not those (potentially non-unique) of the full dataframe.
    
    * `rescaletofirst=true (default)` insures that the rescaling is redone according to the first split and not simply a copy of the original. This should be `true` when splitting into training/validation/test datasets.

    """
    function Base.split(dataset::MNCRPDataset, n::Int=3; standardize_with_first=true)
        n <= 1 && return dataset

        datasets = MNCRPDataset[]
        data = shuffle(dataset.data)
        uv_splits = [data[i:n:end] for i in 1:n]

        for i in 1:n
            reduced_dataframe = reduce(vcat, (dataset.unique_map[x] for x in uv_splits[i]))
            reduced_dataset = MNCRPDataset(reduced_dataframe,
                                           dataset.layers[:],
                                           layernames=dataset.layernames[:],
                                           longlatcols=dataset.longlatcols[:]
                                          )
                                          
            if i > 1 && standardize_with_first
                standardize_with!(reduced_dataset, first(datasets))
            end
            
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
        return collect(first(dataset.unique_map[element])[dataset.layernames])
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

    function latitudes(elements::Union{AbstractVector{Vector{Float64}}, Cluster}, dataset::MNCRPDataset; unique=false, flatten=false)
        lats = [latitudes(element, dataset, unique=unique) for element in elements]
        if flatten
            lats = reduce(vcat, lats)
        end

        return lats
    end

    function latitudes(dataset::MNCRPDataset; unique=false, flatten=false)
        return latitudes(dataset.data, dataset, unique=unique, flatten=flatten)
    end

    
    function standardize_with(dataset::MNCRPDataset, standardize_with::MNCRPDataset)
        restandardized_dataset = deepcopy(dataset)
        return standardize_with!(restandardized_dataset, standardize_with)
    end

    """
    `function standardize(element::Vector{Float64}, dataset::MNCRPDataset)`

    Return standard score of `element` against mean and standard deviation already determined for `dataset`.
    """
    function standardize_with(element::Vector{Float64}, dataset::MNCRPDataset)
        return (element .- dataset.data_zero) ./ dataset.data_scale
    end
    
    
    """
    `    function standardize(elements::AbstractVector{Vector{Float64}}, dataset::MNCRPDataset)`

    Return standard score of `elements` against mean and standard deviation already determined for `dataset`.
    """
    function standardize_with(elements::Vector{Vector{Float64}}, dataset::MNCRPDataset)
        return standardize_with.(elements, Ref(dataset))
    end

    function standardize_with!(dataset::MNCRPDataset, standardize_with::MNCRPDataset)
        
        new_data_z = standardize_with.data_zero[:]
        new_data_s = standardize_with.data_scale[:]
        new_data = [((dataset.data_scale .* old_X + dataset.data_zero) .- new_data_z) ./ new_data_s for old_X in dataset.data]
        dataset.unique_map = Dict(new_X => dataset.unique_map[old_X] for (old_X, new_X) in zip(dataset.data, new_data)) 
        dataset.data_zero = new_data_z
        dataset.data_scale = new_data_s
        dataset.data = new_data
        
        return dataset

    end

    function standardize!(dataset::MNCRPDataset)
        data_zero = mean(dataset.data)
        data_scale = std(dataset.data)
        standardized_data = [(x - data_zero) ./ data_scale for x in dataset.data]
        standardized_unique_map = Dict(new_X => dataset.unique_map[old_X] for (old_X, new_X) in zip(dataset.data, standardized_data)) 
        dataset.data = standardized_data
        dataset.unique_map = standardized_unique_map
        dataset.data_zero = data_zero
        dataset.data_scale = data_scale
        return dataset
    end
end
