module Dataset
    using DataFrames
    using SimpleSDMLayers
    using StatsBase: std, mean
    using Random: shuffle!
    using MultivariateNormalCRP: Cluster
    import Base: show

    export MNCRPDataset
    export load_dataset, dataframe, original, longitudes, latitudes

    struct MNCRPDataset
        dataframe::DataFrame
        valid_mask::BitVector
        predictors::Vector{<:SimpleSDMLayer}
        predictornames::Vector{String}
        longlatcols::Vector{Union{Symbol, String, Int}}
        data_mean::Vector{Float64}
        data_scale::Vector{Float64}
        unique_map::Dict
        data::Vector{Vector{Float64}}
        train_data::Union{Nothing, Vector{Vector{Float64}}}
        test_data::Union{Nothing, Vector{Vector{Float64}}}
    end

    function show(io::IO, dataset::MNCRPDataset)
        println(io, "     #records: $(size(dataset.dataframe, 1))")
        println(io, "       #valid: $(sum(dataset.valid_mask))")
        println(io, "      #unique: $(length(dataset.unique_map))")
        println(io, "  #predictors: $(length(dataset.predictors))")
    end

    function load_dataset(dataframe::DataFrame, 
                          predictors::Vector{T};
                          predictornames=nothing,
                          longlatcols=[:decimalLongitude, :decimalLatitude],
                          split=true
                          ) where {T <: SimpleSDMLayer}
    
        @assert length(longlatcols) == 2
        @assert isnothing(predictornames) || length(predictors) == length(predictornames)
        if isnothing(predictornames)
            predictornames = ["predictor$i" for i in 1:length(predictors)]
        end

        longc, latc = longlatcols

        valid_mask = (.!isnothing.(dataframe[!, longc]) 
                  .&& .!ismissing.(dataframe[!, longc])
                  .&& .!isnothing.(dataframe[!, latc])
                  .&& .!ismissing.(dataframe[!, latc]))

        obspredvals = [valid_mask[k] ? [predictor[obs[longc], obs[latc]] for predictor in predictors] : [nothing for _ in 1:length(predictors)] for (k, obs) in enumerate(eachrow(dataframe))]

        valid_mask = valid_mask .&& map(x -> all(.!isnothing.(x)), obspredvals)

        predobsvals = Array{Union{Nothing, Float64}}(reduce(hcat, obspredvals))

        dataframe = deepcopy(dataframe)

        for (predname, vals) in zip(predictornames, eachrow(predobsvals))
            dataframe[!, predname] = collect(vals)
        end
        
        unique_predclusters = groupby(dataframe[valid_mask, :], predictornames)
        unique_predvals = [collect(first(group)[predictornames]) for group in unique_predclusters]

        # standardize #
        unique_predvals = reduce(hcat, unique_predvals)
        m = mean(unique_predvals, dims=2)
        s = mean(unique_predvals, dims=2)
        unique_standardized_predvals = 1.0 .* eachcol((unique_predvals .- m) ./ s)
        ################

        unique_map = Dict(val => group 
        for (val, group) in zip(unique_standardized_predvals, unique_predclusters))

        # unique_map now contains a map between
        # points in standardized environmental space to
        # groups of original observations.
        # Those points will be loaded onto the chain
        
        if split
            shuffle!(unique_standardized_predvals)
            train_dataset = unique_standardized_predvals[1:2:end]
            test_dataset = unique_standardized_predvals[2:2:end]
        end

        return MNCRPDataset(dataframe, 
                            valid_mask, 
                            predictors, 
                            predictornames,
                            longlatcols,
                            m[:, 1], s[:, 1],
                            unique_map,
                            unique_standardized_predvals,
                            train_dataset,
                            test_dataset)

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

    function longlats(element::Vector{Float64}, dataset::MNCRPDataset)
        group = dataset.unique_map[element]
        return collect.(eachrow(group[:, dataset.longlatcols]))
    end

    function longlats(cluster::Cluster, dataset::MNCRPDataset)
        return reduce(vcat, longlats.(cluster, Ref(dataset)))
    end

    function longitudes(element::Vector{Float64}, dataset::MNCRPDataset)
        return dataset.unique_map[element][:, dataset.longlatcols[1]]
    end

    function longitudes(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset; flatten=false)
        longs = [longitudes(element, dataset) for element in elements]
        if flatten
            longs = reduce(vcat, longs)
        end
    end

    function latitudes(element::Vector{Float64}, dataset::MNCRPDataset)
        return dataset.unique_map[element][:, dataset.longlatcols[2]]
    end

    function latitudes(elements::Union{Cluster, AbstractVector{Vector{Float64}}}, dataset::MNCRPDataset; flatten=false)
        lats = [latitudes(element, dataset) for element in elements]
        if flatten
            lats = reduce(vcat, lats)
        end

        return lats
    end
end
