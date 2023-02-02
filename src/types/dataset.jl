module Dataset
    using DataFrames
    using SimpleSDMLayers
    using StatsBase: std, mean
    using Random: shuffle!
    using MultivariateNormalCRP: Cluster
    import Base: show

    export load_dataset, cluster_dataframe, original, longlats

    struct MNCRPDataset
        dataframe::DataFrame
        valid_mask::BitVector
        predictors::Vector{<:SimpleSDMLayer}
        predictornames::Vector{String}
        data_mean::Vector{Float64}
        data_scale::Vector{Float64}
        unique_map::Dict
        dataset::Vector{Vector{Float64}}
        train_dataset::Union{Nothing, Vector{Vector{Float64}}}
        test_dataset::Union{Nothing, Vector{Vector{Float64}}}
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
    
        @assert lenght(longlatcols) == 2
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
                            m[:, 1], s[:, 1],
                            unique_map,
                            unique_standardized_predvals,
                            train_dataset,
                            test_dataset)

    end

    function cluster_dataframe(cluster::Cluster, dataset::MNCRPDataset)
        return reduce(vcat, (dataset.unique_map[x] for x in cluster))
    end

    function original(element::Vector{Float64}, dataset::MNCRPDataset)
        return collect(first(dataset.unique_map[element])[dataset.predictornames])
    end
    
    function original(elements::Vector{Vector{Float64}}, dataset::MNCRPDataset)
        return original.(elements, Ref(dataset))
    end

    function longlats(element::Vector{Float64}, dataset::MNCRPDataset; longlatcols=[:decimalLongitude, :decimalLatitude])
        @assert length(longlatcols) == 2
        group = dataset.unique_map[element]
        return collect.(eachrow(group[:, longlatcols]))
    end

    function longlats(cluster::Cluster, dataset::MNCRPDataset; longlatcols=[:decimalLongitude, :decimalLatitude])
        return reduce(vcat, longlats.(cluster, Ref(dataset); longlatcols=longlatcols))
    end

end
