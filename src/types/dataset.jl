module Dataset
    using DataFrames
    using CSV
    using StatsBase: std, mean
    import Random

    export FCDataset
    export presence, absence

    mutable struct FCDataset
        df::AbstractDataFrame
        slices::Union{Nothing, Vector{UnitRange{Int64}}}
        _zero::AbstractDataFrame
        _scale::AbstractDataFrame
    end

    function FCDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
        
        if returncopy
            df = copy(df)
        end

        if shuffle
            df = Random.shuffle!(df)
        end

        if subsample isa Integer
            df = df[1:subsample, :]
        elseif subsample isa Float64
            df = df[1:round(Int, subsample*nrow(df)), :]
        end
        
        @assert all(splits .> 0)
        bnds = cumsum(vcat(0, splits / sum(splits)))
        slices = [round(Int, bnds[i]*nrow(df)+1):round(Int, bnds[i+1]*nrow(df)) for i in 1:length(splits)]

        _zero = mapcols(mean, df)
        _scale = mapcols(std, df)

        return FCDataset(df, slices, _zero, _scale)

    end

    function FCDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\t", shuffle=true, subsample=nothing) 
        return FCDataset(DataFrame(CSV.File(csvfile, delim=delim)), splits=splits, shuffle=shuffle, subsample=subsample, returncopy=false)
    end

    function Base.getindex(dataset::FCDataset, i::Int)
        return dataset.df[dataset.slices[i], :]
    end

    function Base.getproperty(dataset::FCDataset, name::Symbol)
        if name == :slices
            return getfield(dataset, :slices)
        elseif name === :df
            return getfield(dataset, :df)
        elseif name in propertynames(dataset.df)
            return getproperty(dataset.df, name)
        elseif name === :training
            return FCDataset(dataset.df[dataset.slices[1], :], nothing, dataset._zero, dataset._scale)
        elseif name === :validation
            length(dataset.slices) < 3 && throw(ArgumentError("Dataset has less than 3 splits"))
            return FCDataset(dataset.df[dataset.slices[2], :], nothing, dataset._zero, dataset._scale)
        elseif name === :test
            return FCDataset(dataset.df[dataset.slices[length(dataset.slices)], :], nothing, dataset._zero, dataset._scale)
        elseif name === :presence
            return presence(dataset)
        elseif name === :absence
            return absence(dataset)
        elseif name === :standardize
            return standardize(dataset)
        else
            return getfield(dataset, name)
        end
    end

    (dataset::FCDataset)(cols::Symbol...) = length(cols) == 1 ? dataset.df[:, cols[1]] : stack([dataset.df[:, col] for col in cols], dims=1)

    presence(col::Symbol) = dataset -> FCDataset(dataset.df[Bool.(dataset.df[:, col]), :], dataset.slices, dataset._zero, dataset._scale)
    presence(dataset::FCDataset) = col -> FCDataset(dataset.df[Bool.(dataset.df[:, col]), :], dataset.slices, dataset._zero, dataset._scale)
    presence(dataset::FCDataset, name::Symbol) = presence(dataset)(name)
    
    absence(col::Symbol) = dataset -> FCDataset(dataset.df[.!Bool.(dataset.df[:, col]), :], dataset.slices, dataset._zero, dataset._scale)
    absence(dataset::FCDataset) = col -> FCDataset(dataset.df[.!Bool.(dataset.df[:, col]), :], dataset.slices, dataset._zero, dataset._scale)
    absence(dataset::FCDataset, name::Symbol) = absence(dataset)(name)

    standarize(col::Symbol) = dataset -> (dataset.df[:, col] .- dataset._zero[1, col]) ./ dataset._scale[1, col]
    standardize(dataset::FCDataset) = (cols::Symbol...) -> length(cols) == 1 ? (dataset.df[:, cols[1]] .- dataset._zero[1, cols[1]]) ./ dataset._scale[1, cols[1]] : stack([(dataset.df[:, col] .- dataset._zero[1, col]) ./ dataset._scale[1, col] for col in cols], dims=1)
    standardize(dataset::FCDataset, name::Symbol) = standardize(dataset)(name)

    function Base.show(io::IO, dataset::FCDataset)
        print(io, "FCDataset(")
        Base.show(io, dataset.df)
        print(io, ")")
    end

end
