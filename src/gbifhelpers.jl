using DataFrames: DataFrame, DataFrameRow
import GBIF: GBIFRecord, GBIFRecords, occurrences
using CSV

function GBIFRecord(row::DataFrameRow)
    
    row = Dict(names(row) .=> values(row))

    levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    for level in levels
        push!(row, level * "Key" => pop!(row, level))
    end

    push!(row, "key" => pop!(row, "gbifID"))

    issues = pop!(row, "issue")
    push!(row, "issues" => ismissing(issues) ? String[] : split(issues, ";"))

    return GBIFRecord(row)
    
end

function GBIFRecords(dataframe::DataFrame)
    query = nothing
    records = GBIFRecord.(eachrow(dataframe))

    return GBIFRecords(query, records)
end

occurrences(dataframe::DataFrame) = GBIFRecords(dataframe)
occurrences(filename::AbstractString) = GBIFRecords(DataFrame(File(filename, delim="\t")))