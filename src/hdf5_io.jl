function read_dataframe(tablegroup::HDF5.Group)
    columns = read_attribute(tablegroup, "column-order")

    havecat = false
    if haskey(tablegroup, "__categories")
        havecat = true
        catcols = tablegroup["__categories"]
    end

    if haskey(attributes(tablegroup), "_index")
        indexdsetname = read_attribute(tablegroup, "_index")
        rownames = read(tablegroup[indexdsetname])
    else
        rownames = nothing
    end

    df = DataFrame()

    for col in columns
        column = read(tablegroup, col)
        if havecat && haskey(catcols, col)
            cats = read(catcols, col)
            column = compress(CategoricalArray(map(x -> cats[x + 1], column)))
        end
        df[!, col] = column
    end

    return df, rownames
end

read_matrix(f::HDF5.Dataset) = read(f)

function read_matrix(f::HDF5.Group)
    enctype = read_attribute(f, "encoding-type")
    shape = read_attribute(f, "shape")
    if enctype == "csc_matrix"
        return SparseMatrixCSC(
            shape[1],
            shape[2],
            read(f, "indptr") .+ 1,
            read(f, "indices") .+ 1,
            read(f, "data"),
        )
    elseif enctype == "csr_matrix"
        return
        SparseMatrixCSC(
            shape[2],
            shape[1],
            read(f, "indptr") .+ 1,
            read(f, "indices") .+ 1,
            read(f, "data"),
        )'
    else
        throw("unknown encoding $enctype")
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::Dict{String, Any})
    g = create_group(parent, name)
    for (key, val) ∈ data
        write(g, key, val)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::CategoricalVector)
    write(parent, name, data.refs)
    write(parent, "__categories/$name", levels(data))
    attributes(parent[name])["categories"] = HDF5.Reference(parent["__categories"], name)
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::DataFrame)
    g = create_group(parent, name)
    attrs = attributes(g)
    attrs["encoding-type"] = "dataframe"
    attrs["encoding-version"] = "0.1.0"
    attrs["column-order"] = names(data)

    for (name, column) ∈ pairs(eachcol(data))
        write(g, string(name), column)
    end
end

Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Nothing, data::DataFrame) =
    write(parent, name, data)

function Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    rownames::Vector{String},
    data::DataFrame,
)
    write(parent, name, data)
    idxname = "_index"
    columns = names(data)
    while idxname ∈ columns
        idxname = "_" * idxname
    end
    g = parent[name]
    write(g, idxname, rownames)
    attributes(g)["_index"] = idxname
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::AbstractArray{<:Number}; extensible::Bool=false, compress::UInt8=UInt8(9))
    chunksize = HDF5.heuristic_chunk(data)
    if extensible
        dims = (size(data), Tuple(-1 for _ in 1:ndims(data)))
    else
        dims = size(data)
    end
    dtype = datatype(data)
    d = create_dataset(parent, name, dtype, dims, chunk=chunksize, compress=compress)
    write_dataset(d, dtype, data)
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::SparseMatrixCSC)
    g = create_group(parent, name)
    attrs = attributes(g)
    attrs["encoding-type"] = "csc_matrix"
    attrs["encoding-version"] = "0.1.0"
    attrs["shape"] = collect(size(data))
    write(g, "indptr", data.colptr .- 1, extensible=true)
    write(g, "indices", data.rowval .- 1, extensible=true)
    write(g, "data", data.nzval, extensible=true)
end

Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Adjoint{<:Number, SparseMatrixCSC},
) = write(parent, name, data.parent)

Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Nothing) = nothing
