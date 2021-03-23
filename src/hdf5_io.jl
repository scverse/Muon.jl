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

    if enctype == "csc_matrix" || enctype == "csr_matrix"
        shape = read_attribute(f, "shape")
        iscsr = enctype[1:3] == "csr"

        indptr = read(f, "indptr")
        indices = read(f, "indices")
        data = read(f, "data")

        indptr .+= eltype(indptr)(1)
        indices .+= eltype(indptr)(1)

        # the row indices in every column need to be sorted
        @views for (colstart, colend) in zip(indptr[1:(end - 1)], indptr[2:end])
            sort!(indices[colstart:(colend - 1)], data[colstart:(colend - 1)])
        end

        if iscsr
            reverse!(shape)
        end
        mat = SparseMatrixCSC(shape..., indptr, indices, data)
        return iscsr ? mat' : mat
    else
        throw("unknown encoding $enctype")
    end
end

function read_dict_of_matrices(f::HDF5.Group)
    return Dict(key => read_matrix(f[key]) for key in keys(f))
end

function Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Dict{String, <:Any},
)
    g = create_group(parent, name)
    for (key, val) in data
        write(g, key, val)
    end
end

function Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::CategoricalVector,
)
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

    for (name, column) in pairs(eachcol(data))
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

function Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{<:Number};
    extensible::Bool=false,
    compress::UInt8=UInt8(9),
)
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

function Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::SparseMatrixCSC{<:Number, <:Integer};
    transposed = false
)
    g = create_group(parent, name)
    attrs = attributes(g)
    attrs["encoding-type"] = transposed ? "csr_matrix" : "csc_matrix"
    attrs["encoding-version"] = "0.1.0"

    shape = collect(size(data))
    transposed && reverse!(shape)
    attrs["shape"] = shape
    write(g, "indptr", data.colptr .- 1, extensible=true)
    write(g, "indices", data.rowval .- 1, extensible=true)
    write(g, "data", data.nzval, extensible=true)
end

Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Adjoint{T, SparseMatrixCSC{T, V}} where {T <: Number, V <: Integer},
) = write(parent, name, data.parent, transposed=true)

Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Nothing) = nothing
