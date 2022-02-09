function read_dataframe(tablegroup::HDF5.Group; separate_index=true, kwargs...)
    columns = read_attribute(tablegroup, "column-order")

    if separate_index && haskey(attributes(tablegroup), "_index")
        indexdsetname = read_attribute(tablegroup, "_index")
        rownames = read(tablegroup[indexdsetname])
    else
        rownames = nothing
    end

    df = DataFrame()

    for col in columns
        column = read_matrix(tablegroup[col])
        if sum(size(column) .> 1) > 1
            @warn "column $col has more than 1 dimension for data frame $(HDF5.name(tablegroup)), skipping"
        end
        df[!, col] = column
    end

    return df, rownames
end

function read_matrix(f::HDF5.Dataset; kwargs...)
    mat = read(f)
    if HDF5.h5t_get_class(datatype(f)) == HDF5.H5T_COMPOUND
        return StructArray(mat)
    end

    if ndims(f) == 0
        return mat
    end
    if ndims(f) > 1
        mat = PermutedDimsArray(mat, ndims(mat):-1:1) # transpose for h5py compatibility
    end
    if haskey(attributes(f), "categories")
        categories = f[read_attribute(f, "categories")]
        ordered =
            haskey(attributes(categories), "ordered") &&
            read_attribute(categories, "ordered") == true
        cats = read(categories)
        mat = mat .+ 0x1
        mat = compress(
            CategoricalArray{eltype(cats), ndims(mat)}(
                mat,
                CategoricalPool{eltype(cats), eltype(mat)}(cats, ordered),
            ),
        )
    end
    return mat
end

function read_matrix(f::HDF5.Group; kwargs...)
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
        error("unknown encoding $enctype")
    end
end

function read_dict_of_matrices(f::HDF5.Group; kwargs...)
    return Dict{String, AbstractArray{<:Number}}(
        key => read_matrix(f[key]; kwargs...) for key in keys(f)
    )
end

read_auto(f::HDF5.Dataset; kwargs...) = (read_matrix(f; kwargs...), nothing)
function read_auto(f::HDF5.Group; kwargs...)
    if haskey(attributes(f), "encoding-type")
        enctype = read_attribute(f, "encoding-type")
        if enctype == "dataframe"
            return read_dataframe(f; kwargs...)
        elseif endswith(enctype, "matrix")
            return read_matrix(f; kwargs), nothing
        else
            error("unknown encoding $enctype")
        end
    else
        return read_dict_of_mixed(f; kwargs...), nothing
    end
end

function read_dict_of_mixed(f::HDF5.Group; kwargs...)
    ret = Dict{
        String,
        Union{
            DataFrame,
            StructArray,
            <:AbstractArray{<:Number},
            <:AbstractArray{<:AbstractString},
            <:AbstractString,
            <:Number,
            Dict,
        },
    }()
    for k in keys(f)
        ret[k] = read_auto(f[k]; kwargs...)[1] # assume data frames are properly aligned, so we can discard rownames
    end
    return ret
end

function write_attr(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data; kwargs...)
    if haskey(parent, name)
        delete_object(parent, name)
    end
    write_impl(parent, name, data; kwargs...)
end

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Union{<:Number, <:AbstractString};
    kwargs...,
)
    parent[name] = data
end

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractDict{<:AbstractString, <:Any};
    kwargs...,
)
    if length(data) > 0
        g = create_group(parent, name)
        for (key, val) in data
            write_impl(g, key, val; kwargs...)
        end
    end
end

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::CategoricalArray;
    kwargs...,
)
    write_impl(parent, name, data.refs .- 0x1; kwargs...)
    write_impl(parent, "__categories/$name", levels(data); kwargs...)
    attributes(parent["__categories/$name"])["ordered"] = UInt8(isordered(data))
    attributes(parent[name])["categories"] = HDF5.Reference(parent["__categories"], name)
end

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractDataFrame;
    index::AbstractVector{<:AbstractString}=nothing,
    kwargs...,
)
    g = create_group(parent, name)
    attrs = attributes(g)
    attrs["encoding-type"] = "dataframe"
    attrs["encoding-version"] = "0.1.0"
    attrs["column-order"] = names(data)

    for (name, column) in pairs(eachcol(data))
        write_impl(g, string(name), column; kwargs...)
    end

    idxname = "_index"
    columns = names(data)
    if !isnothing(index)
        while idxname ∈ columns
            idxname = "_" * idxname
        end
    else
        if idxname ∈ columns
            index = data[!, idxname]
            select!(data, Not(idxname))
        else
            @warn "Data frame $(HDF5.name(parent))/$name does not have an _index column, a row number index will be written"
            index = collect(1:nrow(data))
        end
    end
    g = parent[name]
    write_impl(g, idxname, values(index); kwargs...)
    attributes(g)["_index"] = idxname
end

# see https://github.com/JuliaIO/HDF5.jl/issues/827
# and https://github.com/JuliaIO/HDF5.jl/issues/826
write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Union{<:AbstractArray{Bool}, BitArray{1}};
    extensible::Bool=false,
    compress::UInt8=UInt8(9),
    kwargs...,
) = write_impl(parent, name, Int8.(data); extensible=extensible, compress=compress, kwargs...)

write_impl(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, data::SubArray; kwargs...) =
    write_impl(parent, name, copy(data); kwargs...)

function write_impl(
    parentgrp::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray,
    ;
    extensible::Bool=false,
    compress::UInt8=0x9,
    kwargs...,
)
    if ndims(data) > 1
        data =
            data isa PermutedDimsArray && typeof(data).parameters[3] == Tuple(ndims(data):-1:1) ?
            parent(data) : permutedims(data, ndims(data):-1:1) # transpose for h5py compatibility
    end                                             # copy because HDF5 apparently can't handle lazy Adjoints

    if extensible
        dims = (size(data), Tuple(-1 for _ in 1:ndims(data)))
    else
        dims = size(data)
    end
    dtype = datatype(data)
    write_impl_array(parentgrp, name, data, dtype, dims, compress)
end

function write_impl_array(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{<:Number},
    dtype::HDF5.Datatype,
    dims::Union{Tuple{Vararg{<:Integer}}, Tuple{Tuple{Vararg{<:Integer, N}}, Tuple{Vararg{<:Integer, N}}}},
    compress::UInt8,
) where N
    chunksize = HDF5.heuristic_chunk(data)
    if length(chunksize) == 0
        chunksize = Tuple(100 for _ in 1:ndims(data))
    end
    d = create_dataset(parent, name, dtype, dims, chunk=chunksize, compress=compress)
    write_dataset(d, dtype, data)
end

# variable-length strings cannot be compressed in HDF5
function write_impl_array(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{<:AbstractString},
    dtype::HDF5.Datatype,
    dims::Union{Tuple{Vararg{<:Integer}}, Tuple{Tuple{Vararg{<:Integer, N}}, Tuple{Vararg{<:Integer, N}}}},
    compress::UInt8,
) where N
    d = create_dataset(parent, name, dtype, dims)
    write_dataset(d, dtype, data)
end

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::SparseMatrixCSC{<:Number, <:Integer};
    transposed=false,
    kwargs...,
)
    g = create_group(parent, name)
    attrs = attributes(g)
    attrs["encoding-type"] = transposed ? "csr_matrix" : "csc_matrix"
    attrs["encoding-version"] = "0.1.0"

    shape = collect(size(data))
    transposed && reverse!(shape)
    attrs["shape"] = shape
    write_impl(g, "indptr", data.colptr .- 1, extensible=true)
    write_impl(g, "indices", data.rowval .- 1, extensible=true)
    write_impl(g, "data", data.nzval, extensible=true)
end

write_impl(
    prt::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Adjoint{T, SparseMatrixCSC{T, V}} where {T <: Number, V <: Integer};
    kwargs...,
) = write_impl(prt, name, parent(data), transposed=true; kwargs...)

write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Union{HDF5.Dataset, SparseDataset, TransposedDataset};
    kwargs...,
) = copy_object(data, parent, name)

write_impl(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Nothing; kwargs...) =
    nothing

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::StructArray;
    extensible::Bool=false,
    compress::UInt8=0x9,
    kwargs...)
    ety = eltype(data)
    dtype = create_datatype(HDF5.H5T_COMPOUND, sizeof(ety))
    for (i, (fname, ftype)) in enumerate(zip(fieldnames(ety), fieldtypes(ety)))
        HDF5.h5t_insert(dtype, string(fname), fieldoffset(ety, i), _datatype(ftype))
    end
    chunksize = HDF5.heuristic_chunk(data)
    dims = size(data)
    if extensible
        dims = (size(data), Tuple(-1 for _ in 1:ndims(data)))
    end
    dset = create_dataset(parent, name, dtype, dataspace(dims), chunk=chunksize, compress=compress, kwargs...)
    write_dataset(dset, dtype, [val for row in data for val in row])
end

_datatype(::Type{T}) where T = datatype(T)

function _datatype(::Type{T}) where {T <: AbstractString}
    strdtype = HDF5.h5t_copy(HDF5.H5T_C_S1)
    HDF5.h5t_set_size(strdtype, HDF5.H5T_VARIABLE)
    HDF5.h5t_set_cset(strdtype, HDF5.H5T_CSET_UTF8)

    HDF5.Datatype(strdtype)
end
