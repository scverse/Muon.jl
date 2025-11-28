create_group(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString) = HDF5.create_group(parent, name)

delete_object(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString) = HDF5.delete_object(parent, name)

datatype(t::Type) = HDF5.datatype(t)
function datatype(::Type{T}) where {T <: AbstractString}
    strdtype = HDF5.API.h5t_copy(HDF5.API.H5T_C_S1)
    HDF5.API.h5t_set_size(strdtype, HDF5.API.H5T_VARIABLE)
    HDF5.API.h5t_set_cset(strdtype, HDF5.API.H5T_CSET_UTF8)

    HDF5.Datatype(strdtype)
end

function datatype(::Type{Bool})
    dtype = create_datatype(HDF5.API.H5T_ENUM, sizeof(Bool))
    HDF5.API.h5t_enum_insert(dtype, "FALSE", Ref(false))
    HDF5.API.h5t_enum_insert(dtype, "TRUE", Ref(true))
    return dtype
end

read_attribute(obj::Union{HDF5.Dataset, HDF5.Group, HDF5.File}, attrname::AbstractString) =
    HDF5.read_attribute(obj, attrname)
has_attribute(obj::Union{HDF5.Dataset, HDF5.Group, HDF5.File}, attrname::AbstractString) =
    haskey(HDF5.attributes(obj), attrname)

is_compound(arr::HDF5.Dataset) = HDF5.API.h5t_get_class(HDF5.datatype(arr)) == HDF5.API.H5T_COMPOUND
is_bool(arr::HDF5.Dataset) = HDF5.datatype(arr) == datatype(Bool)

write_attribute(obj::Union{HDF5.File, HDF5.Group, HDF5.Dataset}, attrname::AbstractString, data) =
    HDF5.write_attribute(obj, attrname, data)
function write_attribute(obj::Union{HDF5.File, HDF5.Group, HDF5.Dataset}, attrname::AbstractString, data::Bool)
    dtype = datatype(Bool)
    dspace = HDF5.Dataspace(HDF5.API.h5s_create(HDF5.API.H5S_SCALAR))
    attr = create_attribute(obj, attrname, dtype, dspace)
    HDF5.write_attribute(attr, dtype, data)
end

read_scalar(d::HDF5.Dataset) = read(d)

function write_empty(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, dtype::Type)
    d = create_dataset(parent, name, dtype, dataspace(nothing))
    return d
end

function write_scalar(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::T,
) where {T <: Union{<:Number, <:AbstractString}}
    d, dtype = create_dataset(parent, name, data)
    write_dataset(d, dtype, data)
    return d
end

write_impl_array(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{Bool},
    compress::UInt8,
    extensible::Bool,
) = write_impl_array(parent, name, Array{UInt8}(data), compress, extensible, dtype=datatype(Bool))

function write_impl_array(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{T},
    compress::UInt8,
    extensible::Bool;
    dtype::HDF5.Datatype=datatype(T),
) where {T <: Number}
    if extensible
        dims = (size(data), Tuple(-1 for _ ∈ 1:ndims(data)))
    else
        dims = size(data)
    end
    if compress > 0x0 && ndims(data) > 0
        chunksize = HDF5.heuristic_chunk(data)
        if length(chunksize) == 0
            chunksize = Tuple(100 for _ ∈ 1:ndims(data))
        end
        d = create_dataset(parent, name, dtype, dims, chunk=chunksize, shuffle=true, deflate=compress)
    else
        d = create_dataset(parent, name, dtype, dims)
    end

    write_attribute(d, "encoding-type", "array")
    write_attribute(d, "encoding-version", "0.2.0")

    write_dataset(d, dtype, data)
end

# variable-length strings cannot be compressed in HDF5
function write_impl_array(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::AbstractArray{<:AbstractString},
    compress::UInt8,
    extensible::Bool,
)
    if extensible
        dims = (size(data), Tuple(-1 for _ ∈ 1:ndims(data)))
    else
        dims = size(data)
    end
    d, dtype = create_dataset(parent, name, data)
    write_attribute(d, "encoding-type", "string-array")
    write_attribute(d, "encoding-version", "0.2.0")
    write_dataset(d, dtype, data)
end

write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::Union{HDF5.Dataset, SparseDataset, TransposedDataset};
    kwargs...,
) = copy_object(data, parent, name)

function write_impl(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    data::StructArray;
    extensible::Bool=false,
    compress::UInt8=0x9,
    kwargs...,
)
    ety = eltype(data)
    dtype = create_datatype(HDF5.API.H5T_COMPOUND, sizeof(ety))
    for (i, (fname, ftype)) ∈ enumerate(zip(fieldnames(ety), fieldtypes(ety)))
        HDF5.API.h5t_insert(dtype, string(fname), fieldoffset(ety, i), datatype(ftype))
    end
    if compress > 0x0 && ndims(data) > 0
        chunksize = HDF5.heuristic_chunk(data)
        dims = size(data)
        if extensible
            dims = (size(data), Tuple(-1 for _ ∈ 1:ndims(data)))
        end
        dset = create_dataset(
            parent,
            name,
            dtype,
            dataspace(dims),
            chunk=chunksize,
            shuffle=true,
            deflate=compress,
            kwargs...,
        )
    else
        dset = create_dataset(parent, name, dtype, dataspace(dims), chunk=chunksize, kwargs...)
    end
    write_dataset(dset, dtype, [val for row ∈ data for val ∈ row])
end
