read_attribute(obj::Union{ZArray, ZGroup}, attrname::AbstractString) = obj.attrs[attrname]
attributes(obj::Union{ZArray, ZGroup}) = obj.attrs
Base.read(arr::ZArray) = collect(arr)
Base.read(grp::ZGroup, name::AbstractString) = collect(grp[name])

create_group(parent::ZGroup, name::AbstractString) = zgroup(parent, name)

delete_from_store(store::Zarr.AbstractStore, path::AbstractString, name::AbstractString) =
    delete!(store, path, name)
delete_from_store(store::Zarr.DirectoryStore, path::AbstractString, name::AbstractString) =
    rm(joinpath(store.folder, path, name), recursive=true)

function delete_object(parent::ZGroup, name::AbstractString)
    delete_from_store(parent.storage, parent.path, name)
    if haskey(parent.groups, name)
        delete!(parent.groups, name)
    else
        delete!(parent.arrays, name)
    end
end

is_compound(arr::ZArray) = false
is_bool(arr::ZArray) = eltype(arr) == Bool

Base.keys(group::ZGroup) = Iterators.flatten((keys(group.arrays), keys(group.groups)))
Base.length(group::ZGroup) = length(group.arrays) + length(group.groups)
Base.close(obj::Union{ZArray, ZGroup}) = nothing

function Base.iterate(group::ZGroup, i=nothing)
    iter = isnothing(i) ? keys(group) : i[1]
    next = isnothing(i) ? iterate(iter) : iterate(iter, i[2])
    return isnothing(next) ? next : ((next[1] => group[next[1]]), (iter, next[2]))
end

function write_attribute(obj::Union{ZArray, ZGroup}, attrname::AbstractString, data)
    obj.attrs[attrname] = data
    Zarr.writeattrs(obj.storage, obj.path, obj.attrs)
end

read_scalar(d::ZArray) = d[]

function write_scalar(
    parent::ZGroup,
    name::AbstractString,
    data::T,
) where {T <: Union{<:Number, <:AbstractString}}
    d = zcreate(T, parent, name)
    d[] = data
    return d
end

function write_array_encoding(dataset::ZArray{<:Number})
    write_attribute(dataset, "encoding-type", "array")
    write_attribute(dataset, "encoding-version", "0.2.0")
end
function write_array_encoding(dataset::ZArray{<:AbstractString})
    write_attribute(dataset, "encoding-type", "string-array")
    write_attribute(dataset, "encoding-version", "0.2.0")
end

function write_impl(parent::ZGroup, name::AbstractString, data::T; kwargs...) where {T <: Number}
    d = zcreate(T, parent, name)
    d[] = data
    write_attribute(parent[name], "encoding-type", "numeric-scalar")
    write_attribute(parent[name], "encoding-version", "0.2.0")
end

function write_impl(
    parent::ZGroup,
    name::AbstractString,
    data::T;
    kwargs...,
) where {T <: AbstractString}
    d = zcreate(T, parent, name)
    d[] = data
    write_attribute(parent[name], "encoding-type", "string")
    write_attribute(parent[name], "encoding-version", "0.2.0")
end

function write_impl_array(
    parent::ZGroup,
    name::AbstractString,
    data::Union{AbstractArray{<:Number}, AbstractArray{<:AbstractString}},
    compress::UInt8,
    extensible::Bool,
)
    dtype = eltype(data)
    if compress > 0x0 && ndims(data) > 0
        chunksize = Tuple(zarr_heuristic_chunk(data))
        if length(chunksize) == 0
            chunksize = Tuple(100 for _ ∈ 1:ndims(data))
        end
        d = zcreate(
            dtype,
            parent,
            name,
            size(data)...,
            chunks=chunksize,
            compressor=Zarr.BloscCompressor(clevel=compress),
        )
    else
        d = zcreate(dtype, parent, name, size(data)..., compressor=Zarr.NoCompressor())
    end

    write_array_encoding(d)
    d[:] = data
end

function zarr_filename(group::ZGroup)
    candidates =
        filter((i, type) -> type <: AbstractString, enumerate(fieldtypes(typeof(group.storage))))
    return candidates[1]
end

# port of Python Zarr chunk size heuristic
_sizeof(x::Type) = sizeof(x)
_sizeof(x::Type{<:AbstractString}) = 8 # same as numpy
function zarr_heuristic_chunk(
    arr::AbstractArray{T};
    increment_bytes::Int=256 * 1024,
    min_bytes::Int=128 * 1024,
    max_bytes::Int=64 * 1024 * 1024,
) where {T}
    nd = ndims(arr)
    _sizeof(T) > max_bytes && return Tuple(ones(Int, nd))

    fullsize = length(arr) * _sizeof(T)
    chunks = [size(arr)...]
    target_size =
        clamp(round(Int, increment_bytes * 2^log10(fullsize / (1024 * 1024))), min_bytes, max_bytes)

    # start with right-most dimension to account for column-major order in Julia
    cdim = nd
    while prod(chunks) * _sizeof(T) > target_size
        chunks[cdim] >>>= 1
        (cdim -= 1) < 1 && (cdim = nd)
    end
    return Tuple(chunks)
end
