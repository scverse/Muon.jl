abstract type AbstractAlignedMapping{T <: Tuple, K, V} <: AbstractDict{K, V} end

struct AlignedMapping{T <: Tuple, K, R} <: AbstractAlignedMapping{
    T,
    K,
    Union{AbstractArray{<:Number}, AbstractArray{Union{Missing, T}} where T <: Number, DataFrame},
}
    ref::R # any type as long as it supports size()
    d::Dict{
        K,
        Union{
            AbstractArray{<:Number},
            AbstractArray{Union{Missing, T}} where T <: Number,
            DataFrame,
        },
    }

    function AlignedMapping{T, K}(r, d::AbstractDict{K}) where {T <: Tuple, K}
        for (k, v) in d
            checkdim(T, v, r, k)
        end
        return new{T, K, typeof(r)}(r, d)
    end
end

mutable struct BackedAlignedMapping{T <: Tuple, R} <:
               AbstractAlignedMapping{T, String, AbstractArray{<:Number}}
    ref::R
    d::Union{HDF5.Group, Nothing}
    parent::Union{HDF5.File, HDF5.Group, Nothing}
    path::Union{String, Nothing}

    function BackedAlignedMapping{T}(r, g::HDF5.Group) where {T <: Tuple}
        for k in keys(g)
            checkdim(T, backed_matrix(g[k]), r, k)
        end
        return new{T, typeof(r)}(r, g, nothing, nothing)
    end
    function BackedAlignedMapping{T}(
        r,
        parent::Union{HDF5.File, HDF5.Group},
        path::String,
    ) where {T <: Tuple}
        if haskey(parent, path)
            return BackedAlignedMapping{T}(r, parent[path])
        else
            return new{T, typeof(r)}(r, nothing, parent, path)
        end
    end
end

function checkdim(::Type{T}, v, ref, k) where {T <: Tuple}
    for (vdim, refdim) in T.parameters
        vsize = size(v, vdim)
        rsize = size(ref, refdim)
        if vsize != rsize
            throw(
                DimensionMismatch(
                    "Value passed for key $k had size $vsize for axis $vdim, but should have had size $rsize.",
                ),
            )
        end
    end
end

# @forward from ReusePatterns.jl doesn't work here because it also tries to forward methods from
# packages that we're not using, e.g. Serialization. This results in
# ERROR: LoadError: LoadError: Evaluation into the closed module `Serialization` breaks incremental
# compilation because the side effects will not be permanent. This is likely due to some other
# module mutating `Serialization` with `eval` during precompilation - don't do this.
#
# So we do this manually
Base.IteratorSize(::AbstractAlignedMapping) = Base.IteratorSize(Dict)
Base.IteratorEltype(::AbstractAlignedMapping) = Base.IteratorEltype(Dict)

Base.delete!(d::AlignedMapping, k) = delete!(d.d, k)
Base.empty!(d::AlignedMapping) = empty!(d.d)
Base.getindex(d::AlignedMapping, key) = getindex(d.d, key)
Base.get(d::AlignedMapping, key, default) = get(d.d, key, default)
Base.get(default::Base.Callable, d::AlignedMapping, key) = get(default, d.d, key)
Base.haskey(d::AlignedMapping, key) = haskey(d.d, key)
Base.isempty(d::AlignedMapping) = isempty(d.d)
Base.iterate(d::AlignedMapping) = iterate(d.d)
Base.iterate(d::AlignedMapping, i) = iterate(d.d, i)
Base.length(d::AlignedMapping) = length(d.d)
Base.pop!(d::AlignedMapping) = pop!(d.d)
Base.pop!(d::AlignedMapping, k) = pop!(d.d, k)
Base.pop!(d::AlignedMapping, k, default) = pop!(d.d, k, default)
function Base.setindex!(d::AlignedMapping{T}, v::Union{AbstractArray, DataFrame}, k) where {T}
    checkdim(T, v, d.ref, k)
    d.d[k] = v
end
Base.sizehint!(d::AbstractAlignedMapping, n) = sizehint!(d.d, n)

AlignedMapping{T}(r, d::AbstractDict) where {T <: Tuple} = AlignedMapping{T, keytype(d)}(r, d)
AlignedMapping{T, K}(ref) where {T, K} = AlignedMapping{T}(ref, Dict{K, AbstractMatrix{<:Number}}())
AlignedMapping{T, K}(ref, ::Nothing) where {T, K} = AlignedMapping{T, K}(ref)
AlignedMapping{T}(r, d::HDF5.Group) where {T <: Tuple} =
    AligedMapping{T}(ref, read_dict_of_mixed(d))

Base.delete!(d::BackedAlignedMapping, k) = !isnothing(d.d) && delete_object(d.d, k)
function Base.empty!(d::BackedAlignedMapping)
    if !isnothing(d.d)
        for k in keys(d.d)
            delete_object(d.d, k)
        end
    end
end
Base.getindex(d::BackedAlignedMapping, key) =
    isnothing(d.d) ? throw(KeyError(key)) : backed_matrix(d.d[key])
Base.get(d::BackedAlignedMapping, key, default) =
    isnothing(d.d) || !haskey(d.d, key) ? default : backed_matrix(d.d[key])
Base.get(default::Base.Callable, d::BackedAlignedMapping, key) =
    isnothing(d.d) || !haskey(d.d, key) ? default() : backed_matrix(d.d[key])
Base.haskey(d::BackedAlignedMapping, key) = isnothing(d.d) ? false : haskey(d.d, key)
Base.isempty(d::BackedAlignedMapping) = isnothing(d.d) ? true : isempty(d.d)
function Base.iterate(d::BackedAlignedMapping)
    if isnothing(d.d)
        return nothing
    else
        next = iterate(d.d)
        return isnothing(next) ? next :
               (hdf5_object_name(next[1]) => backed_matrix(next[1]), next[2])
    end
end
function Base.iterate(d::BackedAlignedMapping, i)
    if isnothing(d.d)
        return nothing
    else
        next = iterate(d.d, i)
        return isnothing(next) ? next :
               (hdf5_object_name(next[1]) => backed_matrix(next[1]), next[2])
    end
end
Base.length(d::BackedAlignedMapping) = isnothing(d.d) ? 0 : length(d.d)
function Base.pop!(d::BackedAlignedMapping)
    if isnothing(d.d)
        throw(ArgumentError("dict must be non-empty"))
    else
        obj = iterate(d.d)[1]
        mat = read_matrix(obj)
        delete_object(obj)
        return mat
    end
end
function Base.pop!(d::BackedAlignedMapping, k)
    if isnothing(d.d)
        throw(ArgumentError("dict must be non-empty"))
    else
        obj = d.d[k]
        mat = read_matrix(obj)
        delete_object(obj)
        return mat
    end
end
function Base.pop!(d::BackedAlignedMapping, k, default)
    try
        return pop!(d, k)
    catch e
        if e isa KeyError
            return default
        else
            rethrow()
        end
    end
end
function Base.setindex!(d::BackedAlignedMapping{T}, v::AbstractArray, k) where {T}
    checkdim(T, v, d.d, k)
    if isnothing(d.ref)
        d.ref = create_group(d.parent, d.path)
    end
    write_attr(d.g, k, v)
end

const StrAlignedMapping{T <: Tuple, R} = AlignedMapping{T, String, R}

function copy_subset(
    src::AbstractAlignedMapping{T},
    dst::AbstractAlignedMapping,
    I,
    J,
) where {T <: Tuple}
    idx = (
        if refdim == 1
            I
        elseif refdim == 2
            J
        else
            Colon()
        end for (vdim, refdim) in T.parameters
    )
    for (k, v) in src
        dst[k] = v[idx..., (Colon() for i in 1:(ndims(v) - length(idx)))...]
    end
end
