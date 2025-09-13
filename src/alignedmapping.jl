abstract type AbstractAlignedMapping{T <: Tuple, K, V} <: AbstractDict{K, V} end

struct AlignedMapping{T <: Tuple, K, R, D, V} <: AbstractAlignedMapping{T, K, V}
    ref::R # any type as long as it supports size()
    d::Dict{K, V}

    function AlignedMapping{T, K, D}(r, d::AbstractDict{K}) where {T <: Tuple, K, D}
        @assert isa(D, Bool)
        for (k, v) ∈ d
            checkdim(T, v, r, k)
        end

        V = Union{AbstractArray{<:Number}, AbstractArray{Union{Missing, T}} where T <: Number, D ? DataFrame : Union{}}
        return new{T, K, typeof(r), D, V}(r, d)
    end
end

mutable struct BackedAlignedMapping{T <: Tuple, G <: Group, R} <:
               AbstractAlignedMapping{T, String, AbstractArray{<:Number}}
    ref::R
    d::Union{G, Nothing}
    parent::Union{G, Nothing}
    path::Union{String, Nothing}

    function BackedAlignedMapping{T}(r, g) where {T <: Tuple}
        for k ∈ keys(g)
            checkdim(T, backed_matrix(g[k]), r, k)
        end
        return new{T, typeof(g), typeof(r)}(r, g, nothing, nothing)
    end
    function BackedAlignedMapping{T}(r, parent::Group, path::String) where {T <: Tuple}
        if haskey(parent, path)
            return BackedAlignedMapping{T}(r, parent[path])
        else
            return new{T, typeof(parent), typeof(r)}(r, nothing, parent, path)
        end
    end
end

function checkdim(::Type{T}, v, ref, k) where {T <: Tuple}
    for (vdim, refdim) ∈ T.parameters
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
    if ndims(v) == 1
        v = reshape(v, :, 1)
    end
    d.d[k] = v
end
Base.sizehint!(d::AlignedMapping, n) = sizehint!(d.d, n)

AlignedMapping{T, K}(r, d::AbstractDict{K}, ::Val{true}) where {T <: Tuple, K} = AlignedMapping{T, K, true}(r, d)
AlignedMapping{T, K}(r, d::AbstractDict{K}, ::Val{false}) where {T <: Tuple, K} = AlignedMapping{T, K, false}(r, d)
AlignedMapping{T, K}(r, d::AbstractDict{K}) where {T <: Tuple, K} = AlignedMapping{T, K}(r, d, Val(true))

AlignedMapping{T}(r, d::AbstractDict) where {T <: Tuple} = AlignedMapping{T, keytype(d)}(r, d)
AlignedMapping{T}(r, d::AbstractDict, ::Val{true}) where {T <: Tuple} = AlignedMapping{T, keytype(d), true}(r, d)
AlignedMapping{T}(r, d::AbstractDict, ::Val{false}) where {T <: Tuple} = AlignedMapping{T, keytype(d), false}(r, d)

AlignedMapping{T}(r, d::Group) where {T <: Tuple} = AligedMapping{T}(ref, read_dict_of_mixed(d))
AlignedMapping{T}(r, d::Group, ::Val{true}) where {T <: Tuple} = AligedMapping{T}(ref, read_dict_of_mixed(d), Val(true))
AlignedMapping{T}(r, d::Group, ::Val{false}) where {T <: Tuple} =
    AligedMapping{T}(ref, read_dict_of_mixed(d), Val(false))

AlignedMapping{T, K}(ref) where {T, K} = AlignedMapping{T, K}(ref, Dict{K, AbstractMatrix{<:Number}}())
AlignedMapping{T, K}(ref, ::Nothing) where {T, K} = AlignedMapping{T, K}(ref)
AlignedMapping{T, K, D}(ref) where {T, K, D} = AlignedMapping{T, K, D}(ref, Dict{K, AbstractMatrix{<:Number}}())
AlignedMapping{T, K, D}(ref, ::Nothing) where {T, K, D} = AlignedMapping{T, K, D}(ref)

Base.delete!(d::BackedAlignedMapping, k) = !isnothing(d.d) && delete_object(d.d, k)
function Base.empty!(d::BackedAlignedMapping)
    if !isnothing(d.d)
        for k ∈ keys(d.d)
            delete_object(d.d, k)
        end
    end
end
Base.getindex(d::BackedAlignedMapping, key) = isnothing(d.d) ? throw(KeyError(key)) : backed_matrix(d.d[key])
Base.get(d::BackedAlignedMapping, key, default) =
    isnothing(d.d) || !haskey(d.d, key) ? default : backed_matrix(d.d[key])
Base.get(default::Base.Callable, d::BackedAlignedMapping, key) =
    isnothing(d.d) || !haskey(d.d, key) ? default() : backed_matrix(d.d[key])
Base.haskey(d::BackedAlignedMapping, key) = isnothing(d.d) ? false : haskey(d.d, key)
Base.isempty(d::BackedAlignedMapping) = isnothing(d.d) ? true : isempty(d.d)
function Base.iterate(d::BackedAlignedMapping{T, G}, i=nothing) where {T, G <: Union{HDF5.File, HDF5.Group}}
    if isnothing(d.d)
        return nothing
    else
        next = iterate(d.d, i)
        return isnothing(next) ? next : (hdf5_object_name(next[1]) => backed_matrix(next[1]), next[2])
    end
end
function Base.iterate(d::BackedAlignedMapping{T, G}, i=nothing) where {T, G <: ZGroup}
    if (isnothing(d.d))
        return nothing
    else
        next = iterate(d.d, i)
        return isnothing(next) ? next : (next[1][1] => backed_matrix(next[1][2]), next[2])
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
    checkdim(T, v, d.ref, k)
    if isnothing(d.d)
        d.d = create_group(d.parent, d.path)
    end
    write_attr(d.d, k, v)
end

function copy_subset(src::AbstractAlignedMapping{T}, dst::AbstractAlignedMapping, I, J) where {T <: Tuple}
    idx = (if refdim == 1
        I
    elseif refdim == 2
        J
    else
        (:)
    end for (vdim, refdim) ∈ T.parameters)
    for (k, v) ∈ src
        dst[k] = v[idx..., ..]
    end
end

struct AlignedMappingView{T <: Tuple, K, V, P <: AbstractAlignedMapping{T, K, V}} <: AbstractAlignedMapping{T, K, V}
    parent::P
    indices::Tuple
end

function aligned_view(d::AlignedMappingView{T}, A) where {T <: Tuple}
    idx = Vector{Union{Colon, typeof(d.indices).parameters...}}(undef, ndims(A))
    idx .= (:)
    for ((vdim, refdim), cidx) ∈ zip(T.parameters, d.indices)
        idx[vdim] = cidx
    end
    return @inbounds view(A, idx...)
end

Base.delete!(d::AlignedMappingView, k) = delete!(d.parent, k)
Base.empty!(d::AlignedMappingView) = empty!(d.parent)
Base.getindex(d::AlignedMappingView, key) = aligned_view(d, getindex(d.parent, key))
Base.get(d::AlignedMappingView, key, default) = haskey(d.parent, key) ? aligned_view(d, get(d.parent, key)) : default
Base.get(default::Base.Callable, d::AlignedMappingView, key) =
    haskey(d.parent, key) ? aligned_view(d, get(d.parent, key)) : default()
Base.haskey(d::AlignedMappingView, key) = haskey(d.parent, key)
Base.isempty(d::AlignedMappingView) = isempty(d.parent)
function Base.iterate(d::AlignedMappingView)
    res = iterate(d.parent)
    if !isnothing(res)
        (k, v), state = res
        return (k, aligned_view(d, v)), state
    else
        return res
    end
end
function Base.iterate(d::AlignedMappingView, i)
    res = iterate(d.parent, i)
    if !isnothing(res)
        (k, v), state = res
        return (k, aligned_view(d, v)), state
    else
        return res
    end
end
Base.length(d::AlignedMappingView) = length(d.parent)
Base.pop!(d::AlignedMappingView) = aligned_view(d, pop!(d.parent))
Base.pop!(d::AlignedMappingView, k) = aligned_view(d, pop!(d.parent, k))
Base.pop!(d::AlignedMappingView, k, default) = haskey(d.parent, k) ? aligned_view(d, pop!(d.d, k)) : default
Base.parent(d::AlignedMappingView) = d.parent
Base.parentindices(d::AlignedMappingView) = d.indices

Base.setindex!(d::AlignedMappingView, v::AbstractArray, k) =
    throw(ArgumentError("Replacing or adding elements of an AlignedMappingView is not supported."))

function Base.view(parent::AbstractAlignedMapping{T}, indices...) where {T <: Tuple}
    @boundscheck if length(T.parameters) != length(indices)
        throw(
            DimensionMismatch(
                "Attempt to index into AlignedMapping with $(length(T.parameters)) aligned dimensions using index of length $(length(indices))",
            ),
        )
    end
    return AlignedMappingView(parent, indices)
end

function Base.view(parentview::AlignedMappingView{T}, indices...) where {T <: Tuple}
    @boundscheck if length(T.parameters) != length(indices)
        throw(
            DimensionMismatch(
                "Attempt to index into AlignedMappingView with $(length(T.parameters)) aligned dimensions using index of length $(length(indices))",
            ),
        )
    end
    return AlignedMappingView(parent(parentview), Base.reindex(parentindices(parentview), indices))
end

const StrAlignedMapping{T <: Tuple, R, D, V} = AlignedMapping{T, String, R, D, V}
const StrAlignedMappingView{T <: Tuple} = AlignedMappingView{T, String}
