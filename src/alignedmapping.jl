struct AlignedMapping{T <: Tuple, K, R} <: AbstractDict{K, Union{AbstractArray{<:Number}, DataFrame}}
    ref::R # any type as long as it supports size()
    d::Dict{K, Union{AbstractArray{<:Number}, DataFrame}}

    function AlignedMapping{T, K}(r, d::AbstractDict{K}) where {T <: Tuple, K}
        for (k, v) in d
            checkdim(T, v, r, k)
        end
        return new{T, K, typeof(r)}(r, d)
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
Base.IteratorSize(::AlignedMapping) = Base.IteratorSize(Dict)
Base.IteratorEltype(::AlignedMapping) = Base.IteratorEltype(Dict)
Base.delete!(d::AlignedMapping, k) = delete!(d.d, k)
Base.empty!(d::AlignedMapping) = empty!(d.d)
Base.get(d::AlignedMapping, key, default) = get(d.d, key, default)
Base.get(default::Union{Function, Type}, d::AlignedMapping, key) = get(default, d.d, key)
Base.get!(default::Union{Function, Type}, d::AlignedMapping, key) = get!(default, d.d, key)
Base.getindex(d::AlignedMapping, key) = getindex(d.d, key)
Base.getkey(d::AlignedMapping, key, default) = getkey(d.d, key, default)
Base.haskey(d::AlignedMapping, key, default) = haskey(d.d, key, default)
Base.isempty(d::AlignedMapping) = isempty(d.d)
Base.iterate(d::AlignedMapping) = iterate(d.d)
Base.iterate(d::AlignedMapping, i) = iterate(d.d, i)
Base.length(d::AlignedMapping) = length(d.d)
Base.pop!(d::AlignedMapping) = pop!(d.d)
Base.pop!(d::AlignedMapping, k) = pop!(d.d, k)
Base.pop!(d::AlignedMapping, k, default) = pop!(d.d, k, default)
function Base.setindex!(d::AlignedMapping{T}, v, k) where {T}
    checkdim(T, v, d.ref, k)
    d.d[k] = v
end
Base.sizehint!(d::AlignedMapping, n) = sizehint!(d.d, n)

AlignedMapping{T}(r, d::AbstractDict) where {T <: Tuple} = AlignedMapping{T, keytype(d)}(r, d)
AlignedMapping{T, K}(ref) where {T, K} = AlignedMapping{T}(ref, Dict{K, AbstractMatrix{<:Number}}())
AlignedMapping{T, K}(ref, ::Nothing) where {T, K} = AlignedMapping{T, K}(ref)

const StrAlignedMapping{T <: Tuple, R} = AlignedMapping{T, String, R}
