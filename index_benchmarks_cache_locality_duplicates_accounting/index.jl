# Robin Hood hash table, linear probing, linear search, backshift deletion

abstract type AbstractIndex{T, V} <: AbstractVector{T} end
mutable struct Index{T, V} <: AbstractIndex{T, V}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}

    longestprobe::V
    initiallongestprobe::V
    deletions::V

    function Index{T}(nelements::Integer) where {T}
        size = ceil(nelements / 0.9) # 0.9 load factor

        mintype = minimum_unsigned_type_for_n(size)
        return new{T, mintype}(
            Vector{T}(undef, nelements),
            zeros(mintype, mintype(size)),
            zeros(mintype, mintype(size)),
            0,
            0,
            0,
        )
    end
end

_length(idx::Index) = length(idx.indices)

function _setindex!(idx::Index{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition::V = 0x0
    k = position
    location = hash(elem) % _length(idx) + 0x1
    prevlongestprobe = idx.longestprobe
    @inbounds while k != 0x0
        probeposition += 0x1
        recordpos = idx.probepositions[location]
        if probeposition > recordpos
            k, idx.indices[location] = idx.indices[location], k
            idx.probepositions[location] = probeposition
            idx.longestprobe = max(idx.longestprobe, probeposition)
            probeposition = recordpos
        end
        location = location == _length(idx) ? UInt(0x1) : location + 0x1
    end

    # reset longestprobe after some deletions deletions, this should maintain stable performance for _getindex and _getindex_array
    if idx.longestprobe == prevlongestprobe &&
       idx.longestprobe > idx.initiallongestprobe &&
       idx.deletions >= length(idx) ÷ 2
        idx.longestprobe = maximum(idx.probepositions)
        idx.deletions = 0x0
    end
end

function _getindex(idx::Index{T}, elem::T) where {T}
    ilength = _length(idx)
    if ilength > 0x0
        location = hash(elem) % ilength + 0x1
        @inbounds for probeposition ∈ 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.vals[pos], elem)
                return location
            elseif pos == 0x0
                return 0x0
            end
            location = location == ilength ? UInt(0x1) : location + 0x1
        end
    end
    return 0x0
end

function _getindex_array(idx::Index{T, V}, elem::T) where {T, V}
    locations = Vector{V}()
    ilength = _length(idx)
    if ilength > 0x0
        location = hash(elem) % ilength + 0x1
        @inbounds for probeposition ∈ 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.vals[pos], elem)
                push!(locations, location)
            elseif pos == 0x0
                return locations
            end
            location = location == ilength ? UInt(0x1) : location + 0x1
        end
    end
    return locations
end

function _getindex_byposition(idx::Index{T}, i::Integer) where {T}
    elem = idx.vals[i]
    ilength = _length(idx)
    if ilength > 0x0
        location = hash(elem) % ilength + 0x1
        @inbounds for probeposition ∈ 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && pos == i
                return location
            elseif pos == 0x0
                break
            end
            location = location == ilength ? UInt64(0x1) : location + 0x1
        end
    end
    error("Element not found. This should never happen.")
end

function _delete!(idx::Index, oldkeyindex::Integer)
    lastidx = findnext(≤(0x1), idx.probepositions, oldkeyindex + 0x1)
    @inbounds if !isnothing(lastidx)
        lastidx -= 0x1
        idx.indices[oldkeyindex:(lastidx - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):lastidx]
        idx.probepositions[oldkeyindex:(lastidx - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):lastidx]) .- 0x1
        idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
    else
        if oldkeyindex < _length(idx)
            idx.indices[oldkeyindex:(end - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):end]
            idx.probepositions[oldkeyindex:(end - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):end]) .- 0x1
        end
        if idx.probepositions[1] > 0x1
            idx.indices[end] = idx.indices[1]
            idx.probepositions[end] = idx.probepositions[1] - 0x1

            lastidx = findnext(≤(0x1), idx.probepositions, 2)
            if isnothing(lastidx)
                lastidx = oldkeyindex
            end
            lastidx -= 0x1
            idx.indices[0x1:(lastidx - 0x1)] .= @view idx.indices[0x2:lastidx]
            idx.probepositions[0x1:(lastidx - 0x1)] .= @view(idx.probepositions[0x2:lastidx]) .- 0x1
            idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
        else
            idx.indices[end] = idx.probepositions[end] = 0x0
        end
    end

    idx.deletions += 0x1
end

function Index(elems::AbstractVector{T}) where {T}
    idx = Index{T}(length(elems))
    for (i, e) ∈ enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    idx.initiallongestprobe = idx.longestprobe
    return idx
end

Base.getindex(idx::AbstractIndex{T}, elem::T) where {T} = getindex(idx, elem, Val(false))
Base.getindex(idx::AbstractIndex{T}, elem::T, x::Bool) where {T} = getindex(idx, elem, Val(x))
Base.getindex(idx::AbstractIndex{T}, elem::T, x::Bool, y::Bool) where {T} = getindex(idx, elem, Val(x), Val(y))
Base.getindex(idx::AbstractIndex{T}, elems::AbstractVector{T}) where {T} = getindex(idx, elems, Val(false))
Base.getindex(idx::AbstractIndex{T}, elems::AbstractVector{T}, x::Bool) where {T} = getindex(idx, elems, Val(x))
Base.getindex(idx::AbstractIndex{T, V}, elems::AbstractVector{T}, x::Union{Val{true}, Val{false}}) where {T, V} =
    isempty(elems) ? V[] : reduce(vcat, (getindex(idx, elem, x) for elem ∈ elems))
Base.in(elem::T, idx::AbstractIndex{T}) where {T} = getindex(idx, elem, Val(false), Val(false)) != 0x0

Base.getindex(idx::Index{T}, elem::T, ::Val{true}, ::Val{false}) where {T} =
    @inbounds idx.indices[_getindex_array(idx, elem)] # exceptions may be undesirable in high-performance scenarios
function Base.getindex(idx::Index{T}, elem::T, ::Val{true}) where {T}
    d = _getindex_array(idx, elem)
    if length(d) == 0
        throw(KeyError(elem))
    end
    @inbounds return idx.indices[d]
end

function Base.getindex(idx::Index{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    i = _getindex(idx, elem)
    return i == 0x0 ? V(0x0) : @inbounds idx.indices[i]
end
function Base.getindex(idx::Index{T}, elem::T, ::Val{false}) where {T}
    i = getindex(idx, elem, Val(false), Val(false))
    if i == 0x0
        throw(KeyError(elem))
    end
    return i
end

Base.@propagate_inbounds function Base.getindex(idx::Index, i::Integer)
    return idx.vals[i]
end

@inline function Base.setindex!(idx::Index{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(idx.vals, i)
    oldkeyindex = _getindex_byposition(idx, i)
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex)
    _setindex!(idx, newval, validx)
    return idx
end

function Base.setindex!(idx::Index{T}, newval::T, oldval::T) where {T}
    oldkeyindex = _getindex(idx, oldval)
    if oldkeyindex == 0x0
        throw(KeyError(oldval))
    end
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex)
    _setindex!(idx, newval, validx)
    return idx
end

Base.length(idx::Index) = length(idx.vals)
Base.size(idx::Index) = (length(idx),)
Base.values(idx::Index) = idx.vals
Base.convert(::Type{<:Index}, x::AbstractArray) = Index(x)
Base.convert(::Type{<:Index}, x::Index) = x

struct SubIndex{T, V, I} <: AbstractIndex{T, V}
    parent::Index{T, V}
    indices::I
    revmapping::Union{Nothing, Index}
end
SubIndex(idx::Index{T, V}, indices::I) where {T, V, I} = SubIndex{T, V, I}(idx, indices, nothing)

@inline function Base.view(idx::Index, I::Union{AbstractRange, Colon})
    @boundscheck checkbounds(idx, I)
    return SubIndex(idx, I)
end
@inline function Base.view(idx::Index{T, V}, I::AbstractArray{<:Integer}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex(idx, I, Index(V.(I)))
end
@inline function Base.view(idx::Index{T, V}, I::AbstractArray{Bool}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex(idx, I, Index(V.(findall(I))))
end
@inline function Base.view(idx::Index, I::Integer)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(idx, I:I)
end
@inline function Base.view(idx::SubIndex, I)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(parent(idx), Base.reindex((parentindices(idx),), (I,))[1])
end

Base.copy(si::SubIndex) = Index(si)
Base.parent(si::SubIndex) = si.parent
Base.parentindices(si::SubIndex) = si.indices
Base.length(si::SubIndex) = length(parentindices(si))
Base.length(si::SubIndex{T, V, Colon}) where {T, V} = length(parent(si))
Base.length(si::SubIndex{T, V, I}) where {T, V, I <: AbstractArray{Bool}} = length(si.revmapping)
Base.size(si::SubIndex) = (length(si),)
Base.values(si::SubIndex) = parent(si)[parentindices(si)]
function Base.getindex(si::SubIndex{T}, elem::T, ::Val{true}, ::Val{false}) where {T}
    res = getindex(parent(si), elem, Val(true), Val(false))
    i = 0
    @inbounds for r ∈ res
        position = findfirst(isequal(r), parentindices(si))
        if !isnothing(position)
            i += 1
            res[i] = position
        end
    end
    resize!(res, i)
    return res
end
Base.getindex(si::SubIndex{T, V, Colon}, elem::T, ::Val{true}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(true), Val(false))
function Base.getindex(si::SubIndex{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractArray{Bool}}
    res = getindex(parent(si), elem, Val(true), Val(false))
    res = res[parentindices(si)[res]]
    @inbounds for (i, r) ∈ enumerate(res)
        res[i] = si.revmapping[r, false, false]
    end

    return res
end
function Base.getindex(
    si::SubIndex{T, V, I},
    elem::T,
    ::Val{true},
    ::Val{false},
) where {T, V, I <: AbstractArray{<:Integer}}
    res = getindex(parent(si), elem, Val(true), Val(false))
    i = 0
    @inbounds for r ∈ res
        position = si.revmapping[r, false, false]
        if position > 0x0
            i += 1
            res[i] = position
        end
    end
    return resize!(res, i)
end
function Base.getindex(si::SubIndex{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractRange}
    res = getindex(parent(si), elem, Val(true), Val(false))
    j = 0
    for i ∈ 1:length(res)
        pos = findfirst(isequal(res[i]), parentindices(si))
        if !isnothing(pos)
            j += 1
            res[j] = pos
        end
    end
    return resize!(res, j)
end
function Base.getindex(si::SubIndex{T}, elem::T, ::Val{true}) where {T}
    res = getindex(si, elem, Val(true), Val(false))
    if length(res) == 0
        throw(KeyError(elem))
    end
    return res
end

function Base.getindex(si::SubIndex{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    res = getindex(parent(si), elem, Val(false), Val(false))
    position = findfirst(isequal(res), parentindices(si))
    return isnothing(position) ? V(0x0) : V(position)
end
Base.getindex(si::SubIndex{T, V, Colon}, elem::T, ::Val{false}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(false), Val(false))
function Base.getindex(
    si::SubIndex{T, V, I},
    elem::T,
    ::Val{false},
    ::Val{false},
) where {T, V, I <: AbstractArray{Bool}}
    res = getindex(parent(si, elem, Val(false), Val(false)))
    if res == 0x0 || res > 0x0 && !(@inbounds parentindices(si)[res])
        return V(0x0)
    end
    return si.revmapping[res, false, false]
end
function Base.getindex(
    si::SubIndex{T, V, I},
    elem::T,
    ::Val{false},
    ::Val{false},
) where {T, V, I <: AbstractArray{<:Integer}}
    res = getindex(parent(si), elem, Val(false), Val(false))
    return res > 0x0 ? si.revmapping[res, false, false] : res
end
function Base.getindex(si::SubIndex{T, V, I}, elem::T, ::Val{false}, ::Val{false}) where {T, V, I <: AbstractRange}
    res = getindex(parent(si), elem, Val(false), Val(false))
    if res > 0x0
        i = findfirst(isequal(res), parentindices(si))
        return isnothing(i) ? 0x0 : i
    else
        return 0x0
    end
end
function Base.getindex(si::SubIndex{T}, elem::T, ::Val{false}) where {T}
    res = getindex(si, elem, Val(false), Val(false))
    if res == 0x0
        throw(KeyError(elem))
    end
    return res
end
Base.@propagate_inbounds function Base.getindex(si::SubIndex, i::Union{Integer, AbstractVector{<:Integer}})
    @boundscheck checkbounds(si, i)
    return parent(si)[Base.reindex((parentindices(si),), (i,))[1]]
end
Base.@propagate_inbounds function Base.getindex(
    si::SubIndex{T, V, Colon},
    i::Union{Integer, AbstractVector{<:Integer}},
) where {T, V}
    @boundscheck checkbounds(si, i)
    return parent(si)[i]
end

Base.@propagate_inbounds function Base.setindex!(si::SubIndex{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, Base.reindex((parentindices(si),), (i,))[1])
    return si
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex{T, V, Colon}, newval::T, i::Integer) where {T, V}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, i)
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex{T}, newval::T, oldval::T) where {T}
    oldidx = parent(si)[oldval, true]
    foldidx = findfirst(in(parentindices(si)), oldidx)
    if isnothing(foldidx)
        throw(KeyError(oldval))
    end
    parent(si)[oldidx[foldidx]] = newval
    return si
end
Base.@propagate_inbounds function Base.setindex!(
    si::SubIndex{T, V, I},
    newval::T,
    oldval::T,
) where {T, V, I <: AbstractArray{Bool}}
    oldidx = parent(si)[oldval, true]
    oldidx = oldidx[parentindices(si)[oldix]]
    if length(oldidx) == 0
        throw(KeyError(oldval))
    end
    parent(si)[oldidx[1]] = newval
    return si
end
Base.@propagate_inbounds function Base.setindex!(
    si::SubIndex{T, V, I},
    newval::T,
    oldval::T,
) where {T, V, I <: AbstractArray{<:Integer}}
    oldidx = parent(si)[oldval, true]
    foldidx = findfirst(in(si.revmapping), oldidx)
    if isnothing(foldidx)
        throw(KeyError(oldval))
    end
    parent(si)[oldidx[foldidx]] = newval
    return si
end
