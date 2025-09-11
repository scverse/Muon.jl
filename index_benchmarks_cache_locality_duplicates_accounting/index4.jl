# Robin Hood hash table, linear probing, linear search, backshift deletion

abstract type AbstractIndex4{T, V} <: AbstractVector{T} end
mutable struct Index4{T, V} <: AbstractIndex4{T, V}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}

    longestprobe::V
    initiallongestprobe::V
    deletions::V
    duplicates::V

    function Index4{T}(nelements::Integer) where {T}
        size = ceil(nelements / 0.9) # 0.9 load factor

        mintype = minimum_unsigned_type_for_n(size)
        return new{T, mintype}(
            Vector{T}(undef, nelements),
            zeros(mintype, mintype(size)),
            zeros(mintype, mintype(size)),
            0x0,
            0x0,
            0x0,
            0x0,
        )
    end
end

_length(idx::Index4) = length(idx.indices)

function _setindex!(idx::Index4{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition::V = 0x0
    k = position
    location = hash(elem) % _length(idx) + 0x1
    prevlongestprobe = idx.longestprobe
    is_duplicate = false
    @inbounds while k != 0x0
        probeposition += 0x1
        recordpos = idx.probepositions[location]
        if !is_duplicate && recordpos > 0 &&  isequal(idx.vals[idx.indices[location]], elem)
            idx.duplicates += 1
            is_duplicate = true
        end
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

function _getindex(idx::Index4{T}, elem::T) where {T}
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

function _getindex_array(idx::Index4{T, V}, elem::T) where {T, V}
    if allunique(idx)
        return [_getindex(idx, elem)]
    end
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

function _getindex_byposition(idx::Index4{T}, i::Integer) where {T}
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

function _delete!(idx::Index4, oldkeyindex::Integer)
    lastidx = findnext(≤(0x1), idx.probepositions, oldkeyindex + 0x1)
    @inbounds if !isnothing(lastidx)
        lastidx -= 0x1

        if !allunique(idx)
            subvals = @view idx.vals[idx.indices[oldkeyindex:lastidx]]
            if !isnothing(findnext(isequal(subvals[1]), subvals, 2))
                idx.duplicates -= 0x1
            end
        end

        idx.indices[oldkeyindex:(lastidx - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):lastidx]
        idx.probepositions[oldkeyindex:(lastidx - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):lastidx]) .- 0x1
        idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
    else
        is_duplicate = false
        if oldkeyindex < _length(idx)
            if !allunique(idx)
                subvals = @view idx.vals[idx.indices[oldkeyindex:end]]
                if !isnothing(findnext(isequal(subvals[1]), subvals, 2))
                    is_duplicate = true
                    idx.duplicates -= 0x1
                end
            else
                is_duplicate = true
            end

            idx.indices[oldkeyindex:(end - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):end]
            idx.probepositions[oldkeyindex:(end - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):end]) .- 0x1
        end
        if idx.probepositions[1] > 0x1
            lastidx = findnext(≤(0x1), idx.probepositions, 2)
            if isnothing(lastidx)
                lastidx = oldkeyindex
            end
            lastidx -= 0x1

            if !is_duplicate
                subvals = @view idx.vals[idx.indices[1:lastidx]]
                if !isnothing(findnext(isequal(subvals[1]), subvals, 2))
                    is_duplicate = true
                    idx.duplicates -= 0x1
                end
            end

            idx.indices[end] = idx.indices[1]
            idx.probepositions[end] = idx.probepositions[1] - 0x1
            idx.indices[0x1:(lastidx - 0x1)] .= @view idx.indices[0x2:lastidx]
            idx.probepositions[0x1:(lastidx - 0x1)] .= @view(idx.probepositions[0x2:lastidx]) .- 0x1
            idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
        else
            idx.indices[end] = idx.probepositions[end] = 0x0
        end
    end

    idx.deletions += 0x1
end

function Index4(elems::AbstractVector{T}) where {T}
    idx = Index4{T}(length(elems))
    for (i, e) ∈ enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    idx.initiallongestprobe = idx.longestprobe
    return idx
end

Base.getindex(idx::AbstractIndex4{T}, elem::T) where {T} = getindex(idx, elem, Val(false))
Base.getindex(idx::AbstractIndex4{T}, elem::T, x::Bool) where {T} = getindex(idx, elem, Val(x))
Base.getindex(idx::AbstractIndex4{T}, elem::T, x::Bool, y::Bool) where {T} = getindex(idx, elem, Val(x), Val(y))
Base.getindex(idx::AbstractIndex4{T}, elems::AbstractVector{T}) where {T} = getindex(idx, elems, Val(false))
Base.getindex(idx::AbstractIndex4{T}, elems::AbstractVector{T}, x::Bool) where {T} = getindex(idx, elems, Val(x))
Base.getindex(idx::AbstractIndex4{T, V}, elems::AbstractVector{T}, x::Union{Val{true}, Val{false}}) where {T, V} =
    isempty(elems) ? V[] : reduce(vcat, (getindex(idx, elem, x) for elem ∈ elems))
Base.in(elem::T, idx::AbstractIndex4{T}) where {T} = getindex(idx, elem, Val(false), Val(false)) != 0x0

@inline Base.allunique(idx::Index4) = idx.duplicates == 0x0
Base.getindex(idx::Index4{T}, elem::T, ::Val{true}, ::Val{false}) where {T} =
    @inbounds idx.indices[_getindex_array(idx, elem)] # exceptions may be undesirable in high-performance scenarios
function Base.getindex(idx::Index4{T}, elem::T, ::Val{true}) where {T}
    d = _getindex_array(idx, elem)
    if length(d) == 0
        throw(KeyError(elem))
    end
    @inbounds return idx.indices[d]
end

function Base.getindex(idx::Index4{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    i = _getindex(idx, elem)
    return i == 0x0 ? V(0x0) : @inbounds idx.indices[i]
end
function Base.getindex(idx::Index4{T}, elem::T, ::Val{false}) where {T}
    i = getindex(idx, elem, Val(false), Val(false))
    if i == 0x0
        throw(KeyError(elem))
    end
    return i
end

Base.@propagate_inbounds function Base.getindex(idx::Index4, i::Integer)
    return idx.vals[i]
end

@inline function Base.setindex!(idx::Index4{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(idx.vals, i)
    oldkeyindex = _getindex_byposition(idx, i)
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex)
    _setindex!(idx, newval, validx)
    return idx
end

function Base.setindex!(idx::Index4{T}, newval::T, oldval::T) where {T}
    oldkeyindex = _getindex(idx, oldval)
    if oldkeyindex == 0x0
        throw(KeyError(oldval))
    end
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex)
    _setindex!(idx, newval, validx)
    return idx
end

Base.length(idx::Index4) = length(idx.vals)
Base.size(idx::Index4) = (length(idx),)
Base.values(idx::Index4) = idx.vals
Base.convert(::Type{<:Index4}, x::AbstractArray) = Index4(x)
Base.convert(::Type{<:Index4}, x::Index4) = x

struct SubIndex4{T, V, I} <: AbstractIndex4{T, V}
    parent::Index4{T, V}
    indices::I
    revmapping::Union{Nothing, Index4}
end
SubIndex4(idx::Index4{T, V}, indices::I) where {T, V, I} = SubIndex4{T, V, I}(idx, indices, nothing)

@inline function Base.view(idx::Index4, I::Union{AbstractRange, Colon})
    @boundscheck checkbounds(idx, I)
    return SubIndex4(idx, I)
end
@inline function Base.view(idx::Index4{T, V}, I::AbstractArray{<:Integer}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex4(idx, I, Index4(V.(I)))
end
@inline function Base.view(idx::Index4{T, V}, I::AbstractArray{Bool}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex4(idx, I, Index4(V.(findall(I))))
end
@inline function Base.view(idx::Index4, I::Integer)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(idx, I:I)
end
@inline function Base.view(idx::SubIndex4, I)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(parent(idx), Base.reindex((parentindices(idx),), (I,))[1])
end

Base.copy(si::SubIndex4) = Index4(si)
Base.parent(si::SubIndex4) = si.parent
Base.parentindices(si::SubIndex4) = si.indices
Base.length(si::SubIndex4) = length(parentindices(si))
Base.length(si::SubIndex4{T, V, Colon}) where {T, V} = length(parent(si))
Base.length(si::SubIndex4{T, V, I}) where {T, V, I <: AbstractArray{Bool}} = length(si.revmapping)
Base.size(si::SubIndex4) = (length(si),)
Base.values(si::SubIndex4) = parent(si)[parentindices(si)]
function Base.getindex(si::SubIndex4{T}, elem::T, ::Val{true}, ::Val{false}) where {T}
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
Base.getindex(si::SubIndex4{T, V, Colon}, elem::T, ::Val{true}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(true), Val(false))
function Base.getindex(si::SubIndex4{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractArray{Bool}}
    res = getindex(parent(si), elem, Val(true), Val(false))
    res = res[parentindices(si)[res]]
    @inbounds for (i, r) ∈ enumerate(res)
        res[i] = si.revmapping[r, false, false]
    end

    return res
end
function Base.getindex(
    si::SubIndex4{T, V, I},
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
function Base.getindex(si::SubIndex4{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractRange}
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
function Base.getindex(si::SubIndex4{T}, elem::T, ::Val{true}) where {T}
    res = getindex(si, elem, Val(true), Val(false))
    if length(res) == 0
        throw(KeyError(elem))
    end
    return res
end

function Base.getindex(si::SubIndex4{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    res = getindex(parent(si), elem, Val(false), Val(false))
    position = findfirst(isequal(res), parentindices(si))
    return isnothing(position) ? V(0x0) : V(position)
end
Base.getindex(si::SubIndex4{T, V, Colon}, elem::T, ::Val{false}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(false), Val(false))
function Base.getindex(
    si::SubIndex4{T, V, I},
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
    si::SubIndex4{T, V, I},
    elem::T,
    ::Val{false},
    ::Val{false},
) where {T, V, I <: AbstractArray{<:Integer}}
    res = getindex(parent(si), elem, Val(false), Val(false))
    return res > 0x0 ? si.revmapping[res, false, false] : res
end
function Base.getindex(si::SubIndex4{T, V, I}, elem::T, ::Val{false}, ::Val{false}) where {T, V, I <: AbstractRange}
    res = getindex(parent(si), elem, Val(false), Val(false))
    if res > 0x0
        i = findfirst(isequal(res), parentindices(si))
        return isnothing(i) ? 0x0 : i
    else
        return 0x0
    end
end
function Base.getindex(si::SubIndex4{T}, elem::T, ::Val{false}) where {T}
    res = getindex(si, elem, Val(false), Val(false))
    if res == 0x0
        throw(KeyError(elem))
    end
    return res
end
Base.@propagate_inbounds function Base.getindex(si::SubIndex4, i::Union{Integer, AbstractVector{<:Integer}})
    @boundscheck checkbounds(si, i)
    return parent(si)[Base.reindex((parentindices(si),), (i,))[1]]
end
Base.@propagate_inbounds function Base.getindex(
    si::SubIndex4{T, V, Colon},
    i::Union{Integer, AbstractVector{<:Integer}},
) where {T, V}
    @boundscheck checkbounds(si, i)
    return parent(si)[i]
end

Base.@propagate_inbounds function Base.setindex!(si::SubIndex4{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, Base.reindex((parentindices(si),), (i,))[1])
    return si
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex4{T, V, Colon}, newval::T, i::Integer) where {T, V}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, i)
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex4{T}, newval::T, oldval::T) where {T}
    oldidx = parent(si)[oldval, true]
    foldidx = findfirst(in(parentindices(si)), oldidx)
    if isnothing(foldidx)
        throw(KeyError(oldval))
    end
    parent(si)[oldidx[foldidx]] = newval
    return si
end
Base.@propagate_inbounds function Base.setindex!(
    si::SubIndex4{T, V, I},
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
    si::SubIndex4{T, V, I},
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
