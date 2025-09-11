# Robin Hood hash table, linear probing, linear search, backshift deletion

abstract type AbstractIndex6{T, V} <: AbstractVector{T} end
mutable struct Index6{T, V} <: AbstractIndex6{T, V}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}
    hashes::Vector{V}

    longestprobe::V
    initiallongestprobe::V
    deletions::V
    duplicates::V

    function Index6{T}(nelements::Integer) where {T}
        size = ceil(nelements / 0.9) # 0.9 load factor

        mintype = minimum_unsigned_type_for_n(size)
        cast_size = mintype(size)
        return new{T, mintype}(
            Vector{T}(undef, nelements),
            zeros(mintype, cast_size),
            zeros(mintype, cast_size),
            Vector{mintype}(undef, cast_size),
            0,
            0,
            0,
        )
    end
end

_length(idx::Index6) = length(idx.indices)

function _setindex!(idx::Index6{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition::V = 0x0
    k = position
    location = elemhash = celemhash = hash(elem) % _length(idx) + 0x1
    prevlongestprobe = idx.longestprobe
    is_duplicate = false
    @inbounds while k != 0x0
        probeposition += 0x1
        recordpos = idx.probepositions[location]
        if !is_duplicate && recordpos > 0 && isequal(idx.hashes[location], elemhash) && isequal(idx.vals[idx.indices[location]], elem)
            idx.duplicates += 1
            is_duplicate = true
        end
        if probeposition > recordpos
            k, idx.indices[location] = idx.indices[location], k
            celemhash, idx.hashes[location] = idx.hashes[location], celemhash
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

function _getindex(idx::Index6{T, V}, elem::T) where {T, V}
    ilength = _length(idx)
    if ilength > 0x0
        location::V = elemhash = hash(elem) % ilength + 0x1
        @inbounds for probeposition ∈ 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.hashes[location], elemhash) && isequal(idx.vals[pos], elem)
                return location
            elseif pos == 0x0
                return 0x0
            end
            location = location == ilength ? UInt(0x1) : location + 0x1
        end
    end
    return 0x0
end

function _getindex_array(idx::Index6{T, V}, elem::T) where {T, V}
    if allunique(idx)
        return [_getindex(idx, elem)]
    end
    locations = Vector{V}()
    ilength = _length(idx)
    if ilength > 0x0
        location = elemhash = hash(elem) % ilength + 0x1
        @inbounds for probeposition ∈ 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.hashes[pos], elemhash) && isequal(idx.vals[pos], elem)
                push!(locations, location)
            elseif pos == 0x0
                return locations
            end
            location = location == ilength ? UInt(0x1) : location + 0x1
        end
    end
    return locations
end

function _getindex_byposition(idx::Index6{T, V}, i::Integer) where {T, V}
    elem = idx.vals[i]
    ilength = _length(idx)
    if ilength > 0x0
        location::V = hash(elem) % ilength + 0x1
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

function _delete!(idx::Index6{T, V}, oldkeyindex::V, elem::T) where {T, V}
    lastidx = findnext(≤(0x1), idx.probepositions, oldkeyindex + 0x1)
    is_duplicate = false
    local hashpos::Union{Nothing, V}
    @inbounds if !isnothing(lastidx)
        lastidx -= 0x1

        if !allunique(idx)
            subhashes = @view idx.hashes[oldkeyindex:lastidx]
            hashpos = 0x1
            while !is_duplicate && !isnothing(hashpos)
                hashpos = findnext(isequal(subhashes[1]), subhashes, hashpos + 0x1)
                if !isnothing(hashpos) && isequal(idx.vals[idx.indices[oldkeyindex + hashpos - 0x1]], elem)
                    idx.duplicates -= 0x1
                    is_duplicate = true
                end
            end
        end

        idx.indices[oldkeyindex:(lastidx - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):lastidx]
        idx.probepositions[oldkeyindex:(lastidx - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):lastidx]) .- 0x1
        idx.hashes[oldkeyindex:(lastidx - 0x1)] .= @view idx.hashes[(oldkeyindex + 0x1):lastidx]
        idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
    else
        if oldkeyindex < _length(idx)
            if !allunique(idx)
                subhashes = @view idx.hashes[oldkeyindex:end]
                hashpos = 0x1
                while !is_duplicate && !isnothing(hashpos)
                    hashpos = findnext(isequal(subhashes[1]), subhashes, hashpos + 0x1)
                    if !isnothing(hashpos) && isequal(idx.vals[idx.indices[oldkeyindex + hashpos - 0x1]], elem)
                        idx.duplicates -= 0x1
                        is_duplicate = true
                    end
                end
            else
                is_duplicate = true
            end

            idx.indices[oldkeyindex:(end - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):end]
            idx.probepositions[oldkeyindex:(end - 0x1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):end]) .- 0x1
            idx.hashes[oldkeyindex:(end - 0x1)] .= @view idx.hashes[(oldkeyindex + 0x1):end]
        end
        if idx.probepositions[1] > 0x1
            lastidx = findnext(≤(0x1), idx.probepositions, 2)
            if isnothing(lastidx)
                lastidx = oldkeyindex
            end
            lastidx -= 0x1

            if !is_duplicate
                subhashes = @view idx.hashes[1:lastidx]
                hashpos = 1
                while !is_duplicate && !isnothing(hashpos)
                    hashpos = findnext(isequal(subhashes[1]), subhashes, hashpos + 1)
                    if !isnothing(hashpos) && isequal(idx.vals[idx.indices[hashpos]], elem)
                        idx.duplicates -= 0x1
                        is_duplicate = true
                    end
                end
            end

            idx.indices[end] = idx.indices[1]
            idx.probepositions[end] = idx.probepositions[1] - 0x1
            idx.hashes[end] = idx.hashes[1]

            idx.indices[0x1:(lastidx - 0x1)] .= @view idx.indices[0x2:lastidx]
            idx.probepositions[0x1:(lastidx - 0x1)] .= @view(idx.probepositions[0x2:lastidx]) .- 0x1
            idx.hashes[0x1:(lastidx - 0x1)] .= @view idx.hashes[0x2:lastidx]
            idx.indices[lastidx] = idx.probepositions[lastidx] = 0x0
        else
            idx.indices[end] = idx.probepositions[end] = 0x0
        end
    end

    idx.deletions += 0x1
end

function Index6(elems::AbstractVector{T}) where {T}
    idx = Index6{T}(length(elems))
    for (i, e) ∈ enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    idx.initiallongestprobe = idx.longestprobe
    return idx
end

Base.getindex(idx::AbstractIndex6{T}, elem::T) where {T} = getindex(idx, elem, Val(false))
Base.getindex(idx::AbstractIndex6{T}, elem::T, x::Bool) where {T} = getindex(idx, elem, Val(x))
Base.getindex(idx::AbstractIndex6{T}, elem::T, x::Bool, y::Bool) where {T} = getindex(idx, elem, Val(x), Val(y))
Base.getindex(idx::AbstractIndex6{T}, elems::AbstractVector{T}) where {T} = getindex(idx, elems, Val(false))
Base.getindex(idx::AbstractIndex6{T}, elems::AbstractVector{T}, x::Bool) where {T} = getindex(idx, elems, Val(x))
Base.getindex(idx::AbstractIndex6{T, V}, elems::AbstractVector{T}, x::Union{Val{true}, Val{false}}) where {T, V} =
    isempty(elems) ? V[] : reduce(vcat, (getindex(idx, elem, x) for elem ∈ elems))
Base.in(elem::T, idx::AbstractIndex6{T}) where {T} = getindex(idx, elem, Val(false), Val(false)) != 0x0

@inline Base.allunique(idx::Index6) = idx.duplicates == 0x0
Base.getindex(idx::Index6{T}, elem::T, ::Val{true}, ::Val{false}) where {T} =
    @inbounds idx.indices[_getindex_array(idx, elem)] # exceptions may be undesirable in high-performance scenarios
function Base.getindex(idx::Index6{T}, elem::T, ::Val{true}) where {T}
    d = _getindex_array(idx, elem)
    if length(d) == 0
        throw(KeyError(elem))
    end
    @inbounds return idx.indices[d]
end

function Base.getindex(idx::Index6{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    i = _getindex(idx, elem)
    return i == 0x0 ? V(0x0) : @inbounds idx.indices[i]
end
function Base.getindex(idx::Index6{T}, elem::T, ::Val{false}) where {T}
    i = getindex(idx, elem, Val(false), Val(false))
    if i == 0x0
        throw(KeyError(elem))
    end
    return i
end

Base.@propagate_inbounds function Base.getindex(idx::Index6, i::Integer)
    return idx.vals[i]
end

@inline function Base.setindex!(idx::Index6{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(idx.vals, i)
    oldkeyindex = _getindex_byposition(idx, i)
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex, idx.vals[i])
    _setindex!(idx, newval, validx)
    return idx
end

function Base.setindex!(idx::Index6{T}, newval::T, oldval::T) where {T}
    oldkeyindex = _getindex(idx, oldval)
    if oldkeyindex == 0x0
        throw(KeyError(oldval))
    end
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex, oldval)
    _setindex!(idx, newval, validx)
    return idx
end

Base.length(idx::Index6) = length(idx.vals)
Base.size(idx::Index6) = (length(idx),)
Base.values(idx::Index6) = idx.vals
Base.convert(::Type{<:Index6}, x::AbstractArray) = Index6(x)
Base.convert(::Type{<:Index6}, x::Index6) = x

struct SubIndex6{T, V, I} <: AbstractIndex6{T, V}
    parent::Index6{T, V}
    indices::I
    revmapping::Union{Nothing, Index6}
end
SubIndex6(idx::Index6{T, V}, indices::I) where {T, V, I} = SubIndex6{T, V, I}(idx, indices, nothing)

@inline function Base.view(idx::Index6, I::Union{AbstractRange, Colon})
    @boundscheck checkbounds(idx, I)
    return SubIndex6(idx, I)
end
@inline function Base.view(idx::Index6{T, V}, I::AbstractArray{<:Integer}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex6(idx, I, Index6(V.(I)))
end
@inline function Base.view(idx::Index6{T, V}, I::AbstractArray{Bool}) where {T, V}
    @boundscheck checkbounds(idx, I)
    return SubIndex6(idx, I, Index6(V.(findall(I))))
end
@inline function Base.view(idx::Index6, I::Integer)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(idx, I:I)
end
@inline function Base.view(idx::SubIndex6, I)
    @boundscheck checkbounds(idx, I)
    return @inbounds view(parent(idx), Base.reindex((parentindices(idx),), (I,))[1])
end

Base.copy(si::SubIndex6) = Index6(si)
Base.parent(si::SubIndex6) = si.parent
Base.parentindices(si::SubIndex6) = si.indices
Base.length(si::SubIndex6) = length(parentindices(si))
Base.length(si::SubIndex6{T, V, Colon}) where {T, V} = length(parent(si))
Base.length(si::SubIndex6{T, V, I}) where {T, V, I <: AbstractArray{Bool}} = length(si.revmapping)
Base.size(si::SubIndex6) = (length(si),)
Base.values(si::SubIndex6) = parent(si)[parentindices(si)]
function Base.getindex(si::SubIndex6{T}, elem::T, ::Val{true}, ::Val{false}) where {T}
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
Base.getindex(si::SubIndex6{T, V, Colon}, elem::T, ::Val{true}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(true), Val(false))
function Base.getindex(si::SubIndex6{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractArray{Bool}}
    res = getindex(parent(si), elem, Val(true), Val(false))
    res = res[parentindices(si)[res]]
    @inbounds for (i, r) ∈ enumerate(res)
        res[i] = si.revmapping[r, false, false]
    end

    return res
end
function Base.getindex(
    si::SubIndex6{T, V, I},
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
function Base.getindex(si::SubIndex6{T, V, I}, elem::T, ::Val{true}, ::Val{false}) where {T, V, I <: AbstractRange}
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
function Base.getindex(si::SubIndex6{T}, elem::T, ::Val{true}) where {T}
    res = getindex(si, elem, Val(true), Val(false))
    if length(res) == 0
        throw(KeyError(elem))
    end
    return res
end

function Base.getindex(si::SubIndex6{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    res = getindex(parent(si), elem, Val(false), Val(false))
    position = findfirst(isequal(res), parentindices(si))
    return isnothing(position) ? V(0x0) : V(position)
end
Base.getindex(si::SubIndex6{T, V, Colon}, elem::T, ::Val{false}, ::Val{false}) where {T, V} =
    getindex(parent(si), elem, Val(false), Val(false))
function Base.getindex(
    si::SubIndex6{T, V, I},
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
    si::SubIndex6{T, V, I},
    elem::T,
    ::Val{false},
    ::Val{false},
) where {T, V, I <: AbstractArray{<:Integer}}
    res = getindex(parent(si), elem, Val(false), Val(false))
    return res > 0x0 ? si.revmapping[res, false, false] : res
end
function Base.getindex(si::SubIndex6{T, V, I}, elem::T, ::Val{false}, ::Val{false}) where {T, V, I <: AbstractRange}
    res = getindex(parent(si), elem, Val(false), Val(false))
    if res > 0x0
        i = findfirst(isequal(res), parentindices(si))
        return isnothing(i) ? 0x0 : i
    else
        return 0x0
    end
end
function Base.getindex(si::SubIndex6{T}, elem::T, ::Val{false}) where {T}
    res = getindex(si, elem, Val(false), Val(false))
    if res == 0x0
        throw(KeyError(elem))
    end
    return res
end
Base.@propagate_inbounds function Base.getindex(si::SubIndex6, i::Union{Integer, AbstractVector{<:Integer}})
    @boundscheck checkbounds(si, i)
    return parent(si)[Base.reindex((parentindices(si),), (i,))[1]]
end
Base.@propagate_inbounds function Base.getindex(
    si::SubIndex6{T, V, Colon},
    i::Union{Integer, AbstractVector{<:Integer}},
) where {T, V}
    @boundscheck checkbounds(si, i)
    return parent(si)[i]
end

Base.@propagate_inbounds function Base.setindex!(si::SubIndex6{T}, newval::T, i::Integer) where {T}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, Base.reindex((parentindices(si),), (i,))[1])
    return si
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex6{T, V, Colon}, newval::T, i::Integer) where {T, V}
    @boundscheck checkbounds(si, i)
    setindex!(parent(si), newval, i)
end
Base.@propagate_inbounds function Base.setindex!(si::SubIndex6{T}, newval::T, oldval::T) where {T}
    oldidx = parent(si)[oldval, true]
    foldidx = findfirst(in(parentindices(si)), oldidx)
    if isnothing(foldidx)
        throw(KeyError(oldval))
    end
    parent(si)[oldidx[foldidx]] = newval
    return si
end
Base.@propagate_inbounds function Base.setindex!(
    si::SubIndex6{T, V, I},
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
    si::SubIndex6{T, V, I},
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
