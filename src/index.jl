# Robin Hood hash table, linear probing, linear search, backshift deletion

mutable struct Index{T, V} <: AbstractVector{T}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}

    longestprobe::V
    initiallongestprobe::V
    deletions::UInt16

    function Index{T}(nelements::Integer) where {T}
        size = ceil(nelements / 0.9) # 0.9 load factor

        mintype = minimum_unsigned_type_for_n(nelements)
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
    probeposition = 0x0
    k = position
    location = hash(elem) % _length(idx)
    prevlongestprobe = idx.longestprobe
    @inbounds while k != 0x0
        probeposition += 0x1
        location = location == _length(idx) ? UInt64(0x1) : location + 0x1
        recordpos = idx.probepositions[location]
        if probeposition > recordpos
            k, idx.indices[location] = idx.indices[location], k
            idx.probepositions[location] = probeposition
            idx.longestprobe = max(idx.longestprobe, probeposition)
            probeposition = recordpos
        end
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
        for probeposition in 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.vals[pos], elem)
                return location
            elseif pos == 0x0
                return 0x0
            end
            location += 0x1
            if location > _length(idx)
                location = location % ilength
            end
        end
    end
    return 0x0
end

function _getindex_array(idx::Index{T, V}, elem::T) where {T, V}
    locations = Vector{V}()
    ilength = _length(idx)
    if ilength > 0x0
        location = hash(elem) % ilength + 0x1
        for probeposition in 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && isequal(idx.vals[pos], elem)
                push!(locations, location)
            elseif pos == 0x0
                return locations
            end
            location += 0x1
            if location > ilength
                location = location % ilength
            end
        end
    end
    return locations
end

function _getindex_byposition(idx::Index{T}, i::Integer) where {T}
    elem = idx.vals[i]
    ilength = _length(idx)
    if ilength > 0x0
        location = hash(elem) % ilength + 0x1
        for probeposition in 0x1:(idx.longestprobe)
            pos = idx.indices[location]
            if pos > 0x0 && pos == i
                return location
            elseif pos == 0x0
                break
            end
            location += 0x1
            if location > ilength
                location = location % ilength
            end
        end
    end
    throw(ErrorException("Element not found. This should never happen."))
end

function _delete!(idx::Index, oldkeyindex::Integer)
    lastidx = findnext(x -> x <= 0x1, idx.probepositions, oldkeyindex + 0x1)
    @inbounds if !isnothing(lastidx)
        idx.indices[oldkeyindex:(lastidx - 0x2)] .=
            @view idx.indices[(oldkeyindex + 0x1):(lastidx - 0x1)]
        idx.probepositions[oldkeyindex:(lastidx - 0x2)] .=
            @view(idx.probepositions[(oldkeyindex + 0x1):(lastidx - 0x1)]) .- 0x1
        idx.indices[lastidx - 0x1] = idx.probepositions[lastidx - 0x1] = 0x0
    else
        if oldkeyindex < _length(idx)
            idx.indices[oldkeyindex:(end - 0x1)] .= @view idx.indices[(oldkeyindex + 0x1):end]
            idx.probepositions[oldkeyindex:(end - 0x1)] .=
                @view(idx.probepositions[(oldkeyindex + 0x1):end]) .- 0x1
        end
        if idx.probepositions[1] > 0x1
            idx.indices[end] = idx.indices[1]
            idx.probepositions[end] = idx.probepositions[1] - 0x1

            lastidx = findnext(x -> x <= 0x1, idx.probepositions, 2)
            if isnothing(lastidx)
                lastidx = oldkeyindex
            end
            idx.indices[0x1:(lastidx - 0x2)] .= @view idx.indices[0x2:(lastidx - 0x1)]
            idx.probepositions[0x1:(lastidx - 0x2)] .=
                @view(idx.probepositions[0x2:(lastidx - 0x1)]) .- 0x1
            idx.indices[lastidx - 0x1] = idx.probepositions[lastidx - 0x1] = 0x0
        else
            idx.indices[end] = idx.probepositions[end] = 0x0
        end
    end

    idx.deletions += 0x1
end

function Index(elems::AbstractVector{T}) where {T}
    idx = Index{T}(length(elems))
    for (i, e) in enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    idx.initiallongestprobe = idx.longestprobe
    return idx
end

Base.getindex(idx::Index{T}, elem::T) where {T} = getindex(idx, elem, Val(false))
Base.getindex(idx::Index{T}, elem::T, x::Bool) where {T} = getindex(idx, elem, Val(x))
Base.getindex(idx::Index{T}, elem::T, x::Bool, y::Bool) where {T} = getindex(idx, elem, Val(x), Val(y))
Base.getindex(idx::Index{T}, elems::AbstractVector{T}) where {T} = getindex(idx, elems, Val(false))
Base.getindex(idx::Index{T}, elems::AbstractVector{T}, x::Bool) where {T} =
    getindex(idx, elems, Val(x))

Base.getindex(idx::Index{T}, elem::T, ::Val{true}, ::Val{false}) where {T} =
    idx.indices[_getindex_array(idx, elem)] # exceptions may be undesirable in high-performance scenarios
function Base.getindex(idx::Index{T}, elem::T, ::Val{true}) where {T}
    d = _getindex_array(idx, elem)
    if length(d) == 0
        throw(KeyError(elem))
    end
    @inbounds return idx.indices[d]
end

function Base.getindex(idx::Index{T, V}, elem::T, ::Val{false}, ::Val{false}) where {T, V}
    i = _getindex(idx, elem)
    @inbounds return i == 0x0 ? V[] : idx.indices[i]
end
function Base.getindex(idx::Index{T}, elem::T, ::Val{false}) where {T}
    i = _getindex(idx, elem)
    if i == 0x0
        throw(KeyError(elem))
    end
    @inbounds return idx.indices[i]
end

function Base.getindex(idx::Index, i::Integer)
    return idx.vals[i]
end

Base.getindex(idx::Index{T}, elems::AbstractVector{T}, x::Union{Val{true}, Val{false}}) where {T} =
    reduce(vcat, (getindex(idx, elem, x) for elem in elems))

function Base.setindex!(idx::Index{T}, newval::T, i::Integer) where {T}
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

Base.in(elem::T, idx::Index{T}) where {T} = _getindex(idx, elem) != 0x0

Base.length(idx::Index) = length(idx.vals)
Base.size(idx::Index) = (length(idx),)
