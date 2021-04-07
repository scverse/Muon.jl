# Robin Hood hash table, double hashing, smart search, tombstone deletion

mutable struct Index{T, V} <: AbstractVector{T}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}
    tombstones::BitVector

    totalcost::UInt64
    longestprobe::V

    function Index{T}(nelements::Integer) where T
        size = ceil(nelements / 0.9) # 0.9 load factor

        local mintype::Type
        for type in [UInt8, UInt16, UInt32, UInt64, UInt128]
            mval = typemax(type)
            if mval >= size
                mintype = type
                break
            end
        end

        return new{T, mintype}(Vector{T}(undef, nelements), zeros(mintype, mintype(size)), zeros(mintype, mintype(size)), falses(mintype(size)), 0, 0)
    end
end

_length(idx::Index) = length(idx.indices)

_H(elem, i::UInt64, size::Integer) = (i == 0 ? hash(elem) : hash(elem, i)) % size + 0x1

function _setindex!(idx::Index{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition = 0x0
    k = position
    tombstone = false
    @inbounds while k != 0 && !tombstone
        probeposition += 0x1
        elem = idx.vals[k]
        location = _H(elem, UInt64(probeposition), _length(idx))
        idx.totalcost += 0x1
        recordpos = idx.probepositions[location]
        if probeposition > recordpos
            k, idx.indices[location] = idx.indices[location], k
            tombstone, idx.tombstones[location] = idx.tombstones[location], false
            idx.probepositions[location] = probeposition
            idx.longestprobe = max(idx.longestprobe, probeposition)
            probeposition = recordpos
        end
    end
end

function _getindex(idx::Index{T}, elem::T) where T
    meanposition = trunc(UInt64, idx.totalcost / length(idx.vals))
    downposition = meanposition
    upposition = downposition + 0x1
    @inbounds while downposition >= 0x1 && upposition <= idx.longestprobe
        downlocation = _H(elem, downposition, _length(idx))
        pos = idx.indices[downlocation]
        if pos > 0x0 && !idx.tombstones[downlocation] && idx.vals[pos] == elem
            return downlocation
        end
        uplocation = _H(elem, upposition, _length(idx))
        pos = idx.indices[uplocation]
        if pos > 0x0  && !idx.tombstones[uplocation]&& idx.vals[pos] == elem
            return uplocation
        end
        downposition -= 0x1
        upposition += 0x1
    end
    @inbounds while downposition >= 0x1
        downlocation = _H(elem, downposition, _length(idx))
        pos = idx.indices[downlocation]
        if pos > 0x0 && !idx.tombstones[downlocation] && idx.vals[pos] == elem
            return downlocation
        end
        downposition -= 0x1
    end
    @inbounds while upposition <= idx.longestprobe
        uplocation = _H(elem, upposition, _length(idx))
        pos = idx.indices[uplocation]
        if pos > 0x0 && !idx.tombstones[uplocation] && idx.vals[pos] == elem
            return uplocation
        end
        upposition += 0x1
    end
    return 0x0
end

function Index(elems::AbstractVector{T}) where T
    idx = Index{T}(length(elems))
    for (i, e) in enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    return idx
end

function Base.getindex(idx::Index{T}, elem::T) where T
    i = _getindex(idx, elem)
    if i == 0x0
        throw(KeyError(elem))
    end
    return idx.indices[i]
end

function Base.getindex(idx::Index, i::Integer)
    return idx.vals[i]
end

function Base.setindex!(idx::Index{T}, elem::T, i::Integer) where T
    @boundscheck checkbounds(idx.vals, i)
    setindex!(idx, elem, idx.vals[i])
end

function Base.setindex!(idx::Index{T}, newval::T, oldval::T) where T
    oldkeyindex = _getindex(idx, oldval)
    if oldkeyindex == 0x0
        throw(KeyError(oldval))
    end
    idx.tombstones[oldkeyindex] = true
    idx.totalcost -= idx.probepositions[oldkeyindex]
    _setindex!(idx, newval, idx.indices[oldkeyindex])
end

Base.length(idx::Index) = length(idx.vals)
Base.size(idx::Index) = (length(idx),)
