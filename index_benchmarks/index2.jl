# Robin Hood hash table, linear probing, smart search, tombstone deletion

mutable struct Index2{T, V} <: AbstractVector{T}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}
    tombstones::BitVector

    totalcost::UInt64
    longestprobe::V

    function Index2{T}(nelements::Integer) where T
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

_length(idx::Index2) = length(idx.indices)

function _setindex!(idx::Index2{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition = 0x0
    k = position
    tombstone = false
    location = hash(elem) % _length(idx)
    @inbounds while k != 0 && !tombstone
        probeposition += 0x1
        location = location == _length(idx) ? UInt64(0x1) : location + 0x1
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

function _getindex(idx::Index2{T}, elem::T) where T
    meanposition = trunc(UInt64, idx.totalcost / length(idx.vals))
    downposition = meanposition
    upposition = downposition + 0x1
    location = hash(elem) % _length(idx)
    @inbounds while downposition >= 0x1 && upposition <= idx.longestprobe
        downlocation = location + downposition
        downlocation > _length(idx) && (downlocation = downlocation % _length(idx))
        pos = idx.indices[downlocation]
        if pos > 0x0 && !idx.tombstones[downlocation] && idx.vals[pos] == elem
            return downlocation
        end
        uplocation = location + upposition
        uplocation > _length(idx) && (uplocation = uplocation % _length(idx))
        pos = idx.indices[uplocation]
        if pos > 0x0 && !idx.tombstones[uplocation]&& idx.vals[pos] == elem
            return uplocation
        end
        downposition -= 0x1
        upposition += 0x1
    end
    @inbounds while downposition >= 0x1
        downlocation = location + downposition
        downlocation > _length(idx) && (downlocation = downlocation % _length(idx))
        pos = idx.indices[downlocation]
        if pos > 0x0 && !idx.tombstones[downlocation] && idx.vals[pos] == elem
            return downlocation
        end
        downposition -= 0x1
    end
    @inbounds while upposition <= idx.longestprobe
        uplocation = location + upposition
        uplocation > _length(idx) && (uplocation = uplocation % _length(idx))
        pos = idx.indices[uplocation]
        if pos > 0x0 && !idx.tombstones[uplocation] && idx.vals[pos] == elem
            return uplocation
        end
        upposition += 0x1
    end
    return 0x0
end

function Index2(elems::AbstractVector{T}) where T
    idx = Index2{T}(length(elems))
    for (i, e) in enumerate(elems)
        _setindex!(idx, e, UInt(i))
    end
    return idx
end

function Base.getindex(idx::Index2{T}, elem::T) where T
    i = _getindex(idx, elem)
    if i == 0x0
        throw(KeyError(elem))
    end
    return idx.indices[i]
end

function Base.getindex(idx::Index2, i::Integer)
    return idx.vals[i]
end

function Base.setindex!(idx::Index2{T}, elem::T, i::Integer) where T
    @boundscheck checkbounds(idx.vals, i)
    setindex!(idx, elem, idx.vals[i])
end

function Base.setindex!(idx::Index2{T}, newval::T, oldval::T) where T
    oldkeyindex = _getindex(idx, oldval)
    if oldkeyindex == 0x0
        throw(KeyError(oldval))
    end
    idx.tombstones[oldkeyindex] = true
    idx.totalcost -= idx.probepositions[oldkeyindex]
    _setindex!(idx, newval, idx.indices[oldkeyindex])
end

Base.length(idx::Index2) = length(idx.vals)
Base.size(idx::Index2) = (length(idx),)
