# Robin Hood hash table, linear probing, linear search, backshift deletion

mutable struct Index{T, V} <: AbstractVector{T}
    vals::Vector{T}
    indices::Vector{V}
    probepositions::Vector{V}

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

        return new{T, mintype}(Vector{T}(undef, nelements), zeros(mintype, mintype(size)), zeros(mintype, mintype(size)), 0)
    end
end

_length(idx::Index) = length(idx.indices)

function _setindex!(idx::Index{T, V}, elem::T, position::Unsigned) where {T, V}
    idx.vals[position] = elem
    probeposition = 0x0
    k = position
    location = hash(elem) % _length(idx)
    @inbounds while k != 0
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
end

function _getindex(idx::Index{T}, elem::T) where T
    location = hash(elem) % _length(idx) + 1
    while true
        pos = idx.indices[location]
        if pos > 0 && isequal(idx.vals[pos], elem)
            return location
        elseif pos == 0
            return 0x0
        end
        location += 0x1
        if location > _length(idx)
            location = location % _length(idx)
        end
    end
end

function _delete!(idx::Index, oldkeyindex::Integer)
    lastidx = findnext(x -> x <= 0x1, idx.probepositions, oldkeyindex + 0x1)
    @inbounds if !isnothing(lastidx)
        idx.indices[oldkeyindex:(lastidx - 2)] .= @view idx.indices[(oldkeyindex + 0x1):(lastidx - 1)]
        idx.probepositions[oldkeyindex:(lastidx - 2)] .= @view(idx.probepositions[(oldkeyindex + 0x1):(lastidx - 1)]) .- 0x1
        idx.indices[lastidx - 1] = idx.probepositions[lastidx - 1] = 0x0
    else
        if oldkeyindex < _length(idx)
            idx.indices[oldkeyindex:(end - 1)] .= @view idx.indices[(oldkeyindex + 0x1):end]
            idx.probepositions[oldkeyindex:(end - 1)] .= @view(idx.probepositions[(oldkeyindex + 0x1):end]) .- 1
        end
        if idx.probepositions[1] > 0x1
            idx.indices[end] = idx.indices[1]
            idx.probepositions[end] = idx.probepositions[1] - 0x1

            lastidx = findnext(x -> x <= 0x1, idx.probepositions, 2)
            if isnothing(lastidx)
                lastidx = oldkeyindex
            end
            idx.indices[1:(lastidx - 2)] .= @view idx.indices[2:(lastidx - 1)]
            idx.probepositions[1:(lastidx - 2)] .= @view(idx.probepositions[2:(lastidx - 1)]) .- 0x1
            idx.indices[lastidx - 1] = idx.probepositions[lastidx - 1] = 0x0
        else
            idx.indices[end] = idx.probepositions[end] = 0x0
        end
    end
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
    validx = idx.indices[oldkeyindex]
    _delete!(idx, oldkeyindex)
    _setindex!(idx, newval, validx)
end

Base.in(elem::T, idx::Index{T}) where T = _getindex(idx, elem) != 0x0

Base.length(idx::Index) = length(idx.vals)
Base.size(idx::Index) = (length(idx),)
