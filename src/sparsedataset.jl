struct SparseDataset
    group::HDF5.Group
    csr::Bool

    function SparseDataset(group::HDF5.Group)
        csr = read_attribute(group, "encoding-type")[1:3] == "csr"
        return new(group, csr)
    end
end

function Base.getproperty(dset::SparseDataset, s::Symbol)
    if s === :id
        return getfield(dset, :group).id
    elseif s === :file
        return getfield(dset, :group).file
    elseif s === :xfer
        return getfield(dset, :group).xfer
    else
        return getfield(dset, s)
    end
end

struct DatasetOffsetWrapper{T<:Integer, N} <: AbstractArray{T, N}
    dset::HDF5.Dataset
    function DatasetOffsetWrapper(dset::HDF5.Dataset)
        return new{eltype(dset), ndims(dset)}(dset)
    end
end

Base.size(A::DatasetOffsetWrapper) = size(A.dset)
(Base.getindex(A::DatasetOffsetWrapper{T, N}, i::Int)::T) where {T, N} = getindex(A.dset, i) + T(1)
(Base.getindex(A::DatasetOffsetWrapper{T, N}, I::UnitRange{<:Integer})::Array{T, 1}) where {T, N} = getindex(A.dset, I) .+ T(1)
# apparently some console printing functions always use 2 indices, even for a 1d array
function Base.getindex(A::DatasetOffsetWrapper{T, N}, I::Integer...)::T where {T, N}
    @boundscheck if length(I) > N && any(I[(N+1):end] .> 1)
        throw(BoundsError(A, I))
    end
    return getindex(A.dset, I[1:N]...) .+ T(1)
end
Base.setindex!(A::DatasetOffsetWrapper{T, N}, v::Integer, i::Int) where {T, N} = setindex!(A.dset, v .- T(1), i)
Base.setindex!(A::DatasetOffsetWrapper{T, N}, v::Integer, I...) where {T, N} = setindex!(A.dset, v .- T(1), I...)
Base.length(A::DatasetOffsetWrapper) = length(A.dset)
Base.axes(A::DatasetOffsetWrapper) = axes(A.dset)
Base.eachindex(A::DatasetOffsetWrapper) = eachindex(A.dset)
function Base.isassigned(A::DatasetOffsetWrapper{T, N}, I::Integer...) where {T, N}
    @boundscheck if length(I) > N && any(I[(N+1):end] .> 1)
        return false
    else
        @boundscheck begin
            if length(I) < N
                throw(BoundsError(A, I))
            end
            sz = size(A)
            for (i, d) in zip(I, sz)
                if i > d
                    return false
                end
            end
        end
        return true
    end
end



struct BackedSparseMatrixCSC{Tv<:Number, Ti<:Integer} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
    m::Int
    n::Int
    colptr::DatasetOffsetWrapper{Ti, 1}
    rowval::DatasetOffsetWrapper{Ti, 1}
    nzval::HDF5.Dataset

    function BackedSparseMatrixCSC(m::Int, n::Int, colptr::HDF5.Dataset, rowval::HDF5.Dataset, nzval::HDF5.Dataset)
        cols = DatasetOffsetWrapper(colptr)
        rows = DatasetOffsetWrapper(rowval)
        @assert eltype(cols) ===  eltype(rows)
        return new{eltype(nzval), eltype(cols)}(m, n, cols, rows, nzval)
    end
end
Base.size(S::BackedSparseMatrixCSC) = (S.m, S.n)
SparseArrays.getcolptr(S::BackedSparseMatrixCSC) = S.colptr
SparseArrays.rowvals(S::BackedSparseMatrixCSC) = S.rowval
SparseArrays.nonzeros(S::BackedSparseMatrixCSC) = S.nzval

Base.isassigned(dset::HDF5.Dataset, i) = i <= length(dset)
Base.getindex(dset::HDF5.Dataset, I::UnitRange{<:Integer}) = isempty(I) ? Array{eltype(dset), 1}() : getindex(dset, I)

function to_backed_mat(dset::SparseDataset)
    m, n = size(dset)
    if dset.csr
        m, n = n, m
    end
    return BackedSparseMatrixCSC(m, n, dset.group["indptr"], dset.group["indices"], dset.group["data"])
end

Base.ndims(dset::SparseDataset) = length(attributes(dset.group)["shape"])
Base.size(dset::SparseDataset) = Tuple(read_attribute(dset.group, "shape"))
Base.size(dset::SparseDataset, d::Integer) = size(dset)[d]
Base.length(dset::SparseDataset) = prod(size(dset))

Base.lastindex(dset::SparseDataset) = length(dset)
Base.lastindex(dset::SparseDataset, d::Int) = size(dset, d)
HDF5.datatype(dset::SparseDataset) = datatype(dset.group["data"])
Base.read(dset::SparseDataset) = read_matrix(dset.group)

Base.getindex(dset::SparseDataset, I::Tuple{Integer, Integer}) = getindex(dset, I[1], I[2])
function Base.getindex(dset::SparseDataset, i0::Integer, i1::Integer)
    mtx = to_backed_mat(dset)
    return dset.csr ? mtx[i1, i0] : mtx[i0, i1]
end
Base.getindex(dset::SparseDataset, ::Colon, ::Colon) = read(dset)
function Base.getindex(dset::SparseDataset, i, j)
    mtx = to_backed_mat(dset)
    return dset.csr ? mtx[j, i]' : mtx[i, j]
end

Base.eachindex(dset::SparseDataset) = CartesianIndices(size(dset))

Base.axes(dset::SparseDataset) = map(Base.OneTo, size(dset))

function Base.setindex!(dset::SparseDataset, X::Union{<:Number, Array{<:Number}}, I::HDF5.IndexType, J::HDF5.IndexType)
    mtx = to_backed_mat(dset)
    dset.csr ? setindex!(mtx, X, J, I) : setindex!(mtx, X, I, J)
end
