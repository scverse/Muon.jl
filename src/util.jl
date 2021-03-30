function Base.sort!(idx::AbstractArray, vals::AbstractArray...)
    if !issorted(idx)
        ordering = sortperm(idx)
        permute!(idx, ordering)
        for v in vals
            permute!(v, ordering)
        end
    end
end

backed_matrix(obj::Union{HDF5.File, HDF5.Group}) = SparseDataset(obj)
backed_matrix(obj::HDF5.Dataset) = TransposedDataset(obj)

function hdf5_object_name(obj::Union{HDF5.File, HDF5.Group, HDF5.Dataset})
    name = HDF5.name(obj)
    return name[(last(findlast("/", name)) + 1):end]
end

Base.getindex(adata::Union{AnnData, MuData}, i::Integer, J::Union{AbstractUnitRange, Colon, Vector{<:Integer}}) =
    getindex(adata, i:i, J)
Base.getindex(adata::Union{AnnData, MuData}, I::Union{AbstractUnitRange, Colon, Vector{<:Integer}}, j::Integer) =
    getindex(adata, I, j:j)
Base.getindex(adata::Union{AnnData, MuData}, i::Integer, j::Integer) = getindex(adata, i:i, j:j)
