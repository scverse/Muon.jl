function Base.sort!(idx::AbstractArray, vals::AbstractArray...)
    if !issorted(idx)
        ordering = sortperm(idx)
        permute!(idx, ordering)
        for v in vals
            permute!(v, ordering)
        end
    end
end

backed_matrix(obj::Union{HDF5.File, HDF5.Group}) =
    return obj isa HDF5.Dataset ? TransposedDataset(obj) : SparseDataset(obj)

function hdf5_object_name(obj::Union{HDF5.File, HDF5.Group, HDF5.Dataset})
    name = HDF5.name(obj)
    return name[(last(findlast("/", name)) + 1):end]
end
