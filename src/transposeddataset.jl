# Julia stores arrays in column-major order, whereas NumPy stores arrays in row-major order.
# Both languages save arrays to HDF5 following their memory layout, so arrays that were saved
# using h5py are transposed in Julia and vice versa. To guarantee compatibility for backed files,
# we need an additional translation layer that transposes array accesses

struct TransposedDataset{T, N} <: AbstractArray{T, N}
    dset::HDF5.Dataset

    function TransposedDataset(dset::HDF5.Dataset)
        return new{eltype(dset), ndims(dset)}(dset)
    end
end

HDF5.filename(dset::TransposedDataset) = HDF5.filename(dset.dset)
HDF5.copy_object(
    src_obj::TransposedDataset,
    dst_parent::Union{HDF5.File, HDF5.Group},
    dst_path::AbstractString,
) = copy_object(src_obj.dset, dst_parent, dst_path)

Base.ndims(dset::TransposedDataset) = ndims(dset.dset)
Base.size(dset::TransposedDataset) = reverse(size(dset.dset))
Base.size(dset::TransposedDataset, d::Integer) = size(dset.dset)[ndims(dset.dset) - d + 1]

Base.lastindex(dset::TransposedDataset) = length(dset)
Base.lastindex(dset::TransposedDataset, d::Int) = size(dset, d)
HDF5.datatype(dset::TransposedDataset) = datatype(dset.dset)
Base.read(dset::TransposedDataset) = read_matrix(dset.dset)

Base.getindex(dset::TransposedDataset, i::Integer) = getindex(dset.dset, ndims(dset.dset) - d + 1)
Base.getindex(dset::TransposedDataset, I::Vararg{<:Integer, N}) where N = getindex(dset.dset, reverse(I)...)
Base.getindex(dset::TransposedDataset, I...) = getindex(dset.dset, reverse(I)...)
Base.setindex!(dset::TransposedDataset, v, i::Int) = setindex!(dset.dset, v, ndims(dset.dset) - d + 1)
Base.setindex!(dset::TransposedDataset, v, I::Vararg{Int, N}) where N = setindex!(dset.dset, reverse(I)...)
Base.setindex!(dset::TransposedDataset, v, I...) = setIndex!(dset.dset, reverse(I)...)

Base.eachindex(dset::TransposedDataset) = CartesianIndices(size(dset))
