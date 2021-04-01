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

function Base.getproperty(dset::TransposedDataset, s::Symbol)
    if s === :id
        return getfield(dset, :dset).id
    elseif s === :file
        return getfield(dset, :dset).file
    elseif s === :xfer
        return getfield(dset, :dset).xfer
    else
        return getfield(dset, s)
    end
end

HDF5.filename(dset::TransposedDataset) = HDF5.filename(dset.dset)
HDF5.copy_object(
    src_obj::TransposedDataset,
    dst_parent::Union{HDF5.File, HDF5.Group},
    dst_path::AbstractString,
) = copy_object(src_obj.dset, dst_parent, dst_path)
HDF5.isvalid(dset::TransposedDataset) = isvalid(dset.dset)

Base.ndims(dset::TransposedDataset) = ndims(dset.dset)
Base.size(dset::TransposedDataset) = reverse(size(dset.dset))
Base.size(dset::TransposedDataset, d::Integer) = size(dset.dset)[ndims(dset.dset) - d + 1]

Base.lastindex(dset::TransposedDataset) = length(dset)
Base.lastindex(dset::TransposedDataset, d::Int) = size(dset, d)
HDF5.datatype(dset::TransposedDataset) = datatype(dset.dset)
Base.read(dset::TransposedDataset) = read_matrix(dset.dset)

Base.getindex(dset::TransposedDataset, i::Integer) = getindex(dset.dset, ndims(dset.dset) - d + 1)
function Base.getindex(dset::TransposedDataset, I::Vararg{<:Integer, N}) where N
    mat = getindex(dset.dset, reverse(I)...)
    return ndims(mat) == 1 ? mat : mat'
end
function Base.getindex(dset::TransposedDataset{T, N}, I...) where {T, N}
    emptydims = Vector{UInt8}()
    for (j, i) in enumerate(I)
        try
            if isempty(i)
                push!(emptydims, j)
            end
        catch e
            if !(e isa MethodError) # Colon doesn't support isempty
                rethrow(e)
            end
        end
    end
    if !isempty(emptydims)
        dims = collect(size(dset))
        dims[emptydims] .= UInt8(0)
        return Array{T, N}(undef, dims...)[I...]
    end
    mat = getindex(dset.dset, reverse(I)...)
    return ndims(mat) == 1 ? mat : mat'
end
Base.setindex!(dset::TransposedDataset, v, i::Int) = setindex!(dset.dset, v, ndims(dset.dset) - d + 1)
function Base.setindex!(dset::TransposedDataset, v, I::Vararg{Int, N}) where N
    if ndims(v) > 1
        v = copy(v')
    end
    setindex!(dset.dset, v, reverse(I)...)
end
function Base.setindex!(dset::TransposedDataset, v, I...)
    if ndims(v) > 1
        v = copy(v')
    end
    setindex!(dset.dset, v, reverse(I)...)
end

Base.eachindex(dset::TransposedDataset) = CartesianIndices(size(dset))

function Base.show(io::IO, dset::TransposedDataset)
    if isvalid(dset)
        print(io, "Transposed HDF5 dataset: ", HDF5.name(dset.dset), " (file: ", dset.file.filename, " xfer_mode: ", dset.xfer.id, ")")
    else
        print(io, "Transposed HDF5 datset: (invalid)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", dset::TransposedDataset)
    if get(io, :compact, false)::Bool
        show(io, dset)
    else
        print(io, HDF5._tree_icon(HDF5.Dataset), " ", dset)
    end
end
