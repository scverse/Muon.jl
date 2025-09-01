# a normal HDF5 dataset is not a subtype of AbstractArray, but then again HDF5.jl does not perform
# ahead-of-time bounds checking. checkbounds() and CartesianIndices are only defined for AbstractArrays
# and I don't want to also reimplement those
struct SparseDataset{G <: Group, T} <: AbstractArray{T, 2}
    group::G
    csr::Bool

    function SparseDataset(group::Group)
        csr = read_attribute(group, "encoding-type")[1:3] == "csr"
        if typeof(group) <: HDF5Group
            stype = HDF5Group
        else
            stype = typeof(group)
        end
        return new{stype, eltype(group["data"])}(group, csr)
    end
end

function Base.getproperty(dset::SparseDataset{G}, s::Symbol) where {G <: HDF5Group}
    if s === :id
        return getfield(dset, :group).id
    elseif s === :file
        return getfield(dset, :group).file
    elseif s === :xfer
        return Tuple(d.xfer for d ∈ getfield(dset, :group))
    else
        return getfield(dset, s)
    end
end

function Base.getproperty(dset::SparseDataset{G}, s::Symbol) where {G <: ZGroup}
    if s === :storage
        return getfield(dset, :group).storage
    elseif s === :path
        return getfield(dset, :group).path
    else
        return getfield(dset, s)
    end
end

HDF5.filename(dset::SparseDataset{G}) where {G <: HDF5Group} = HDF5.filename(dset.group)
HDF5.copy_object(
    src_obj::SparseDataset{T, G},
    dst_parent::HDF5Group,
    dst_path::AbstractString,
) where {T, G <: HDF5Group} = copy_object(src_obj.group, dst_parent, dst_path)
Base.isvalid(dset::SparseDataset{<:HDF5Group}) = isvalid(dset.group)
Base.isvalid(dset::SparseDataset{<:ZGroup}) = true

getcolptr(dset::SparseDataset) = dset.group["indptr"]
rowvals(dset::SparseDataset) = dset.group["indices"]
nonzeros(dset::SparseDataset) = dset.group["data"]

Base.isassigned(dset::Dataset, i) = i <= length(dset)

Base.ndims(dset::SparseDataset) = length(read_attribute(dset.group, "shape"))
Base.size(dset::SparseDataset) = Tuple(read_attribute(dset.group, "shape"))
Base.size(dset::SparseDataset, d::Integer) = size(dset)[d]
Base.length(dset::SparseDataset) = prod(size(dset))

rawsize(dset::SparseDataset) = dset.csr ? reverse(size(dset)) : size(dset)

Base.lastindex(dset::SparseDataset) = length(dset)
Base.lastindex(dset::SparseDataset, d::Int) = size(dset, d)
HDF5.datatype(dset::SparseDataset) = datatype(dset.group["data"])
Base.read(dset::SparseDataset) = read_matrix(dset.group)

Base.getindex(dset::SparseDataset, i::Integer) = getindex(dset, CartesianIndices(dset)[i])
Base.getindex(dset::SparseDataset, I::Tuple{Integer, Integer}) = getindex(dset, I[1], I[2])
function Base.getindex(dset::SparseDataset, i::Integer, j::Integer)
    @boundscheck checkbounds(dset, i, j)
    if dset.csr
        i, j = j, i
    end
    colptr = getcolptr(dset)
    rowval = rowvals(dset)
    c1, c2 = colptr[j] + 1, colptr[j + 1]
    if c2 ≥ c1
        rowidx = findfirst(x -> x == i - 1, rowval[c1:c2])
    else
        rowidx = nothing
    end
    return rowidx === nothing ? eltype(dset)(0) : nonzeros(dset)[c1 + rowidx - 1]
end
Base.getindex(dset::SparseDataset, ::Colon, ::Colon) = read(dset)
function Base.getindex(dset::SparseDataset, I::AbstractUnitRange, j::Integer)
    @boundscheck checkbounds(dset, I, j)
    return dset.csr ? _getindex(dset, j, I) : _getindex(dset, I, j)
end
function Base.getindex(dset::SparseDataset, i::Integer, J::AbstractUnitRange)
    @boundscheck checkbounds(dset, i, J)
    return dset.csr ? _getindex(dset, J, i) : _getindex(dset, i, J)
end

function Base.getindex(dset::SparseDataset, I::AbstractUnitRange, J::AbstractUnitRange)
    @boundscheck checkbounds(dset, I, J)
    sz = size(dset)
    if first(I) == 1 && last(I) == sz[1] && first(J) == 1 && last(J) == sz[2]
        return dset[:, :]
    end
    if dset.csr
        I, J = J, I
    end
    colptr = getcolptr(dset)
    rows = rowvals(dset)
    data = nonzeros(dset)

    newcols = Vector{eltype(colptr)}(undef, length(J) + 1)
    newcols[1] = 1
    newrows = Vector{eltype(rows)}()
    newdata = Vector{eltype(dset)}()
    for (nc, c) ∈ enumerate(J)
        c1, c2 = colptr[c] + 1, colptr[c + 1]
        if c2 ≥ c1
            currrows = rows[c1:c2] .+ convert(eltype(rows), 1)
            rowidx = findall(x -> x ∈ I, currrows)
            newcols[nc + 1] = newcols[nc] + length(rowidx)

            if length(rowidx) > 0
                currdata = data[c1:c2][rowidx]
                currrows = currrows[rowidx]
                sort!(rowidx, currdata, currrows)
                append!(newrows, currrows .- convert(eltype(newrows), first(I)) .+ convert(eltype(newrows), 1))
                append!(newdata, currdata)
            end
        else
            newcols[nc + 1] = newcols[nc]
        end
    end
    mat = SparseMatrixCSC(length(I), length(J), newcols, newrows, newdata)
    return dset.csr ? mat' : mat
end

function Base.getindex(dset::SparseDataset, i::Integer, ::Colon)
    @boundscheck checkbounds(dset, i, :)
    return dset.csr ? _getindex(dset, :, i) : _getindex(dset, i, :)
end

function Base.getindex(dset::SparseDataset, ::Colon, i::Integer)
    @boundscheck checkbounds(dset, :, i)
    return dset.csr ? _getindex(dset, i, :) : _getindex(dset, :, i)
end

Base.getindex(dset::SparseDataset, I::AbstractUnitRange, ::Colon) = dset[I, 1:size(dset, 2)]

Base.getindex(dset::SparseDataset, ::Colon, J::AbstractUnitRange) = dset[1:size(dset, 1), J]

function _getindex(dset, i::Integer, J::AbstractUnitRange)
    colptr = getcolptr(dset)
    c1, c2 = colptr[first(J)] + 1, colptr[last(J) + 1]
    rows = rowvals(dset)[c1:c2] .+ true
    rowidx = findall(x -> x == i, rows)

    if length(rowidx) == 0
        return SparseVector(length(J), Vector{eltype(rowidx)}(), Vector{eltype(dset)}())
    end

    data = nonzeros(dset)[c1:c2][rowidx]

    cols = Vector{eltype(rowidx)}(undef, length(data))
    i = 1
    for j ∈ J
        if colptr[j] < rowidx[i] + c1 - 1 <= colptr[j + 1]
            cols[i] = j - first(J) + 1
            if i == length(rowidx)
                break
            end
            i += 1
        end
    end

    sort!(rowidx, data)
    return SparseVector(length(J), cols, data)
end

function _getindex(dset, I::AbstractUnitRange, j::Integer)
    colptr = getcolptr(dset)
    c1, c2 = colptr[j] + 1, colptr[j + 1]
    rows = rowvals(dset)[c1:c2] .+ true
    rowidx = findall(x -> x ∈ I, rows)
    data = nonzeros(dset)[c1:c2][rowidx]

    sort!(rowidx, data)
    return SparseVector(length(I), rows[rowidx] .- first(I) .+ true, data)
end

function _getindex(dset, i::Integer, ::Colon)
    rows = read(rowvals(dset))
    nz = nonzeros(dset)
    rowidx = findall(x -> x + 1 == i, rows)
    data = [nz[i] for i ∈ rowidx]

    cols = Vector{eltype(rowidx)}(undef, length(data))
    colptr = getcolptr(dset)
    i = 1
    for (cstart, cend) ∈ zip(1:(length(colptr) - 1), 2:length(colptr))
        if colptr[cstart] < rowidx[i] <= colptr[cend]
            cols[i] = cstart
            if i == length(rowidx)
                break
            end
            i += 1
        end
    end

    return SparseVector(rawsize(dset)[2], cols, data)
end

function _getindex(dset, ::Colon, j::Integer)
    colptr = getcolptr(dset)
    rows = read(rowvals(dset))
    c1, c2 = colptr[j] + 1, colptr[j + 1]
    rowidx = rows[c1:c2] .+ true
    data = nonzeros(dset)[c1:c2]

    sort!(rowidx, data)
    return SparseVector(rawsize(dset)[1], rowidx, data)
end

function Base.setindex!(dset::SparseDataset, x::Number, i::Integer, j::Integer)
    @boundscheck checkbounds(dset, i, j)
    if dset.csr
        i, j = j, i
    end
    cols = getcolptr(dset)
    rows = rowvals(dset)

    c1, c2 = cols[j] + 1, cols[j + 1]
    rowidx = findfirst(x -> x + 1 == i, rows[c1:c2])
    if rowidx === nothing && x != 0
        throw(KeyError("changing the sparsity structure of a SparseDataset is not supported"))
    elseif x != 0
        nonzeros(dset)[c1 + rowidx - 1] = x
    end
end
function Base.setindex!(dset::SparseDataset, x::AbstractArray{<:Number, 2}, I::AbstractUnitRange, J::AbstractUnitRange)
    @boundscheck checkbounds(dset, I, J)
    length(x) == length(I) * length(J) || throw(
        DimensionMismatch("tried to assign $(length(x)) elements to destination of size $(length(I) * length(J))"),
    )
    if dset.csr
        I, J = J, I
        x = x'
    end
    linxidx = LinearIndices(x)
    cols = getcolptr(dset)
    rows = rowvals(dset)

    xidx = Int[]
    dsetidx = Int[]
    for (ic, c) ∈ enumerate(J)
        c1, c2 = cols[c] + 1, cols[c + 1]
        crows = rows[c1:c2] .+ true
        rowidx = findall(x -> x ∈ I, crows)
        xvals = x[I[I .∉ ((@view crows[rowidx]),)] .- first(I) .+ true, ic]
        if length(rowidx) != length(I) && any(xvals .!= 0)
            throw(KeyError("changing the sparsity structure of a SparseDataset is not supported"))
        end
        append!(xidx, linxidx[crows[rowidx] .- first(I) .+ true, ic])
        append!(dsetidx, c1 - 1 .+ rowidx)
    end
    # HDF5 doesn't support assignment using Arrays as indices, so we have to loop
    nz = nonzeros(dset)
    for (di, dx) ∈ zip(dsetidx, @view x[xidx])
        nz[di] = dx
    end
end

Base.eachindex(dset::SparseDataset) = CartesianIndices(size(dset))

function Base.show(io::IO, dset::SparseDataset{G}) where {G <: HDF5Group}
    if isvalid(dset)
        print(
            io,
            "Sparse ",
            join(size(dset), "×"),
            " HDF5 dataset: ",
            HDF5.name(dset.group),
            " (file: ",
            dset.file.filename,
            " xfer_mode: ",
            Tuple(x.id for x ∈ dset.xfer),
            ")",
        )
    else
        print(io, "Sparse HDF5 dataset: (invalid)")
    end
end

function Base.show(io::IO, dset::SparseDataset{G}) where {G <: ZGroup}
    if isvalid(dset)
        print(io, "Sparse ", join(size(dset), "×"), " Zarr dataset: ", dset.path, " (storage: ", dset.storage, ")")
    else
        print(io, "Sparse Zarr dataset: (invalid)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", dset::SparseDataset)
    if get(io, :compact, false)::Bool
        show(io, dset)
    else
        print(io, HDF5._tree_icon(HDF5.Dataset), " ", dset)
    end
end
