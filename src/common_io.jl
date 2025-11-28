function read_dataframe(tablegroup::Group; separate_index=true, kwargs...)
    columns = read_attribute(tablegroup, "column-order")

    if separate_index && has_attribute(tablegroup, "_index")
        indexdsetname = read_attribute(tablegroup, "_index")
        rownames = read(tablegroup[indexdsetname])
    else
        rownames = nothing
    end

    df = DataFrame()

    for col ∈ columns
        column = read_matrix(tablegroup[col])
        if sum(size(column) .> 1) > 1
            @warn "column $col has more than 1 dimension for data frame $(HDF5.name(tablegroup)), skipping"
        end
        df[!, col] = column
    end

    return df, rownames
end

function read_matrix(f::Dataset; kwargs...)
    mat = read(f)

    if is_compound(f)
        return StructArray(mat)
    end

    if is_bool(f)
        mat = BitArray(mat)
    end

    if ndims(f) == 0
        return mat[]
    end
    if ndims(f) > 1
        mat = PermutedDimsArray(mat, ndims(mat):-1:1) # transpose for h5py compatibility
    end
    if has_attribute(f, "categories")
        categories = f[read_attribute(f, "categories")]
        ordered = has_attribute(categories, "ordered") && read_attribute(categories, "ordered") == true
        cats = read(categories)
        mat .+= 1
        mat = if ordered
            compress(
                CategoricalArray{eltype(cats), ndims(mat)}(
                    mat,
                    CategoricalPool{eltype(cats), eltype(mat)}(cats, ordered),
                ),
            )
        else
            PooledArray(PooledArrays.RefArray(mat), Dict{eltype(cats), eltype(mat)}(v => i for (i, v) ∈ enumerate(cats)))
        end
    end
    return mat
end

function read_matrix(f::Group; kwargs...)
    enctype = read_attribute(f, "encoding-type")

    if enctype == "csc_matrix" || enctype == "csr_matrix"
        shape = read_attribute(f, "shape")
        iscsr = enctype[1:3] == "csr"

        indptr = read(f, "indptr")
        indices = read(f, "indices")
        data = read(f, "data")

        indptr .+= 1
        indices .+= 1

        # the row indices in every column need to be sorted
        @views for (colstart, colend) ∈ zip(indptr[1:(end - 1)], indptr[2:end])
            sort!(indices[colstart:(colend - 1)], data[colstart:(colend - 1)])
        end

        if iscsr
            reverse!(shape)
        end
        mat = SparseMatrixCSC(shape..., indptr, indices, data)
        return iscsr ? mat' : mat
    elseif enctype == "categorical"
        ordered = read_attribute(f, "ordered") > 0
        categories = read(f, "categories")
        codes = read(f, "codes") .+ true

        T = any(iszero, codes) ? Union{Missing, eltype(categories)} : eltype(categories)
        mat = if ordered
            CategoricalVector{T}(undef, length(codes); levels=categories, ordered=ordered)
            copy!(mat.refs, codes)
        else
            PooledArray(PooledArrays.RefArray(codes), Dict{T, eltype(codes)}(v => i for (i, v) ∈ enumerate(categories)))
        end
        return mat
    else
        error("unknown encoding $enctype")
    end
end

function read_nullable_integer_array(f::Group, kwargs...)
    mask = read_matrix(f["mask"])
    values = read(f["values"])
    return [m ? missing : v for (m, v) ∈ zip(mask, values)]
end

function read_nullable_boolean_array(f::Group, kwargs...)
    mask = read_matrix(f["mask"])
    values = read(f["values"])
    return [m ? missing : v!=0 for (m, v) ∈ zip(mask, values)]
end

function read_dict_of_matrices(f::Group; kwargs...)
    return Dict{String, AbstractArray{<:Number}}(key => read_matrix(f[key]; kwargs...) for key ∈ keys(f))
end

function read_auto(f::Dataset; kwargs...)
    if has_attribute(f, "encoding-type")
        enctype = read_attribute(f, "encoding-type")
        if endswith(enctype, "scalar") || enctype == "string"
            return read_scalar(f), nothing
        elseif enctype == "null"
            return nothing, nothing
        end
    end
    return read_matrix(f; kwargs...), nothing
end
function read_auto(f::Group; kwargs...)
    if has_attribute(f, "encoding-type")
        enctype = read_attribute(f, "encoding-type")
        if enctype == "dataframe"
            return read_dataframe(f; kwargs...)
        elseif endswith(enctype, "matrix") || enctype == "categorical"
            return read_matrix(f; kwargs), nothing
        elseif enctype == "dict"
            return read_dict_of_mixed(f; kwargs...), nothing
        elseif enctype == "nullable-integer"
            return read_nullable_integer_array(f; kwargs...), nothing
        elseif enctype == "nullable-boolean"
            return read_nullable_boolean_array(f; kwargs...), nothing
        else
            error("unknown encoding $enctype")
        end
    else
        return read_dict_of_mixed(f; kwargs...), nothing
    end
end

function read_dict_of_mixed(f::Group; kwargs...)
    ret = Dict{
        String,
        Union{
            DataFrame,
            StructArray,
            <:AbstractArray{<:Number},
            <:AbstractArray{<:AbstractString},
            <:CategoricalArray{<:Number},
            <:CategoricalArray{<:AbstractString},
            <:AbstractArray{<:Union{Integer, Missing}},
            <:AbstractArray{Union{Bool, Missing}},
            <:AbstractString,
            <:Number,
            Nothing,
            Dict,
        },
    }()
    for k ∈ keys(f)
        ret[k] = read_auto(f[k]; kwargs...)[1] # assume data frames are properly aligned, so we can discard rownames
    end
    return ret
end

function write_attr(parent::Group, name::AbstractString, data; kwargs...)
    if haskey(parent, name)
        delete_object(parent, name)
    end
    write_impl(parent, name, data; kwargs...)
end

function write_impl(parent::Group, name::AbstractString, data::Union{<:Number}; kwargs...)
    dset = write_scalar(parent, name, data)
    write_attribute(dset, "encoding-type", "numeric-scalar")
    write_attribute(dset, "encoding-version", "0.2.0")
end

function write_impl(parent::Group, name::AbstractString, data::Union{<:AbstractString}; kwargs...)
    dset = write_scalar(parent, name, data)
    write_attribute(dset, "encoding-type", "string")
    write_attribute(dset, "encoding-version", "0.2.0")
end

function write_impl(parent::Group, name::AbstractString, data::AbstractDict{<:AbstractString, <:Any}; kwargs...)
    if length(data) > 0
        g = create_group(parent, name)
        write_attribute(g, "encoding-type", "dict")
        write_attribute(g, "encoding-version", "0.1.0")
        for (key, val) ∈ data
            write_impl(g, key, val; kwargs...)
        end
    end
end

function write_impl(parent::Group, name::AbstractString, data::Union{CategoricalArray, PooledArray}; kwargs...)
    g = create_group(parent, name)
    write_attribute(g, "encoding-type", "categorical")
    write_attribute(g, "encoding-version", "0.2.0")
    write_attribute(g, "ordered", isa(data, CategoricalArray) && isordered(data))
    write_impl(g, "categories", unwrap.(levels(data)); kwargs...)
    write_impl(g, "codes", data.refs .- true; kwargs...)
end

function write_impl(
    parent::Group,
    name::AbstractString,
    data::AbstractDataFrame;
    index::AbstractVector{<:AbstractString}=nothing,
    kwargs...,
)
    g = create_group(parent, name)
    write_attribute(g, "encoding-type", "dataframe")
    write_attribute(g, "encoding-version", "0.2.0")
    write_attribute(g, "column-order", names(data))

    for (name, column) ∈ pairs(eachcol(data))
        write_impl(g, string(name), column; kwargs...)
    end

    idxname = "_index"
    columns = names(data)
    if !isnothing(index)
        while idxname ∈ columns
            idxname = "_" * idxname
        end
    else
        if idxname ∈ columns
            index = data[!, idxname]
            select!(data, Not(idxname))
        else
            @warn "Data frame $(HDF5.name(parent))/$name does not have an _index column, a row number index will be written"
            index = collect(1:nrow(data))
        end
    end
    g = parent[name]
    write_impl(g, idxname, values(index); kwargs...)
    write_attribute(g, "_index", idxname)
end

write_impl(parent::Group, name::AbstractString, data::SubArray; kwargs...) =
    write_impl(parent, name, copy(data); kwargs...)

function write_impl(parent::Group, name::AbstractString, data::AbstractArray{Union{Bool, Missing}}; kwargs...)
    g = create_group(parent, name)

    write_attribute(g, "encoding-type", "nullable-boolean")
    write_attribute(g, "encoding-version", "0.1.0")

    write_impl(g, "mask", ismissing.(data))
    write_impl(g, "values", Bool[ismissing(v) ? 0 : v for v ∈ data])
end

function write_impl(
    parent::Group,
    name::AbstractString,
    data::AbstractArray{Union{T, Missing}};
    kwargs...,
) where {T <: Integer}
    g = create_group(parent, name)
    write_attribute(g, "encoding-type", "nullable-integer")
    write_attribute(g, "encoding-version", "0.1.0")

    write_impl(g, "mask", ismissing.(data))
    write_impl(g, "values", T[ismissing(v) ? 0 : v for v ∈ data])
end

function write_impl(
    parentgrp::Group,
    name::AbstractString,
    data::AbstractArray,
    ;
    extensible::Bool=false,
    compress::UInt8=0x9,
    kwargs...,
)
    if ndims(data) > 1
        data =
            data isa PermutedDimsArray && typeof(data).parameters[3] == Tuple(ndims(data):-1:1) ? parent(data) :
            permutedims(data, ndims(data):-1:1) # transpose for h5py compatibility
    end                                             # copy because HDF5 apparently can't handle lazy Adjoints

    write_impl_array(parentgrp, name, data, compress, extensible)
end

function write_impl(
    parent::Group,
    name::AbstractString,
    data::SparseMatrixCSC{<:Number, <:Integer};
    transposed=false,
    kwargs...,
)
    g = create_group(parent, name)
    write_attribute(g, "encoding-type", transposed ? "csr_matrix" : "csc_matrix")
    write_attribute(g, "encoding-version", "0.1.0")

    shape = collect(size(data))
    transposed && reverse!(shape)
    write_attribute(g, "shape", shape)
    write_impl(g, "indptr", data.colptr .- true, extensible=true)
    write_impl(g, "indices", data.rowval .- true, extensible=true)
    write_impl(g, "data", data.nzval, extensible=true)
end

write_impl(
    prt::Group,
    name::AbstractString,
    data::Adjoint{T, SparseMatrixCSC{T, V}} where {T <: Number, V <: Integer};
    kwargs...,
) = write_impl(prt, name, parent(data), transposed=true; kwargs...)

function write_impl(parent::Group, name::AbstractString, ::Nothing; kwargs...)
    dset = write_empty(parent, name, Float32)
    write_attribute(dset, "encoding-type", "null")
    write_attribute(dset, "encoding-version", "0.1.0")
end
