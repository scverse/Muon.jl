module Muon

import SparseArrays: SparseMatrixCSC
using HDF5
import DataFrames: DataFrame
import CategoricalArrays: CategoricalArray

function read_dataframe(tablegroup::HDF5.Group)
    columns = read_attribute(tablegroup, "column-order")

    havecat = false
    if haskey(tablegroup, "__categories")
        havecat = true
        catcols = tablegroup["__categories"]
    end

    df = DataFrame()

    if haskey(attributes(tablegroup), "_index")
        indexdsetname = read_attribute(tablegroup, "_index")
        df[!, indexdsetname] = read(tablegroup[indexdsetname])
    end

    for col ∈ columns
        column = read(tablegroup, col)
        if havecat && haskey(catcols, col)
            cats = read(catcols, col)
            column = CategoricalArray(map(x -> cats[x + 1], column))
        end
        df[!, col] = column
    end

    return df
end

function read_csc_matrix(f::HDF5.Group)
    shape = read_attribute(f, "shape")
    return SparseMatrixCSC(
        shape[1],
        shape[2],
        read(f, "indptr") .+ 1,
        read(f, "indices") .+ 1,
        read(f, "data"),
    )
end

function read_csr_matrix(f::HDF5.Group)
    shape = read_attribute(f, "shape")
    return copy(
        SparseMatrixCSC(
            shape[2],
            shape[1],
            read(f, "indptr") .+ 1,
            read(f, "indices") .+ 1,
            read(f, "data"),
        )',
    )
end

muread(f) = read(f)

function muread(f::HDF5.Group)
    if haskey(attributes(f), "encoding-type")
        enctype = read_attribute(f, "encoding-type")
        if enctype == "dataframe"
            return read_dataframe(f)
        elseif enctype == "csc_matrix"
            return read_csc_matrix(f)
        elseif enctype == "csr_matrix"
            return read_csr_matrix(f)
        else
            throw("unknown encoding $enctype")
        end
    else
        return read(f)
    end
end


mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group}

    X::Union{AbstractArray{Float64, 2}, AbstractArray{Float32, 2}, AbstractArray{Int, 2}}
    layers::Union{Dict{String, Any}, Nothing}

    obs::Union{DataFrame, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}

    var::Union{DataFrame, Nothing}
    varm::Union{Dict{String, Any}, Nothing}

    function AnnData(file::Union{HDF5.File, HDF5.Group})
        adata = new(file)

        # Observations
        adata.obs = muread(file["obs"])
        adata.obsm = muread(file["obsm"])

        # Variables
        adata.var = muread(file["var"])
        adata.varm = "varm" ∈ keys(file) ? muread(file["varm"]) : nothing

        # X
        adata.X = muread(file["X"])

        # Layers
        if "layers" in HDF5.keys(file)
            adata.layers = Dict{String, Any}()
            layers = HDF5.keys(file["layers"])
            for layer in layers
                # TODO: Make a SparseMatrix if sparse
                adata.layers[layer] = muread(file["layers"][layer])
            end
        end

        adata
    end
end

mutable struct MuData
    file::HDF5.File
    mod::Union{Dict{String, AnnData}, Nothing}

    obs::Union{DataFrame, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}

    var::Union{DataFrame, Nothing}
    varm::Union{Dict{String, Any}, Nothing}

    function MuData(file::HDF5.File)
        mdata = new(file)

        # Observations
        mdata.obs = muread(file["obs"])
        mdata.obsm = muread(file["obsm"])

        # Variables
        mdata.var = muread(file["var"])
        mdata.varm = muread(file["varm"])

        # Modalities
        mdata.mod = Dict{String, AnnData}()
        mods = HDF5.keys(mdata.file["mod"])
        for modality in mods
            adata = AnnData(mdata.file["mod"][modality])
            mdata.mod[modality] = adata
        end

        mdata
    end
end

function readh5mu(filename::AbstractString; backed=true)
    if backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    mdata = MuData(fid)
    return mdata
end

function readh5ad(filename::AbstractString; backed=true)
    if backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    adata = AnnData(fid)
    return adata
end

Base.size(adata::AnnData) =
    (size(adata.file["obs"]["_index"])[1], size(adata.file["var"]["_index"])[1])
Base.size(mdata::MuData) =
    (size(mdata.file["obs"]["_index"])[1], size(mdata.file["var"]["_index"])[1])

Base.getindex(mdata::MuData, modality::Symbol) = mdata.mod[String(modality)]
Base.getindex(mdata::MuData, modality::AbstractString) = mdata.mod[modality]

function Base.show(io::IO, adata::AnnData)
    compact = get(io, :compact, false)
    print(io, """AnnData object $(size(adata)[1]) \u2715 $(size(adata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", adata::AnnData)
    show(io, adata)
end

function Base.show(io::IO, mdata::MuData)
    compact = get(io, :compact, false)
    print(io, """MuData object $(size(mdata)[1]) \u2715 $(size(mdata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", mdata::MuData)
    show(io, mdata)
end

export readh5mu, readh5ad
export AnnData, MuData

end # module
