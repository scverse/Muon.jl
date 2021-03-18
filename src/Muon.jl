module Muon

import SparseArrays: SparseMatrixCSC
import HDF5
import DataFrames: DataFrame
import CategoricalArrays: CategoricalArray

function readtable(tablegroup::HDF5.Group)
  tabledict = HDF5.read(tablegroup)

  if haskey(tabledict, "__categories")
    for (k, cats) in tabledict["__categories"]
      tabledict[k] = CategoricalArray(map(x -> cats[x+1], tabledict[k]))
    end
  end

  delete!(tabledict, "__categories")
  table = DataFrame(tabledict)

  table
end

function read_matrix(f::HDF5.Dataset)
  return HDF5.read(f)
end

function read_matrix(f::HDF5.Group)
  enctype = HDF5.read_attribute(f, "encoding-type")
  shape = HDF5.read_attribute(f, "shape")
  if enctype == "csc_matrix"
    return SparseMatrixCSC(shape[1], shape[2], HDF5.read(f, "indptr") .+ 1, HDF5.read(f, "indices") .+ 1, HDF5.read(f, "data"))
  elseif enctype == "csr_matrix"
    return copy(SparseMatrixCSC(shape[2], shape[1], HDF5.read(f, "indptr") .+ 1, HDF5.read(f, "indices") .+ 1, HDF5.read(f, "data"))')
  else
    throw("unknown matrix encoding $enctype")
  end
end


mutable struct AnnData
  file::Union{HDF5.File,HDF5.Group}

  X::Union{AbstractArray{Float64,2}, AbstractArray{Float32,2}, AbstractArray{Int,2}}
  layers::Union{Dict{String, Any}, Nothing}

  obs::Union{DataFrame, Nothing}
  obsm::Union{Dict{String, Any}, Nothing}

  var::Union{DataFrame, Nothing}
  varm::Union{Dict{String, Any}, Nothing}

  function AnnData(;file::Union{HDF5.File,HDF5.Group})
    adata = new(file)

    # Observations
    adata.obs = readtable(file["obs"])
    adata.obsm = HDF5.read(file["obsm"])

    # Variables
    adata.var = readtable(file["var"])
    adata.varm = "varm" ∈ keys(file) ? HDF5.read(file["varm"]) : nothing

    # X
    adata.X = read_matrix(file["X"])

    # Layers
    if "layers" in HDF5.keys(file)
      adata.layers = Dict{String,Any}()
      layers = HDF5.keys(file["layers"])
      for layer in layers
        # TODO: Make a SparseMatrix if sparse
        adata.layers[layer] = read_matrix(file["layers"][layer])
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

  function MuData(;file::HDF5.File)
    mdata = new(file)

    # Observations
    mdata.obs = readtable(file["obs"])
    mdata.obsm = HDF5.read(file["obsm"])

    # Variables
    mdata.var = readtable(file["var"])
    mdata.varm = HDF5.read(file["varm"])

    # Modalities
    mdata.mod = Dict{String,AnnData}()
    mods = HDF5.keys(mdata.file["mod"])
    for modality in mods
      adata = AnnData(file=mdata.file["mod"][modality])
      mdata.mod[modality] = adata
    end

    mdata
  end
end

function readh5mu(filename::AbstractString; backed=true)
  if backed
    fid = HDF5.h5open(filename, "r")
  else
    fid = HDF5.h5open(filename, "r+")
  end
  mdata = MuData(file = fid)
  return mdata
end

function readh5ad(filename::AbstractString; backed=true)
  if backed
    fid = HDF5.h5open(filename, "r")
  else
    fid = HDF5.h5open(filename, "r+")
  end
  adata = AnnData(file = fid)
  return adata
end

Base.size(adata::AnnData) = (size(adata.file["obs"]["_index"])[1], size(adata.file["var"]["_index"])[1])
Base.size(mdata::MuData) = (size(mdata.file["obs"]["_index"])[1], size(mdata.file["var"]["_index"])[1])

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
