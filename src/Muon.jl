module Muon

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
  
  return table
end

mutable struct AnnData
    X::Array{Float64,2}
    obs::DataFrame
end

mutable struct MuData
  file::HDF5.File
  mod::Union{Dict{String, AnnData}, Nothing}

  obsm::Union{Dict{String, Any}, Nothing}
  obs::Union{DataFrame, Nothing}
  
  var::Union{DataFrame, Nothing}

  n_obs::Int64
  n_var::Int64
  
  function MuData(;file::HDF5.File)
    mdata = new(file)

    # Observations
    mdata.obs = readtable(file["obs"])
    mdata.obsm = HDF5.read(file["obsm"])

    # Variables
    mdata.var = readtable(file["var"])
    
    mdata.n_obs = size(mdata.file["obs"]["_index"])[1]
    mdata.n_var = size(mdata.file["var"]["_index"])[1]
    mdata
  end
end

function readh5mu(filename::AbstractString; backed=true)
  if backed
    fid = HDF5.h5open(filename)
  else
    fid = HDF5.h5open(filename, "r+")
  end
  mdata = MuData(file = fid)
  return mdata 
end

Base.size(mdata::MuData) = (mdata.n_obs, mdata.n_var)

Base.show(mdata::MuData) = print("""MuData object $(mdata.n_obs) \u2715 $(mdata.n_var)""")

export readh5mu, size
export AnnData, MuData

end # module
