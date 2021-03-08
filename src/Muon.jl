module Muon

import HDF5

struct AnnData
    X::Array{Float64,2}
end

mutable struct MuData
  file::HDF5.File
  mod::Union{Dict{String, AnnData}, Nothing}

  obsm::Union{Dict{String, Any}, Nothing}
  n_obs::Int64
  n_var::Int64
  
  function MuData(;file::HDF5.File)
    mdata = new(file)

    mdata.obsm = HDF5.read(file["obsm"])
    
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
