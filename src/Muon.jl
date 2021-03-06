module Muon

import HDF5

struct AnnData
    X::Array{Float64,2}
end

Base.@kwdef mutable struct MuData
  mod::Union{Dict{String, AnnData}, Missing} = missing
  file::HDF5.File

  obsm::Union{Dict{String, Any}, Missing} = missing

  n_obs::Int64 = 0
  n_var::Int64 = 0
  
  function MuData(Missing, file::HDF5.File, obsm, n_obs::Int64, n_var::Int64)
    mdata = new(missing, file)

    mdata.obsm = HDF5.read(file["obsm"])
    
    mdata.n_obs = size(mdata.file["obs"]["_index"])[1]
    mdata.n_var = size(mdata.file["var"]["_index"])[1]
    mdata
  end

end

function read(filename::AbstractString; backed=true)
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

h5muopen = read

export h5muopen, size
export AnnData, MuData

end # module
