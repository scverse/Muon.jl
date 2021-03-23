module Muon

using SparseArrays
import LinearAlgebra: Adjoint

using HDF5
using DataFrames
using CategoricalArrays

export readh5mu, readh5ad, writeh5mu, writeh5ad, isbacked
export AnnData, MuData

include("hdf5_io.jl")
include("sparsedataset.jl")
include("anndata.jl")
include("mudata.jl")

end # module
