module Muon

using Random
using SparseArrays
import LinearAlgebra: Adjoint

using HDF5
using DataFrames
using CategoricalArrays
using StructArrays
using PooledArrays
import CompressHashDisplace: FrozenDict
import OrderedCollections: OrderedDict
using FileIO

export readh5mu, readh5ad, writeh5mu, writeh5ad, isbacked, update_obs!, update_var!, update!
export AnnData, MuData

import Pkg
# this executes only during precompilation
let
    pkg = Pkg.Types.read_package(joinpath(@__DIR__, "..", "Project.toml"))
    global VERSION = pkg.version
    global NAME = pkg.name * ".jl"
end
MUDATAVERSION = v"0.1.0"
ANNDATAVERSION = v"0.1.0"


include("index.jl")
include("sparsedataset.jl")
include("transposeddataset.jl")
include("hdf5_io.jl")
include("alignedmapping.jl")
include("anndata.jl")
include("mudata.jl")
include("util.jl")

end # module
