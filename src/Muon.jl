module Muon

using Random
using SparseArrays
import LinearAlgebra: Adjoint

using EllipsisNotation
using HDF5
using Zarr
using DataFrames
using CategoricalArrays
using StructArrays
using PooledArrays
using FillArrays
using DataStructures
import CompressHashDisplace: FrozenDict
using FileIO

export readh5mu,
    readh5ad,
    readzarrmu,
    readzarrad,
    writeh5mu,
    writeh5ad,
    writezarrmu,
    writezarrad,
    isbacked,
    update_obs!,
    update_var!,
    update!,
    push_obs!,
    push_var!,
    pull_obs!,
    pull_var!
export AnnData, MuData
export var_names_make_unique!, obs_names_make_unique!

import Pkg
# this executes only during precompilation
let
    pkg = Pkg.Types.read_package(joinpath(@__DIR__, "..", "Project.toml"))
    global VERSION = pkg.version
    global NAME = pkg.name * ".jl"
end
MUDATAVERSION = v"0.1.0"
ANNDATAVERSION = v"0.1.0"

include("typedefs.jl")
include("index.jl")
include("sparsedataset.jl")
include("transposeddataset.jl")
include("common_io.jl")
include("hdf5_io.jl")
include("zarr_io.jl")
include("alignedmapping.jl")
include("anndata.jl")
include("mudata.jl")
include("util.jl")

end # module
