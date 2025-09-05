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

using Compat

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
    pull_var!,
    var_names_make_unique!,
    obs_names_make_unique!
export AnnData, MuData
@compat public write, size, getindex, setindex!, view
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

# documentation common for AnnData and MuData
"""
    getindex(
        data::Union{AbstractAnnData, AbstractMuData},
        I::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
        J::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
    )

Subset `data` to observations `I` and variables `J`.

Indexing can be performed by numerical index or by name, where names are looked up in `data.obs_names` and `data.var_names`.
"""
getindex

"""
    view(data::Union{AbstractAnnData, AbstractMuData},
        I::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
        J::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
    )

Like [`getindex`](@ref), but returns a lightweight object that lazily references the parent object.
"""
view

"""
    size(data::Union{AbstractAnnData, AbstractMuData}, [dim::Integer])

Return a tuple containing the number of observations and variables of `adata`. Optionally you can
specify a dimension to get just the length of that dimension.
"""
size

"""
    write(data::Union{AbstractAnnData, AbstractMuData}; compress::UInt8=0x9)
    write(parent::Union{HDF5.File, HDF5.Group, ZGroup}, data::Union{AbstractAnnData, AbstractMuData}; compress::UInt8=0x9)

Write the `data` to disk.

The first form writes the metadata of a backed [`AnnData`](@ref) or [`MuData`](@ref) object to disk.
The second form writes the `data` to an already open HDF5 or Zarr file into the group `parent`.
"""
write
end # module
