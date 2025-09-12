abstract type AbstractAnnData end

"""
An annotated data object that stores data matrices with associated metadata.

# Constructor
```julia
AnnData(;
    X::AbstractMatrix{<:Number},
    obs::Union{DataFrame, Nothing}=nothing,
    obs_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
    var::Union{DataFrame, Nothing}=nothing,
    var_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
    obsm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
    varm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
    obsp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
    varp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
    layers::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
    uns::Union{AbstractDict{<:AbstractString, <:Any}, Nothing}=nothing,
)
```

# Keyword arguments / fields of the object
- `X`: An observations × variables matrix.
- `obs`: A [`DataFrame`](@extref DataFrames.DataFrame) with observation-level metadata.
- `obs_names`: A vector of observation names (identifiers).
- `var`: A [`DataFrame`](@extref DataFrames.DataFrame) with variable-level metadata.
- `var_names`: A vector of ariable names (identifiers).
- `obsm`: Dictionary with observation-level metadata.
- `varm`: Dictionary of observation-level metadata.
- `obsp`: Dictionary of pairwise observation-level metadata. Each element of `obsp` is a square matrix.
- `varp`: Dictionary of pairwise variable-level metadata. Each element of `varp` is a square matrix.
- `layers`: Dictionary of additional observations × variables matrices, e.g. for different processing/normalization
  steps.
- `uns`: Dictionary with unstructured metadata.
"""
mutable struct AnnData <: AbstractAnnData
    file::Union{HDF5.File, HDF5.Group, ZGroup, Nothing}

    X::Union{AbstractMatrix{<:Number}, Nothing}

    obs::DataFrame
    obs_names::Index{<:AbstractString}

    var::DataFrame
    var_names::Index{<:AbstractString}

    obsm::StrAlignedMapping{Tuple{1 => 1}, AnnData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, AnnData, false}

    varm::StrAlignedMapping{Tuple{1 => 2}, AnnData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, AnnData, false}

    layers::AbstractAlignedMapping{Tuple{1 => 1, 2 => 2}, String}

    uns::Dict{<:AbstractString, <:Any}

    function AnnData(file::Union{HDF5.File, HDF5.Group, ZGroup}, backed=false, checkversion=true)
        if checkversion
            if !has_attribute(file, "encoding-type")
                @warn "This file was not created by muon, we can't guarantee that everything will work correctly"
            elseif read_attribute(file, "encoding-type") != "anndata"
                error("This file does not appear to hold an AnnData object")
            end
        end

        adata = new(backed ? file : nothing)

        # this needs to go first because it's used by size() and size()
        # is used for dimensionalty checks
        adata.obs, obs_names = read_dataframe(file["obs"])
        adata.var, var_names = read_dataframe(file["var"])
        adata.obs_names = Index(obs_names)
        adata.var_names = Index(var_names)

        # observations
        adata.obsm =
            StrAlignedMapping{Tuple{1 => 1}}(adata, haskey(file, "obsm") ? read_dict_of_mixed(file["obsm"]) : nothing)
        adata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}, false}(
            adata,
            haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing,
        )

        # Variables
        adata.varm =
            StrAlignedMapping{Tuple{1 => 2}}(adata, haskey(file, "varm") ? read_dict_of_mixed(file["varm"]) : nothing)
        adata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}, false}(
            adata,
            haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing,
        )

        # X
        adata.X = backed ? nothing : read_matrix(file["X"])

        # Layers
        if !backed
            adata.layers = StrAlignedMapping{Tuple{1 => 1, 2 => 2}, false}(
                adata,
                haskey(file, "layers") ? read_dict_of_matrices(file["layers"]) : nothing,
            )
        else
            adata.layers = BackedAlignedMapping{Tuple{1 => 1, 2 => 2}}(adata, adata.file, "layers")
        end

        # unstructured
        adata.uns = haskey(file, "uns") ? read_dict_of_mixed(file["uns"], separate_index=false) : Dict{String, Any}()

        return adata
    end

    function AnnData(;
        X::AbstractMatrix{<:Number},
        obs::Union{DataFrame, Nothing}=nothing,
        obs_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        var::Union{DataFrame, Nothing}=nothing,
        var_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        obsm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
        varm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
        obsp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        varp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        layers::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        uns::Union{AbstractDict{<:AbstractString, <:Any}, Nothing}=nothing,
    )
        m, n = size(X)
        if isnothing(obs)
            obs = DataFrame()
        elseif size(obs, 1) != m
            throw(DimensionMismatch("X has $m rows, but obs has $(size(obs, 1)) rows"))
        end

        if isnothing(obs_names)
            obs_names = string.(collect(1:m))
        elseif length(obs_names) != m
            throw(DimensionMismatch("X has $m rows, but $(length(obs_names)) obs_names given"))
        end

        if isnothing(var)
            var = DataFrame()
        elseif size(var, 1) != n
            throw(DimensionMismatch("X has $n columns, but var has $(size(var, 1)) rows"))
        end

        if isnothing(var_names)
            var_names = string.(collect(1:n))
        elseif length(var_names) != n
            throw(DimensionMismatch("X has $n columns, but $(length(var_names)) var_names given"))
        end
        adata = new(nothing, X, obs, Index(obs_names), var, Index(var_names))
        adata.obsm = StrAlignedMapping{Tuple{1 => 1}}(adata, obsm)
        adata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}, false}(adata, obsp)
        adata.varm = StrAlignedMapping{Tuple{1 => 2}}(adata, varm)
        adata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}, false}(adata, varp)
        adata.layers = StrAlignedMapping{Tuple{1 => 1, 2 => 2}, false}(adata, layers)
        adata.uns = isnothing(uns) ? Dict{String, Any}() : uns

        return adata
    end
end

file(ad::AnnData) = ad.file

"""
    readh5ad(filename::AbstractString; backed=false)::AnnData

Read an [`AnnData`](@ref) object stored in an `h5ad` file.

In `backed` mode, matrices `X` and `layers` are not read into memory, but are instead represented
by proxy objects reading the required matrix elements from disk upon access.

See also [`readzarrad`](@ref), [`writeh5ad`](@ref), [`writezarrad`](@ref), [`isbacked`](@ref).
"""
function readh5ad(filename::AbstractString; backed=false)
    filename = abspath(filename) # this gets stored in the HDF5 objects for backed datasets
    if !backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    local adata
    try
        adata = AnnData(fid, backed, false)
    catch e
        close(fid)
        rethrow()
    end
    if !backed
        close(fid)
    end
    return adata
end

"""
    readzarrad(filename::AbstractString; backed=false)::AnnData

Read an [`AnnData`](@ref) object stored in a Zarr file.

In `backed` mode, matrices `X` and `layers` are not read into memory, but are instead represented
by proxy objects reading the required matrix elements from disk upon access.

See also [`readh5ad`](@ref), [`writeh5ad`](@ref), [`writezarrad`](@ref), [`isbacked`](@ref).
"""
function readzarrad(filename::AbstractString; backed=false)
    filename = abspath(filename)
    if !backed
        fid = zopen(filename, "r")
    else
        fid = zopen(filename, "r+")
    end
    local adata
    try
        adata = AnnData(fid, backed, true)
    catch e
        close(fid)
        rethrow()
    end
    if !backed
        close(fid)
    end
    return adata
end

"""
    writeh5ad(filename::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)

Write an [`AnnData`](@ref) object to disk using the h5ad format (HDF5 with a particular structure).

`compress` indicates the level of compression to apply, from 0 (no compression) to 9 (highest compression).

See also [`writezarrad`](@ref), [`readh5ad`](@ref), [`readzarrad`](@ref).
"""
function writeh5ad(filename::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)
    filename = abspath(filename)
    if isnothing(file(adata)) || filename != HDF5.filename(file(adata))
        hfile = h5open(filename, "w", userblock=512)
        try
            write(hfile, adata, compress=compress)
            close(hfile)
            hfile = open(filename, "r+")
            write(hfile, "AnnData (format-version=$ANNDATAVERSION;creator=$NAME;creator-version=$VERSION)")
        finally
            close(hfile)
        end
    else
        write(adata, compress=compress)
    end
    return nothing
end

"""
    writezarrad(filename::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)

Write an [`AnnData`](@ref) object to disk using the Zarr format.

`compress` indicates the level of compression to apply, from 0 (no compression) to 9 (highest compression).

See also [`writeh5ad`](@ref), [`readh5ad`](@ref), [`readzarrad`](@ref).
"""
function writezarrad(filename::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)
    filename = abspath(filename)
    if isnothing(file(adata)) || filename != zarr_filename(file(adata))
        rm(filename, force=true, recursive=true)
        zfile = zgroup(filename)
        write(zfile, adata, compress=compress)
    else
        write(adata, compress=compress)
    end
    return nothing
end

# HDF5.jl defines Base.write(::Union{HDF5.File, HDF5.Group}, ::Union{Nothing, AbstractString}, ::Any; kwargs...)
# Using the below as Base.write leads to ambiguity: The HDF5.jl definition is more specific in the first argument,
# ours is more specific in the third argument. Thus the ugly workaround.
function _write(parent::Group, name::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)
    g = create_group(parent, name)
    write(g, adata, compress=compress)
end

"""
    write(parent::Union{HDF5.File, HDF5.Group, ZGroup}, name::AbstractString, adata::AbstractAnnData; compress::UInt8=0x9)

Write the `adata` to an already open HDF5 or Zarr file into the group `parent` as a subgroup with name `name`.
"""
Base.write(
    parent::Union{HDF5.File, HDF5.Group},
    name::AbstractString,
    adata::AbstractAnnData;
    compress::UInt8=0x9,
)=_write(parent, name, adata, compress=compress)
Base.write(
    parent::ZGroup,
    name::AbstractString,
    adata::AbstractAnnData;
    compress::UInt8=0x9,
)=_write(parent, name, adata, compress=compress)

function Base.write(parent::Group, adata::AbstractAnnData; compress::UInt8=0x9)
    write_attribute(parent, "encoding-type", "anndata")
    write_attribute(parent, "encoding-version", string(ANNDATAVERSION))
    write_attribute(parent, "encoder", NAME)
    write_attribute(parent, "encoder-version", string(VERSION))
    if parent === file(adata)
        write(adata, compress=compress)
    else
        write_attr(parent, "X", adata.X, compress=compress)
        write_attr(parent, "layers", adata.layers, compress=compress)
        write_metadata(parent, adata, compress=compress)
    end
end

function Base.write(adata::AbstractAnnData; compress::UInt8=0x9)
    if !isbacked(adata)
        error("adata is not backed, need somewhere to write to")
    end
    write_metadata(file(adata), adata, compress=compress)
end

function write_metadata(parent::Group, adata::AbstractAnnData; compress::UInt8=0x9)
    write_attr(parent, "obs", adata.obs, index=adata.obs_names, compress=compress)
    write_attr(parent, "obsm", adata.obsm, index=adata.obs_names, compress=compress)
    write_attr(parent, "obsp", adata.obsp, compress=compress)
    write_attr(parent, "var", adata.var, index=adata.var_names, compress=compress)
    write_attr(parent, "varm", adata.varm, index=adata.var_names, compress=compress)
    write_attr(parent, "varp", adata.varp, compress=compress)
    write_attr(parent, "uns", adata.uns, compress=compress)
end
# FileIO support
load(f::File{format"h5ad"}; backed::Bool=false) = readh5ad(filename(f), backed=backed)
save(f::File{format"h5ad"}, data::AbstractAnnData; compress::UInt8=0x9) =
    writeh5ad(filename(f), data, compress=compress)

Base.size(adata::AbstractAnnData) = (length(adata.obs_names), length(adata.var_names))
Base.size(adata::AbstractAnnData, d::Integer) = size(adata)[d]

function Base.show(io::IO, adata::AbstractAnnData)
    compact = get(io, :compact, false)
    print(io, """$(typeof(adata).name.name) object $(size(adata)[1]) \u2715 $(size(adata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", adata::AbstractAnnData)
    show(io, adata)
end

function Base.getproperty(adata::AnnData, s::Symbol)
    if s === :X && isbacked(adata)
        return backed_matrix(file(adata)["X"])
    else
        return getfield(adata, s)
    end
end

function Base.getindex(
    adata::AbstractAnnData,
    I::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Integer, AbstractString},
    J::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Integer, AbstractString},
)
    @boundscheck checkbounds(adata, I, J)
    i, j = convertidx(I, adata.obs_names), convertidx(J, adata.var_names)
    newad = AnnData(
        X=adata.X[i, j],
        obs=isempty(adata.obs) ? nothing : adata.obs[i, :],
        obs_names=adata.obs_names[i],
        var=isempty(adata.var) ? nothing : adata.var[j, :],
        var_names=adata.var_names[j],
    )
    copy_subset(adata.obsm, newad.obsm, i, j)
    copy_subset(adata.varm, newad.varm, i, j)
    copy_subset(adata.obsp, newad.obsp, i, j)
    copy_subset(adata.varp, newad.varp, j, j)
    copy_subset(adata.layers, newad.layers, i, j)
    return newad
end

struct AnnDataView{Ti, Tj} <: AbstractAnnData
    parent::AnnData
    I::Ti
    J::Tj

    X::Union{AbstractMatrix{<:Number}, Nothing}

    obs::SubDataFrame
    obs_names::SubIndex{<:AbstractString}

    var::SubDataFrame
    var_names::SubIndex{<:AbstractString}

    obsm::StrAlignedMappingView{Tuple{1 => 1}}
    obsp::StrAlignedMappingView{Tuple{1 => 1, 2 => 1}}

    varm::StrAlignedMappingView{Tuple{1 => 2}}
    varp::StrAlignedMappingView{Tuple{1 => 2, 2 => 2}}

    layers::StrAlignedMappingView{Tuple{1 => 1, 2 => 2}}

    uns::Dict{<:AbstractString, <:Any}
end

function Base.view(ad::AnnData, I, J)
    @boundscheck checkbounds(ad, I, J)
    i, j = convertidx(I, ad.obs_names), convertidx(J, ad.var_names)
    X = isbacked(ad) ? nothing : @view ad.X[i, j]

    return AnnDataView(
        ad,
        i,
        j,
        X,
        view(ad.obs, nrow(ad.obs) > 0 ? i : (:), :),
        view(ad.obs_names, i),
        view(ad.var, nrow(ad.var) > 0 ? j : (:), :),
        view(ad.var_names, j),
        view(ad.obsm, i),
        view(ad.obsp, i, i),
        view(ad.varm, j),
        view(ad.varp, j, j),
        view(ad.layers, i, j),
        ad.uns,
    )
end
function Base.view(ad::AnnDataView, I, J)
    @boundscheck checkbounds(ad, I, J)
    i, j = Base.reindex(parentindices(ad), (convertidx(I, ad.obs_names), convertidx(J, ad.var_names)))
    return view(parent(ad), i, j)
end

function Base.getproperty(ad::AnnDataView, s::Symbol)
    if s === :X && isbacked(ad)
        return @view parent(ad).X[parentindices(ad)...]
    else
        return getfield(ad, s)
    end
end

Base.parent(ad::AnnData) = ad
Base.parent(ad::AnnDataView) = ad.parent
Base.parentindices(ad::AnnData) = axes(ad)
Base.parentindices(ad::AnnDataView) = (ad.I, ad.J)
file(ad::AnnDataView) = file(parent(ad))

"""
    var_names_make_unique!(ad::AnnData, join='-')

Make `ad.var_names` unique by appending `join` and sequential numbers
(1, 2, 3 etc) to duplicate elements, leaving the first unchanged.

See also [`obs_names_make_unique!(::AnnData)`](@ref).
"""
function var_names_make_unique!(ad::AnnData, join='-')
    attr_make_unique!(ad, :var_names, join)
end

"""
    obs_names_make_unique!(ad::AnnData, join='-')

Make `ad.obs_names` unique by appending `join` and sequential numbers
(1, 2, 3 etc) to duplicate elements, leaving the first unchanged.

See also [`var_names_make_unique!(::AnnData)`](@ref).
"""
function obs_names_make_unique!(ad::AnnData, join='-')
    attr_make_unique!(ad, :obs_names, join)
end

function attr_make_unique!(ad::AnnData, namesattr::Symbol, join)
    index = getproperty(ad, namesattr)
    if !allunique(index)
        duplicates = duplicateindices(index)

        example_colliding_names = []
        for (name, positions) ∈ duplicates
            i = 1
            for pos ∈ Iterators.rest(positions, 2)
                while true
                    potential = string(index[pos], join, i)
                    i += 1
                    if potential ∈ index
                        if length(example_colliding_names) <= 5
                            push!(example_colliding_names, potential)
                        end
                    else
                        index[pos] = potential
                        break
                    end
                end
            end
        end

        if !isempty(example_colliding_names)
            @warn """
                Appending $(join)[1-9...] to duplicates caused collision with another name.
                Example(s): $example_colliding_names
                This may make the names hard to interperet.
                Consider setting a different delimiter with `join={delimiter}`
                """
        end
    else
        @info "var names are already unique, doing nothing"
    end
    return ad
end

"""
    DataFrame(ad::AnnData; layer=nothing, columns=:var)

Return a DataFrame containing the data matrix `ad.X` (or `layer` by
passing `layer="layername"`). By default, the first column contains
`ad.obs_names` and the remaining columns are named according to
`ad.var_names`, to obtain the transpose, pass `columns=:obs`.
"""
function DataFrames.DataFrame(ad::AnnData; layer::Union{String, Nothing}=nothing, columns=:var)
    if columns ∉ [:obs, :var]
        throw(ArgumentError("columns must be :obs or :var (got: $columns)"))
    end
    rows = columns == :var ? :obs : :var
    colnames = getproperty(ad, Symbol(columns, :_names))
    if !allunique(colnames)
        throw(ArgumentError("duplicate column names ($(columns)_names); run $(columns)_names_make_unique!"))
    end
    rownames = getproperty(ad, Symbol(rows, :_names))

    M = if isnothing(layer)
        ad.X
    elseif layer in keys(ad.layers)
        ad.layers[layer]
    else
        throw(ArgumentError("no layer $layer in adata layers"))
    end
    df = DataFrame(columns == :var ? M : transpose(M), colnames)
    setproperty!(df, rows, rownames)
    select!(df, rows, All())
    df
end
