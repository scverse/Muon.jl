abstract type AbstractAnnData end

mutable struct AnnData <: AbstractAnnData
    file::Union{HDF5.File, HDF5.Group, Nothing}

    X::Union{AbstractMatrix{<:Number}, Nothing}

    obs::DataFrame
    obs_names::Index{<:AbstractString}

    var::DataFrame
    var_names::Index{<:AbstractString}

    obsm::StrAlignedMapping{Tuple{1 => 1}, AnnData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, AnnData}

    varm::StrAlignedMapping{Tuple{1 => 2}, AnnData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, AnnData}

    layers::AbstractAlignedMapping{Tuple{1 => 1, 2 => 2}, String}

    function AnnData(file::Union{HDF5.File, HDF5.Group}, backed=true)
        adata = new(backed ? file : nothing)

        # this needs to go first because it's used by size() and size()
        # is used for dimensionalty checks
        adata.obs, obs_names = read_dataframe(file["obs"])
        adata.var, var_names = read_dataframe(file["var"])
        adata.obs_names = Index(obs_names)
        adata.var_names = Index(var_names)

        # observations
        adata.obsm = StrAlignedMapping{Tuple{1 => 1}}(
            adata,
            haskey(file, "obsm") ? read_dict_of_mixed(file["obsm"]) : nothing,
        )
        adata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(
            adata,
            haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing,
        )

        # Variables
        adata.varm = StrAlignedMapping{Tuple{1 => 2}}(
            adata,
            haskey(file, "varm") ? read_dict_of_mixed(file["varm"]) : nothing,
        )
        adata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(
            adata,
            haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing,
        )

        # X
        adata.X = backed ? nothing : read_matrix(file["X"])

        # Layers
        if !backed
            adata.layers = StrAlignedMapping{Tuple{1 => 1, 2 => 2}}(
                adata,
                haskey(file, "layers") ? read_dict_of_matrices(file["layers"]) : nothing,
            )
        else
            adata.layers = BackedAlignedMapping{Tuple{1 => 1, 2 => 2}}(adata, adata.file, "layers")
        end

        return adata
    end

    function AnnData(;
        X::AbstractMatrix{<:Number},
        obs::Union{DataFrame, Nothing}=nothing,
        obs_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        var::Union{DataFrame, Nothing}=nothing,
        var_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        obsm::Union{
            AbstractDict{<:AbstractString, Union{AbstractArray{<:Number}, DataFrame}},
            Nothing,
        }=nothing,
        varm::Union{
            AbstractDict{<:AbstractString, Union{AbstractArray{<:Number}, DataFrame}},
            Nothing,
        }=nothing,
        obsp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
        varp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
        layers::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
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
        adata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(adata, obsp)
        adata.varm = StrAlignedMapping{Tuple{1 => 2}}(adata, varm)
        adata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(adata, varp)
        adata.layers = StrAlignedMapping{Tuple{1 => 1, 2 => 2}}(adata, layers)

        return adata
    end
end

file(ad::AnnData) = ad.file

function readh5ad(filename::AbstractString; backed=true)
    filename = abspath(filename) # this gets stored in the HDF5 objects for backed datasets
    if !backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    local adata
    try
        adata = AnnData(fid, backed)
    finally
        if !backed
            close(fid)
        end
    end
    return adata
end

function writeh5ad(filename::AbstractString, adata::AbstractAnnData)
    filename = abspath(filename)
    if file(adata) === nothing || filename != HDF5.filename(file(adata))
        file = h5open(filename, "w")
        try
            write(file, adata)
        finally
            close(file)
        end
    else
        write(adata)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, adata::AbstractAnnData)
    g = create_group(parent, name)
    write(g, adata)
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, adata::AbstractAnnData)
    if parent === file(adata)
        write(adata)
    else
        write_attr(parent, "X", adata.X)
        write_attr(parent, "layers", adata.layers)
        write_unbacked(parent, adata)
    end
end

function Base.write(adata)
    if file(adata) === nothing
        throw("adata is not backed, need somewhere to write to")
    end
    write_unbacked(file(adata), adata)
end

function write_unbacked(parent::Union{HDF5.File, HDF5.Group}, adata::AbstractAnnData)
    write_attr(parent, "obs", adata.obs_names, adata.obs)
    write_attr(parent, "obsm", adata.obsm)
    write_attr(parent, "obsp", adata.obsp)
    write_attr(parent, "var", adata.var_names, adata.var)
    write_attr(parent, "varm", adata.varm)
    write_attr(parent, "varp", adata.varp)
end

Base.size(adata::AbstractAnnData) = (length(adata.obs_names), length(adata.var_names))
Base.size(adata::AbstractAnnData, d::Integer) = size(adata)[d]

function Base.show(io::IO, adata::AbstractAnnData)
    compact = get(io, :compact, false)
    print(io, """AnnData object $(size(adata)[1]) \u2715 $(size(adata)[2])""")
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
    I::Union{AbstractUnitRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Integer, AbstractString},
    J::Union{AbstractUnitRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Integer, AbstractString},
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
end

function Base.view(ad::AnnData, I, J)
    @boundscheck checkbounds(ad, I, J)
    i, j = convertidx(I, ad.obs_names), convertidx(J, ad.var_names)
    X = isbacked(ad) ? nothing : @view ad.X[i, j]

    return AnnDataView(ad, i, j, X, view(ad.obs, i, :), view(ad.obs_names, i), view(ad.var, j, :), view(ad.var_names, j), view(ad.obsm, i), view(ad.obsp, i, i), view(ad.varm, j), view(ad.varp, j, j), view(ad.layers, i, j))
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

Base.parent(ad::AnnDataView) = ad.parent
Base.parentindices(ad::AnnDataView) = (ad.I, ad.J)
file(ad::AnnDataView) = file(parent(ad))
