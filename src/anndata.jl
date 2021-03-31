mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group, Nothing}

    X::Union{AbstractMatrix{<:Number}, Nothing}

    obs::DataFrame
    obs_names::AbstractVector{<:AbstractString}

    var::DataFrame
    var_names::AbstractVector{<:AbstractString}

    obsm::StrAlignedMapping{Tuple{1 => 1}, AnnData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, AnnData}

    varm::StrAlignedMapping{Tuple{1 => 2}, AnnData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, AnnData}

    layers::AbstractAlignedMapping{Tuple{1 => 1, 2 => 2}, String}

    function AnnData(file::Union{HDF5.File, HDF5.Group}, backed=true)
        adata = new(backed ? file : nothing)

        # this needs to go first because it's used by size() and size()
        # is used for dimensionalty checks
        adata.obs, adata.obs_names = read_dataframe(file["obs"])
        adata.var, adata.var_names = read_dataframe(file["var"])

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
        adata = new(nothing, X, obs, obs_names, var, var_names)
        adata.obsm = StrAlignedMapping{Tuple{1 => 1}}(adata, obsm)
        adata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(adata, obsp)
        adata.varm = StrAlignedMapping{Tuple{1 => 2}}(adata, varm)
        adata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(adata, varp)
        adata.layers = StrAlignedMapping{Tuple{1 => 1, 2 => 2}}(adata, layers)

        return adata
    end
end

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

function writeh5ad(filename::AbstractString, adata::AnnData)
    filename = abspath(filename)
    if adata.file === nothing || filename != HDF5.filename(adata.file)
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

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, adata::AnnData)
    g = create_group(parent, name)
    write(g, adata)
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, adata::AnnData)
    if parent === adata.file
        write(adata)
    else
        write_attr(parent, "X", adata.X)
        write_unbacked(parent, adata)
    end
end

function Base.write(adata)
    if adata.file === nothing
        throw("adata is not backed, need somewhere to write to")
    end
    write_unbacked(adata.file, adata)
end

function write_unbacked(parent::Union{HDF5.File, HDF5.Group}, adata::AnnData)
    write_attr(parent, "layers", adata.layers)
    write_attr(parent, "obs", adata.obs_names, adata.obs)
    write_attr(parent, "obsm", adata.obsm)
    write_attr(parent, "obsp", adata.obsp)
    write_attr(parent, "var", adata.var_names, adata.var)
    write_attr(parent, "varm", adata.varm)
    write_attr(parent, "varp", adata.varp)
end

Base.size(adata::AnnData) = (length(adata.obs_names), length(adata.var_names))
Base.size(adata::AnnData, d::Integer) = size(adata)[d]

function Base.show(io::IO, adata::AnnData)
    compact = get(io, :compact, false)
    print(io, """AnnData object $(size(adata)[1]) \u2715 $(size(adata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", adata::AnnData)
    show(io, adata)
end

isbacked(adata::AnnData) = adata.file !== nothing

function Base.getproperty(adata::AnnData, s::Symbol)
    if s === :X && isbacked(adata)
        return backed_matrix(adata.file["X"])
    else
        return getfield(adata, s)
    end
end

function Base.getindex(
    adata::AnnData,
    I::Union{AbstractUnitRange, Colon, Vector{<:Integer}},
    J::Union{AbstractUnitRange, Colon, Vector{<:Integer}},
)
    newad = AnnData(
        X=adata.X[I, J],
        obs=isnothing(adata.obs) ? nothing : adata.obs[I, :],
        obs_names=adata.obs_names[I],
        var=isnothing(adata.var) ? nothing : adata.var[J, :],
        var_names=adata.var_names[J],
    )
    copy_subset(adata.obsm, newad.obsm, I, J)
    copy_subset(adata.varm, newad.varm, I, J)
    copy_subset(adata.obsp, newad.obsp, I, J)
    copy_subset(adata.varp, newad.varp, I, J)
    copy_subset(adata.layers, newad.layers, I, J)
    return newad
end
