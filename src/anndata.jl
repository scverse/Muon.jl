mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group, Nothing}

    X::Union{AbstractMatrix{<:Number}, Nothing}
    layers::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{Vector{String}, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}
    obsp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{Vector{String}, Nothing}
    varm::Union{Dict{String, Any}, Nothing}
    varp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    function AnnData(file::Union{HDF5.File, HDF5.Group}, backed=true)
        adata = new(backed ? file : nothing)

        # Observations
        adata.obs, adata.obs_names = read_dataframe(file["obs"])
        adata.obsm = haskey(file, "obsm") ? read(file["obsm"]) : nothing
        adata.obsp = haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing

        # Variables
        adata.var, adata.var_names = read_dataframe(file["var"])
        adata.varm = haskey(file, "varm") ? read(file["varm"]) : nothing
        adata.varp = haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing

        # X
        adata.X = backed ? nothing : read_matrix(file["X"])

        # Layers
        adata.layers = haskey(file, "layers") ? read_dict_of_matrices(file["layers"]) : nothing

        return adata
    end

    function AnnData(x::AbstractMatrix{<:Number},
		     obs_names::Union{Vector{String}, Nothing}=nothing,
		     var_names::Union{Vector{String}, Nothing}=nothing,
		     )
        adata = new(nothing)
        adata.X = x

        n, d = size(x)[1:2]

        # Observations
        # TODO: check size
        adata.obs_names = isnothing(obs_names) ? string.(collect(1:n)) : obs_names

        # Variables
        # TODO: check size
        adata.var_names = isnothing(var_names) ? string.(collect(1:d)) : obs_names

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
        X = adata.file["X"]
        return X isa HDF5.Dataset ? X : SparseDataset(X)
    else
        return getfield(adata, s)
    end
end
