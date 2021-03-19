mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group}

    X::AbstractMatrix{<:Number}
    layers::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{Vector{String}, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}
    obsp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{Vector{String}, Nothing}
    varm::Union{Dict{String, Any}, Nothing}
    varp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    function AnnData(file::Union{HDF5.File, HDF5.Group})
        adata = new(file)

        # Observations
        adata.obs, adata.obs_names = read_dataframe(file["obs"])
        adata.obsm = "obsm" ∈ keys(file) ? read(file["obsm"]) : nothing
        adata.obsp = "obsp" ∈ keys(file) ? read_dict_of_matrices(file["obsp"]) : nothing

        # Variables
        adata.var, adata.var_names = read_dataframe(file["var"])
        adata.varm = "varm" ∈ keys(file) ? read(file["varm"]) : nothing
        adata.varp = "varp" ∈ keys(file) ? read_dict_of_matrices(file["varp"]) : nothing


        # X
        adata.X = read_matrix(file["X"])

        # Layers
        adata.layers = "layers" ∈ keys(file) ? read_dict_of_matrices(file["layers"]) : nothing

        return adata
    end
end

function readh5ad(filename::AbstractString; backed=true)
    if backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    adata = AnnData(fid)
    return adata
end

function writeh5ad(filename::AbstractString, adata::AnnData)
    file = h5open(filename, "w")
    try
        write(file, adata)
    finally
        close(file)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, adata::AnnData)
    g = create_group(parent, name)
    write(g, adata)
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, adata::AnnData)
    write(parent, "X", adata.X)
    write(parent, "layers", adata.layers)
    write(parent, "obs", adata.obs_names, adata.obs)
    write(parent, "obsm", adata.obsm)
    write(parent, "obsp", adata.obsp)
    write(parent, "var", adata.var_names, adata.var)
    write(parent, "varm", adata.varm)
    write(parent, "varp", adata.varp)
end

Base.size(adata::AnnData) =
    (size(adata.file["obs"]["_index"])[1], size(adata.file["var"]["_index"])[1])

function Base.show(io::IO, adata::AnnData)
    compact = get(io, :compact, false)
    print(io, """AnnData object $(size(adata)[1]) \u2715 $(size(adata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", adata::AnnData)
    show(io, adata)
end
