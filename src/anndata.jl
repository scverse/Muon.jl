mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group}

    X::AbstractArray{<:Number, 2}
    layers::Union{Dict{String, Any}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{Vector{String}, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{Vector{String}, Nothing}
    varm::Union{Dict{String, Any}, Nothing}

    function AnnData(file::Union{HDF5.File, HDF5.Group})
        adata = new(file)

        # Observations
        adata.obs, adata.obs_names = read_dataframe(file["obs"])
        adata.obsm = read(file["obsm"])

        # Variables
        adata.var, adata.var_names = read_dataframe(file["var"])
        adata.varm = "varm" ∈ keys(file) ? read(file["varm"]) : nothing

        # X
        adata.X = read_matrix(file["X"])

        # Layers
        if "layers" in HDF5.keys(file)
            adata.layers = Dict{String, Any}()
            layers = HDF5.keys(file["layers"])
            for layer in layers
                adata.layers[layer] = read_matrix(file["layers"][layer])
            end
        end

        adata
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
    write(parent, "var", adata.var_names, adata.var)
    write(parent, "varm", adata.varm)
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
