mutable struct MuData
    file::Union{HDF5.File, Nothing}
    mod::Union{Dict{String, AnnData}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{Vector{String}, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}
    obsp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{Vector{String}, Nothing}
    varm::Union{Dict{String, Any}, Nothing}
    varp::Union{Dict{String, AbstractMatrix{<:Number}}, Nothing}

    function MuData(file::HDF5.File, backed=true)
        mdata = new(backed ? file : nothing)

        # Observations
        mdata.obs, mdata.obs_names = read_dataframe(file["obs"])
        mdata.obsm = "obsm" ∈ keys(file) ? read(file["obsm"]) : nothing
        mdata.obsp = "obsp" ∈ keys(file) ? read_dict_of_matrices(file["obsp"]) : nothing

        # Variables
        mdata.var, mdata.var_names = read_dataframe(file["var"])
        mdata.varm = "varm" ∈ keys(file) ? read(file["varm"]) : nothing
        mdata.varp = "varp" ∈ keys(file) ? read_dict_of_matrices(file["varp"]) : nothing

        # Modalities
        mdata.mod = Dict{String, AnnData}()
        mods = HDF5.keys(file["mod"])
        for modality in mods
            mdata.mod[modality] = AnnData(file["mod"][modality], backed)
        end
        return mdata
    end
end

function readh5mu(filename::AbstractString; backed=true)
    if backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    mdata = MuData(fid, backed)
    return mdata
end

function writeh5mu(filename::AbstractString, mudata::MuData)
    file = h5open(filename, "w")
    try
        write(file, mudata)
    finally
        close(file)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, mudata::MuData)
    g = create_group(parent, "mod")
    for (mod, adata) in mudata.mod
        write(g, mod, adata)
    end
    write(parent, "obs", mudata.obs_names, mudata.obs)
    write(parent, "obsm", mudata.obsm)
    write(parent, "obsp", mudata.obsp)
    write(parent, "var", mudata.var_names, mudata.var)
    write(parent, "varm", mudata.varm)
    write(parent, "varp", mudata.varp)
end

Base.size(mdata::MuData) =
    (length(mdata.obs_names), length(mdata.var_names))

Base.getindex(mdata::MuData, modality::Symbol) = mdata.mod[String(modality)]
Base.getindex(mdata::MuData, modality::AbstractString) = mdata.mod[modality]

function Base.show(io::IO, mdata::MuData)
    compact = get(io, :compact, false)
    print(io, """MuData object $(size(mdata)[1]) \u2715 $(size(mdata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", mdata::MuData)
    show(io, mdata)
end

isbacked(mdata::MuData) = mdata.file !== nothing
