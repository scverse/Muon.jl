mutable struct MuData
    file::HDF5.File
    mod::Union{Dict{String, AnnData}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{Vector{String}, Nothing}
    obsm::Union{Dict{String, Any}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{Vector{String}, Nothing}
    varm::Union{Dict{String, Any}, Nothing}

    function MuData(file::HDF5.File)
        mdata = new(file)

        # Observations
        mdata.obs, mdata.obs_names = read_dataframe(file["obs"])
        mdata.obsm = read(file["obsm"])

        # Variables
        mdata.var, mdata.var_names = read_dataframe(file["var"])
        mdata.varm = read(file["varm"])

        # Modalities
        mdata.mod = Dict{String, AnnData}()
        mods = HDF5.keys(mdata.file["mod"])
        for modality in mods
            adata = AnnData(mdata.file["mod"][modality])
            mdata.mod[modality] = adata
        end

        mdata
    end
end

function readh5mu(filename::AbstractString; backed=true)
    if backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    mdata = MuData(fid)
    return mdata
end

Base.size(mdata::MuData) =
    (size(mdata.file["obs"]["_index"])[1], size(mdata.file["var"]["_index"])[1])

Base.getindex(mdata::MuData, modality::Symbol) = mdata.mod[String(modality)]
Base.getindex(mdata::MuData, modality::AbstractString) = mdata.mod[modality]

function Base.show(io::IO, mdata::MuData)
    compact = get(io, :compact, false)
    print(io, """MuData object $(size(mdata)[1]) \u2715 $(size(mdata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", mdata::MuData)
    show(io, mdata)
end
