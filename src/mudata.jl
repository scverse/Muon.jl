mutable struct MuData
    file::Union{HDF5.File, Nothing}
    mod::Dict{String, AnnData}

    obs::Union{DataFrame, Nothing}
    obs_names::AbstractVector{<:AbstractString}
    obsm::StrAlignedMapping{Tuple{1 => 1}, MuData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, MuData}

    var::Union{DataFrame, Nothing}
    var_names::AbstractVector{<:AbstractString}
    varm::StrAlignedMapping{Tuple{1 => 2}, MuData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, MuData}

    function MuData(file::HDF5.File, backed=true)
        mdata = new(backed ? file : nothing)

        # this needs to go first because it's used by size() and size()
        # is used for dimensionalty checks
        mdata.obs, mdata.obs_names = read_dataframe(file["obs"])
        mdata.var, mdata.var_names = read_dataframe(file["var"])

        # Observations
        mdata.obsm = StrAlignedMapping{Tuple{1 => 1}}(mdata, haskey(file, "obsm") ? read_dict_of_mixed(file["obsm"]) : nothing)
        mdata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(mdata, haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing)

        # Variables
        mdata.varm = StrAlignedMapping{Tuple{1 => 2}}(mdata, haskey(file, "varm") ? read_dict_of_mixed(file["varm"]) : nothing)
        mdata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(mdata, haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing)

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
    filename = abspath(filename) # this gets stored in the HDF5 objects for backed datasets
    if !backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    local mdata
    try
        mdata = MuData(fid, backed)
    finally
        if !backed
            close(fid)
        end
    end
    return mdata
end

function writeh5mu(filename::AbstractString, mudata::MuData)
    filename = abspath(filename)
    if mudata.file === nothing || filename != HDF5.filename(mudata.file)
        file = h5open(filename, "w")
        try
            write(file, mudata)
        finally
            close(file)
        end
    else
        write(mudata)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, mudata::MuData)
    if parent === mudata.file
        write(mudata)
    else
        g = create_group(parent, "mod")
        for (mod, adata) in mudata.mod
            write(g, mod, adata)
        end
        write_metadata(parent, mudata)
    end
end

function Base.write(mudata::MuData)
    if mudata.file === nothing
        throw("adata is not backed, need somewhere to write to")
    end
    for adata in values(mudata.mod)
        write(adata)
    end
    write_metadata(mudata.file, mudata)
end

function write_metadata(parent::Union{HDF5.File, HDF5.Group}, mudata::MuData)
    write_attr(parent, "obs", mudata.obs_names, mudata.obs)
    write_attr(parent, "obsm", mudata.obsm)
    write_attr(parent, "obsp", mudata.obsp)
    write_attr(parent, "var", mudata.var_names, mudata.var)
    write_attr(parent, "varm", mudata.varm)
    write_attr(parent, "varp", mudata.varp)
end

Base.size(mdata::MuData) = (length(mdata.obs_names), length(mdata.var_names))
Base.size(mdata::MuData, d::Integer) = size(mdata)[d]

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
