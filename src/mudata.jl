mutable struct MuData
    file::Union{HDF5.File, Nothing}
    mod::Dict{String, AnnData}

    obs::DataFrame
    obs_names::AbstractVector{<:AbstractString}
    obsm::StrAlignedMapping{Tuple{1 => 1}, MuData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, MuData}

    var::DataFrame
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
        mdata.obsm = StrAlignedMapping{Tuple{1 => 1}}(
            mdata,
            haskey(file, "obsm") ? read_dict_of_mixed(file["obsm"]) : nothing,
        )
        mdata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(
            mdata,
            haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing,
        )

        # Variables
        mdata.varm = StrAlignedMapping{Tuple{1 => 2}}(
            mdata,
            haskey(file, "varm") ? read_dict_of_mixed(file["varm"]) : nothing,
        )
        mdata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(
            mdata,
            haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing,
        )

        # Modalities
        mdata.mod = Dict{String, AnnData}()
        mods = HDF5.keys(file["mod"])
        for modality in mods
            mdata.mod[modality] = AnnData(file["mod"][modality], backed)
        end

        update_attr!(mdata, :var)
        update_attr!(mdata, :obs)
        return mdata
    end

    function MuData(;
        mod::AbstractDict{<:AbstractString, AnnData}=nothing,
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
        varp::Union{AbstractDict{<:AbstractString, AbstractMatrix <: Number}, Nothing}=nothing,
    )
        mdata = new(nothing, Dict{String, AnnData}())
        if !isnothing(mod)
            merge!(mdata.mod, mod)
        end

        # TODO: dimension checking. This needs merging dimensions of all the AnnDatas based on var_names/obs_names
        mdata.obs = isnothing(obs) ? DataFrame() : obs
        mdata.obs_names = isnothing(obs_names) ? String[] : obs_names
        mdata.var = isnothing(var) ? DataFrame() : var
        mdata.var_names = isnothing(var_names) ? String[] : var_names

        mdata.obsm = StrAlignedMapping{Tuple{1 => 1}}(mdata, obsm)
        mdata.varm = StrAlignedMapping{Tuple{1 => 2}}(mdata, varm)
        mdata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(mdata, obsp)
        mdata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(mdata, varp)

        update_attr!(mdata, :var)
        update_attr!(mdata, :obs)
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
    write_attr(parent, "obs", mudata.obs_names, shrink_attr(mudata, :obs))
    write_attr(parent, "obsm", mudata.obsm)
    write_attr(parent, "obsp", mudata.obsp)
    write_attr(parent, "var", mudata.var_names, shrink_attr(mudata, :var))
    write_attr(parent, "varm", mudata.varm)
    write_attr(parent, "varp", mudata.varp)
end

Base.size(mdata::MuData) = (length(mdata.obs_names), length(mdata.var_names))
Base.size(mdata::MuData, d::Integer) = size(mdata)[d]

Base.getindex(mdata::MuData, modality::Symbol) = mdata.mod[String(modality)]
Base.getindex(mdata::MuData, modality::AbstractString) = mdata.mod[modality]
function Base.getindex(
    mdata::MuData,
    I::Union{AbstractUnitRange, Colon, Vector{<:Integer}},
    J::Union{AbstractUnitRange, Colon, Vector{<:Integer}},
)
    newmu = MuData(
        mod=Dict{String, AnnData}(k => ad[getadidx(I, mdata.obsm[k]), getadidx(J, mdata.varm[k])] for (k, ad) in mdata.mod),
        obs=isempty(mdata.obs) ? nothing : mdata.obs[I, :],
        obs_names=mdata.obs_names[I],
        var=isempty(mdata.var) ? nothing : mdata.var[J, :],
        var_names=mdata.var_names[J],
    )
    copy_subset(mdata.obsm, newmu.obsm, I, J)
    copy_subset(mdata.varm, newmu.varm, I, J)
    copy_subset(mdata.obsp, newmu.obsp, I, J)
    copy_subset(mdata.varp, newmu.varp, I, J)
    return newmu
end

getadidx(idx::Colon, ref::AbstractVector{Bool}) = idx
function getadidx(idx::AbstractUnitRange, ref::AbstractVector{Bool})
    allidx = findall(ref)
    start = findfirst(x -> x ≥ first(idx), allidx)
    stop = findlast(x -> x ≤ last(idx), allidx)
    return start:stop
end
function getadidx(idx::AbstractVector{<:Integer}, ref::AbstractVector{Bool})
    allidx = findall(ref)
    revmapping = zeros(UInt32, length(ref))
    @inbounds revmapping[allidx] .= 1:length(allidx)
    i = 0
    @inbounds for id in idx
        if revmapping(id) > 0
            i += 1
            allidx[i] = revmapping[id]
        end
    end
    resize!(allidx, i)
    return allidx
end


function Base.show(io::IO, mdata::MuData)
    compact = get(io, :compact, false)
    print(io, """MuData object $(size(mdata)[1]) \u2715 $(size(mdata)[2])""")
end

function Base.show(io::IO, ::MIME"text/plain", mdata::MuData)
    show(io, mdata)
end

isbacked(mdata::MuData) = mdata.file !== nothing

function update_attr!(mdata::MuData, attr::Symbol)
    globalcols = [
        col for
        col in names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
    ]
    globaldata = getproperty(mdata, attr)[!, globalcols]
    namesattr = Symbol(string(attr) * "_names")
    old_rownames = getproperty(mdata, namesattr)

    idxcol = find_unique_rownames_colname(mdata, attr)

    try
        newdf = reduce((
            mod => insertcols!(getproperty(ad, attr), idxcol => getproperty(ad, namesattr)) for
            (mod, ad) in mdata.mod
        )) do df1, df2
            outerjoin(
                df1.second,
                df2.second,
                on=idxcol,
                renamecols=((x -> df1.first * ":" * x) => (x -> df2.first * ":" * x)),
            )
        end
    finally
        for ad in values(mdata.mod)
            df = getproperty(ad, attr)
            select!(df, 1:(ncol(df) - 1)) # delete the rownames column
        end
    end

    newdf =
        leftjoin(newdf, insertcols!(globaldata, idxcol => getproperty(mdata, namesattr)), on=idxcol)
    rownames = newdf[!, idxcol]
    try
        rownames = convert(Vector{nonmissingtype(eltype(rownames))}, rownames)
    catch e
        if e isa MethodError
            throw("New $(string(namesattr)) contain missing values. That should not happen, ever.")
        else
            rethrow(e)
        end
    end

    select!(newdf, Not(idxcol))
    setproperty!(mdata, attr, newdf)
    setproperty!(mdata, namesattr, rownames)

    mattr = Symbol(string(attr) * "m")
    for (mod, ad) in mdata.mod
        getproperty(mdata, mattr)[mod] = rownames .∈ (Set(getproperty(ad, namesattr)),)
    end

    keep_index = rownames .∈ (Set(old_rownames),)
    if sum(keep_index) != length(old_rownames)
        for (k, v) in getproperty(mdata, mattr)
            if k ∉ keys(mdata.mod)
                getproperty(mdata, mattr)[k] = v[keep_index, :]
            end
        end

        pattr = Symbol(string(attr) * "p")
        for (k, v) in getproperty(mdata, pattr)
            if k ∉ keys(mdata.mod)
                getproperty(mdata, pattr)[k] = v[keep_index, keep_index]
            end
        end
    end

    nothing
end

function shrink_attr(mdata::MuData, attr::Symbol)
    globalcols = [
        col for
        col in names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
    ]
    return disallowmissing!(select(getproperty(mdata, attr), globalcols))
end
