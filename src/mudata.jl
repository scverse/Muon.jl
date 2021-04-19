abstract type AbstractMuData end

mutable struct MuData <: AbstractMuData
    file::Union{HDF5.File, Nothing}
    mod::Dict{String, AnnData}

    obs::DataFrame
    obs_names::Index{<:AbstractString}
    obsm::StrAlignedMapping{Tuple{1 => 1}, MuData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, MuData}

    var::DataFrame
    var_names::Index{<:AbstractString}
    varm::StrAlignedMapping{Tuple{1 => 2}, MuData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, MuData}

    function MuData(file::HDF5.File, backed=true)
        mdata = new(backed ? file : nothing)

        # this needs to go first because it's used by size() and size()
        # is used for dimensionalty checks
        mdata.obs, obs_names = read_dataframe(file["obs"])
        mdata.var, var_names = read_dataframe(file["var"])
        mdata.obs_names, mdata.var_names = Index(obs_names), Index(var_names)

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

        _update_attr!(mdata, :var, 0, true)
        _update_attr!(mdata, :obs, 1)
        return mdata
    end

    function MuData(;
        mod::Union{AbstractDict{<:AbstractString, AnnData}, Nothing}=nothing,
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

        mdata.obs = isnothing(obs) ? DataFrame() : obs
        mdata.obs_names = isnothing(obs_names) ? Index(String[]) : Index(obs_names)
        mdata.var = isnothing(var) ? DataFrame() : var
        mdata.var_names = isnothing(var_names) ? Index(String[]) : Index(var_names)

        mdata.obsm = StrAlignedMapping{Tuple{1 => 1}}(mdata, obsm)
        mdata.varm = StrAlignedMapping{Tuple{1 => 2}}(mdata, varm)
        mdata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(mdata, obsp)
        mdata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(mdata, varp)

        _update_attr!(mdata, :var, 0, true)
        _update_attr!(mdata, :obs, 1)
        return mdata
    end
end

file(mu::MuData) = mu.file

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

function writeh5mu(filename::AbstractString, mudata::AbstractMuData)
    filename = abspath(filename)
    if file(mudata) === nothing || filename != HDF5.filename(file(mudata))
        hfile = h5open(filename, "w")
        try
            write(hfile, mudata)
        finally
            close(hfile)
        end
    else
        write(mudata)
    end
end

function Base.write(parent::Union{HDF5.File, HDF5.Group}, mudata::AbstractMuData)
    if parent === file(mudata)
        write(mudata)
    else
        g = create_group(parent, "mod")
        for (mod, adata) in mudata.mod
            write(g, mod, adata)
        end
        write_metadata(parent, mudata)
    end
end

function Base.write(mudata::AbstractMuData)
    if isnothing(file(mudata))
        throw("adata is not backed, need somewhere to write to")
    end
    for adata in values(mudata.mod)
        write(adata)
    end
    write_metadata(mudata.file, mudata)
end

function write_metadata(parent::Union{HDF5.File, HDF5.Group}, mudata::AbstractMuData)
    write_attr(parent, "obs", mudata.obs_names, shrink_attr(mudata, :obs))
    write_attr(parent, "obsm", mudata.obsm)
    write_attr(parent, "obsp", mudata.obsp)
    write_attr(parent, "var", mudata.var_names, shrink_attr(mudata, :var))
    write_attr(parent, "varm", mudata.varm)
    write_attr(parent, "varp", mudata.varp)
end

Base.size(mdata::AbstractMuData) = (length(mdata.obs_names), length(mdata.var_names))
Base.size(mdata::AbstractMuData, d::Integer) = size(mdata)[d]

Base.getindex(mdata::AbstractMuData, modality::Symbol) = mdata.mod[String(modality)]
Base.getindex(mdata::AbstractMuData, modality::AbstractString) = mdata.mod[modality]
function Base.getindex(
    mdata::AbstractMuData,
    I::Union{
        AbstractRange,
        Colon,
        AbstractVector{<:Integer},
        AbstractVector{<:AbstractString},
        Number,
        AbstractString,
    },
    J::Union{
        AbstractRange,
        Colon,
        AbstractVector{<:Integer},
        AbstractVector{<:AbstractString},
        Number,
        AbstractString,
    },
)
    @boundscheck checkbounds(mdata, I, J)
    i, j = convertidx(I, mdata.obs_names), convertidx(J, mdata.var_names)
    newmu = MuData(
        mod=Dict{String, AnnData}(
            k => ad[
                getadidx(I, mdata.obsm[k], mdata.obs_names),
                getadidx(J, mdata.varm[k], mdata.var_names),
            ] for (k, ad) in mdata.mod
        ),
        obs=isempty(mdata.obs) ? nothing : mdata.obs[i, :],
        obs_names=mdata.obs_names[i],
        var=isempty(mdata.var) ? nothing : mdata.var[j, :],
        var_names=mdata.var_names[j],
    )
    copy_subset(mdata.obsm, newmu.obsm, i, j)
    copy_subset(mdata.varm, newmu.varm, i, j)
    copy_subset(mdata.obsp, newmu.obsp, i, j)
    copy_subset(mdata.varp, newmu.varp, i, j)
    return newmu
end

getadidx(
    I::Colon,
    ref::AbstractVector{<:Unsigned},
    idx::AbstractIndex{<:AbstractString},
    reduce_memory=false,
) = I
function getadidx(
    I::Union{AbstractVector{<:Integer}, AbstractRange},
    ref::AbstractVector{<:Unsigned},
    idx::AbstractIndex{<:AbstractString},
    reduce_memory=false,
)
    J = filter(x -> x > 0x0, ref[I])
    if reduce_memory
        diffs = unique(J[2:end] .- J[1:(end - 1)])
        if length(diffs) == 1
            return diffs[1] == 1 ? (J[1]:J[end]) : (J[1]:diffs[1]:J[end])
        end
    end
    return J
end
getadidx(
    I::Union{AbstractString, AbstractVector{<:AbstractString}},
    ref::AbstractVector{<:Unsigned},
    idx::AbstractIndex{<:AbstractString},
    reduce_memory=false,
) = getadidx(idx[I, true], ref, idx)
getadidx(
    I::Number,
    ref::AbstractVector{<:Unsigned},
    idx::Index{<:AbstractString},
    reduce_memory=false,
) = getadidx([I], ref, idx)

function Base.show(io::IO, mdata::AbstractMuData)
    compact = get(io, :compact, false)
    repr = """$(typeof(mdata).name.name) object $(size(mdata)[1]) \u2715 $(size(mdata)[2])"""
    for (name, adata) in mdata.mod
        repr *= """\n\u2514 $(name)"""
        repr *= """\n  $(typeof(adata).name.name) object $(size(adata)[1]) \u2715 $(size(adata)[2])"""
    end
    print(io, repr)
end

function Base.show(io::IO, ::MIME"text/plain", mdata::AbstractMuData)
    show(io, mdata)
end

function _update_attr!(mdata::MuData, attr::Symbol, axis::Integer, join_common::Bool=false)
    globalcols = [
        col for
        col in names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
    ]
    globaldata = getproperty(mdata, attr)[!, globalcols]
    namesattr = Symbol(string(attr) * "_names")
    old_rownames = getproperty(mdata, namesattr)

    idxcol, rowcol, dupidxcol = find_unique_colnames(mdata, attr, 3)

    if join_common &&
       length(mdata.mod) > 1 &&
       !all(
           isdisjoint(getproperty(ad_i, namesattr), getproperty(ad_j, namesattr)) for
           (i, ad_i) in enumerate(values(mdata.mod)), (j, ad_j) in enumerate(values(mdata.mod)) if
           j > i
       )
        @warn "Cannot join columns with the same name because $(string(namesattr)) are intersecting."
        join_common = false
    end

    dupidx = Dict(mod => index_duplicates(getproperty(ad, namesattr)) for (mod, ad) in mdata.mod)
    for (mod, dups) in dupidx, (mod2, ad) in mdata.mod
        if mod != mod2 && any(
            unique(getproperty(mdata.mod[mod], namesattr)[dups .> 0]) .∈
            (getproperty(ad, namesattr),),
        )
            @warn "Duplicated $(string(namesattr)) should not be present in different modalities due to the ambiguitiy that leads to."
            break
        end
    end

    if !isempty(mdata.mod)
        if length(mdata.mod) > 1
            if join_common
                commoncols =
                    intersect((names(getproperty(ad, attr)) for ad in values(mdata.mod))...)
                globaldata = select(globaldata, Not(intersect(commoncols, names(globaldata))))
                dfs = (
                    insertcols!(
                        rename!(
                            x -> mod * ":" * x,
                            select(getproperty(ad, attr), Not(commoncols), copycols=false),
                        ),
                        idxcol => getproperty(ad, namesattr),
                        dupidxcol => dupidx[mod],
                        mod * ":" * rowcol => 1:length(getproperty(ad, namesattr)),
                    ) for (mod, ad) in mdata.mod
                )
                data_mod =
                    axis == 0 ? vcat(dfs..., cols=:union) :
                    outerjoin(dfs..., on=[idxcol, dupidxcol])

                data_common = vcat(
                    (
                        insertcols!(
                            select(getproperty(ad, attr), commoncols..., copycols=false),
                            idxcol => getproperty(ad, namesattr),
                            dupidxcol => dupidx[mod],
                        ) for (mod, ad) in mdata.mod
                    )...,
                )
                data_mod = leftjoin(data_mod, data_common, on=[idxcol, dupidxcol])
            else
                dfs = (
                    insertcols!(
                        rename!(
                            x -> mod * ":" * x,
                            select(getproperty(ad, attr), :, copycols=false),
                        ),
                        idxcol => getproperty(ad, namesattr),
                        dupidxcol => dupidx[mod],
                        mod * ":" * rowcol => 1:length(getproperty(ad, namesattr)),
                    ) for (mod, ad) in mdata.mod
                )
                data_mod =
                    axis == 0 ? vcat(dfs..., cols=:union) :
                    outerjoin(dfs..., on=[idxcol, dupidxcol])
            end
        elseif length(mdata.mod) == 1
            mod, ad = iterate(mdata.mod)[1]
            newdf = insertcols!(
                rename!(x -> mod * ":" * x, select(getproperty(ad, attr), :, copycols=false)),
                idxcol => getproperty(ad, namesattr),
                mod * ":" * rowcol => 1:length(getproperty(ad, namesattr)),
            )
        end
        data_mod = leftjoin(
            data_mod,
            insertcols!(
                globaldata,
                idxcol => old_rownames,
                dupidxcol => index_duplicates(old_rownames),
            ),
            on=[idxcol, dupidxcol],
        )
        rownames = data_mod[!, idxcol]
        select!(data_mod, Not([idxcol, dupidxcol]))

        try
            rownames = convert(Vector{nonmissingtype(eltype(rownames))}, rownames)
        catch e
            if e isa MethodError
                throw(
                    "New $(string(namesattr)) contain missing values. That should not happen, ever.",
                )
            else
                rethrow(e)
            end
        end
    else
        data_mod = DataFrame()
        rownames = Vector{String}()
    end

    setproperty!(mdata, attr, data_mod)
    setproperty!(mdata, namesattr, Index(rownames))

    mattr = Symbol(string(attr) * "m")
    for (mod, ad) in mdata.mod
        colname = mod * ":" * rowcol
        adindices = data_mod[!, colname]
        select!(data_mod, Not(colname))
        replace!(adindices, missing => 0)
        getproperty(mdata, mattr)[mod] =
            convert(Vector{minimum_unsigned_type_for_n(maximum(adindices))}, adindices)
    end

    keep_index = rownames .∈ (old_rownames,)
    @inbounds if sum(keep_index) != length(old_rownames)
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

update_obs!(mdata::MuData) = _update_attr!(mdata, :obs, 1)
update_var!(mdata::MuData) = _update_attr!(mdata, :var, 0, true)
function update!(mdata::MuData)
    update_obs!(mdata)
    update_var!(mdata)
end

function shrink_attr(mdata::AbstractMuData, attr::Symbol)
    globalcols = [
        col for
        col in names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
    ]
    return disallowmissing!(select(getproperty(mdata, attr), globalcols))
end

struct MuDataView{Ti, Tj} <: AbstractMuData
    parent::MuData
    I::Ti
    J::Tj

    mod::FrozenDict{String, <:AnnDataView}
    obs::SubDataFrame
    obs_names::SubIndex{<:AbstractString}
    obsm::StrAlignedMappingView{Tuple{1 => 1}}
    obsp::StrAlignedMappingView{Tuple{1 => 1, 2 => 1}}

    var::SubDataFrame
    var_names::SubIndex{<:AbstractString}
    varm::StrAlignedMappingView{Tuple{1 => 2}}
    varp::StrAlignedMappingView{Tuple{1 => 2, 2 => 2}}
end

function Base.view(mu::MuData, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j = convertidx(I, mu.obs_names), convertidx(J, mu.var_names)
    mod = Dict(
        m => view(
            ad,
            getadidx(i, mu.obsm[m], mu.obs_names, true),
            getadidx(j, mu.obsm[m], mu.var_names, true),
        ) for (m, ad) in mu.mod
    )
    return MuDataView(
        mu,
        i,
        j,
        FrozenDict(mod),
        view(mu.obs, i, :),
        view(mu.obs_names, i),
        view(mu.obsm, i),
        view(mu.obsp, i, i),
        view(mu.var, j, :),
        view(mu.var_names, j),
        view(mu.varm, j),
        view(mu.varp, j, j),
    )
end
function Base.view(mu::MuDataView, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j =
        Base.reindex(parentindices(mu), (convertidx(I, mu.obs_names), convertidx(J, mu.var_names)))
    return view(parent(mu), i, j)
end

Base.parent(mu::MuDataView) = mu.parent
Base.parentindices(mu::MuDataView) = (mu.I, mu.J)
file(mu::MuDataView) = file(parent(mu))
