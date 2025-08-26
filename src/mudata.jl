abstract type AbstractMuData end

mutable struct MuData <: AbstractMuData
    file::Union{HDF5.File, ZGroup, Nothing}
    mod::OrderedDict{String, AnnData}

    obs::DataFrame
    obs_names::Index{<:AbstractString}
    obsm::StrAlignedMapping{Tuple{1 => 1}, MuData}
    obsp::StrAlignedMapping{Tuple{1 => 1, 2 => 1}, MuData}
    obsmap::StrAlignedMapping{Tuple{1 => 1}, MuData}

    var::DataFrame
    var_names::Index{<:AbstractString}
    varm::StrAlignedMapping{Tuple{1 => 2}, MuData}
    varp::StrAlignedMapping{Tuple{1 => 2, 2 => 2}, MuData}
    varmap::StrAlignedMapping{Tuple{1 => 2}, MuData}

    uns::Dict{<:AbstractString, <:Any}

    function MuData(file::Union{HDF5.File, HDF5.Group, ZGroup}, backed=false, checkversion=true)
        if checkversion
            attrs = attributes(file)
            if !haskey(attrs, "encoding-type")
                @warn "This file was not created by muon, we can't guarantee that everything will work correctly"
            elseif attrs["encoding-type"] != "MuData"
                error("This file does not appear to hold a MuData object")
            end
        end

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
        mdata.obsmap = StrAlignedMapping{Tuple{1 => 1}}(
            mdata,
            haskey(file, "obsmap") ? read_dict_of_matrices(file["obsmap"]) : nothing,
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
        mdata.varmap = StrAlignedMapping{Tuple{1 => 2}}(
            mdata,
            haskey(file, "varmap") ? read_dict_of_matrices(file["varmap"]) : nothing,
        )

        # unstructured
        mdata.uns =
            haskey(file, "uns") ? read_dict_of_mixed(file["uns"], separate_index=false) :
            Dict{String, Any}()

        # Modalities
        mdata.mod = OrderedDict{String, AnnData}()
        mods = HDF5.keys(file["mod"])

        modattr = attributes(file["mod"])
        if haskey(modattr, "mod-order")
            mod_order = read_attribute(file["mod"], "mod-order")
            if issubset(mods, mod_order)
                for modality ∈ mod_order
                    mdata.mod[modality] = AnnData(file["mod"][modality], backed, checkversion)
                end
                return update!(mdata)
            else
                @warn "Modality order attribute has some of the modalities missing and will be ignored"
            end
        end

        # no mod-order or not all modalities are in mod-order (then mod-order is ignored) 
        for modality ∈ mods
            mdata.mod[modality] = AnnData(file["mod"][modality], backed, checkversion)
        end
        return update!(mdata)
    end

    function MuData(;
        mod::Union{AbstractDict{<:AbstractString, AnnData}, Nothing}=nothing,
        obs::Union{DataFrame, Nothing}=nothing,
        obs_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        var::Union{DataFrame, Nothing}=nothing,
        var_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        obsm::Union{
            AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}},
            Nothing,
        }=nothing,
        varm::Union{
            AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}},
            Nothing,
        }=nothing,
        obsp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        varp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        obsmap::Union{AbstractDict{<:AbstractString, <:AbstractVector{<:Integer}}, Nothing}=nothing,
        varmap::Union{AbstractDict{<:AbstractString, <:AbstractVector{<:Integer}}, Nothing}=nothing,
        uns::Union{AbstractDict{<:AbstractString, <:Any}, Nothing}=nothing,
        do_update=true,
    )
        mdata = new(nothing, OrderedDict{String, AnnData}())
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
        mdata.obsmap = StrAlignedMapping{Tuple{1 => 1}}(mdata, obsmap)
        mdata.varmap = StrAlignedMapping{Tuple{1 => 2}}(mdata, varmap)
        mdata.uns = isnothing(uns) ? Dict{String, Any}() : uns

        if do_update
            update!(mdata)
        end
        return mdata
    end
end

file(mu::MuData) = mu.file

function readh5mu(filename::AbstractString; backed=false)
    filename = abspath(filename) # this gets stored in the HDF5 objects for backed datasets
    if String(read(filename, 6)) != "MuData"
        if HDF5.ishdf5(filename)
            @warn "The HDF5 file was not created by muon, we can't guarantee that everything will work correctly"
        else
            error("The file is not an HDF5 file")
        end
    end
    if !backed
        fid = h5open(filename, "r")
    else
        fid = h5open(filename, "r+")
    end
    local mdata
    try
        mdata = MuData(fid, backed, false)
    catch e
        close(fid)
        rethrow()
    end
    if !backed
        close(fid)
    end
    return mdata
end

function readzarrmu(filename::AbstractString; backed=false)
    filename = abspath(filename) # this gets stored in the Zarr objects for backed datasets
    if !backed
        fid = zopen(filename, "r")
    else
        fid = zopen(filename, "r+")
    end
    local mdata
    try
        mdata = MuData(fid, backed, true)
    catch e
        close(fid)
        rethrow()
    end
    if !backed
        close(fid)
    end
    return mdata
end

function writeh5mu(filename::AbstractString, mudata::AbstractMuData; compress::UInt8=0x9)
    filename = abspath(filename)
    if isnothing(file(mudata)) || filename != HDF5.filename(file(mudata))
        hfile = h5open(filename, "w", userblock=512)
        try
            write(hfile, mudata, compress=compress)
            close(hfile)
            hfile = open(filename, "r+")
            write(
                hfile,
                "MuData (format-version=$MUDATAVERSION;creator=$NAME;creator-version=$VERSION)",
            )
        finally
            close(hfile)
        end
    else
        write(mudata, compress=compress)
    end
    return nothing
end

function writezarrmu(filename::AbstractString, mudata::AbstractMuData; compress::UInt8=0x9)
    filename = abspath(filename)
    if isnothing(file(mudata)) || filename != zarr_filename(file(mudata))
        rm(filename, force=true, recursive=true)
        zfile = zgroup(filename)
        write(zfile, mudata, compress=compress)
    else
        write(mudata, compress=compress)
    end
    return nothing
end

function Base.write(parent::Group, mudata::AbstractMuData; compress::UInt8=0x9)
    write_attribute(parent, "encoding-type", "MuData")
    write_attribute(parent, "encoding-version", string(MUDATAVERSION))
    write_attribute(parent, "encoder", NAME)
    write_attribute(parent, "encoder-version", string(VERSION))
    if parent === file(mudata)
        write(mudata, compress=compress)
    else
        g = create_group(parent, "mod")
        for (mod, adata) ∈ mudata.mod
            write(g, mod, adata, compress=compress)
        end
        write_metadata(parent, mudata, compress=compress)
        write_attribute(g, "mod-order", collect(keys(mudata.mod)))
    end
end

function Base.write(mudata::AbstractMuData; compress::UInt8=0x9)
    if isnothing(file(mudata))
        error("mudata is not backed, need somewhere to write to")
    end
    for adata ∈ values(mudata.mod)
        write(adata, compress=compress)
    end
    write_metadata(mudata.file, mudata, compress=compress)
end

function write_metadata(parent::Group, mudata::AbstractMuData; compress::UInt8=0x9)
    write_attr(parent, "obs", shrink_attr(mudata, :obs), index=mudata.obs_names, compress=compress)
    write_attr(parent, "obsm", mudata.obsm, index=mudata.obs_names, compress=compress)
    write_attr(parent, "obsp", mudata.obsp, compress=compress)
    write_attr(parent, "obsmap", mudata.obsmap, compress=compress)
    write_attr(parent, "var", shrink_attr(mudata, :var), index=mudata.var_names, compress=compress)
    write_attr(parent, "varm", mudata.varm, index=mudata.var_names, compress=compress)
    write_attr(parent, "varp", mudata.varp, compress=compress)
    write_attr(parent, "varmap", mudata.varmap, compress=compress)
    write_attr(parent, "uns", mudata.uns, compress=compress)
end

# FileIO support
load(f::File{format"h5mu"}; backed::Bool=false) = readh5mu(filename(f), backed=backed)
save(f::File{format"h5mu"}, data::AbstractMuData; compress::UInt8=0x9) =
    writeh5mu(filename(f), data, compress=compress)

Base.size(mdata::AbstractMuData) = (length(mdata.obs_names), length(mdata.var_names))
Base.size(mdata::AbstractMuData, d::Integer) = size(mdata)[d]

Base.getindex(mdata::AbstractMuData, modality::Symbol) = mdata.mod[string(modality)]
Base.getindex(mdata::AbstractMuData, modality::AbstractString) = mdata.mod[modality]
Base.setindex!(mdata::MuData, ad::AnnData, key::AbstractString) = setindex!(mdata.mod, ad, key)
Base.setindex!(mdata::MuData, ad::AnnData, key::Symbol) = setindex!(mdata.mod, ad, string(key))
function Base.getindex(
    mdata::MuData,
    I::Union{
        OrdinalRange,
        Colon,
        AbstractVector{<:Integer},
        AbstractVector{<:AbstractString},
        Number,
        AbstractString,
    },
    J::Union{
        OrdinalRange,
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
        mod=OrderedDict{String, AnnData}(
            k => ad[
                getadidx(i, reshape(mdata.obsmap[k], :), mdata.obs_names),
                getadidx(j, reshape(mdata.varmap[k], :), mdata.var_names),
            ] for (k, ad) ∈ mdata.mod
        ),
        obs=isempty(mdata.obs) ? nothing : mdata.obs[i, :],
        obs_names=mdata.obs_names[i],
        var=isempty(mdata.var) ? nothing : mdata.var[j, :],
        var_names=mdata.var_names[j],
        do_update=false,
    )
    copy_subset(mdata.obsm, newmu.obsm, i, j)
    copy_subset(mdata.varm, newmu.varm, i, j)
    copy_subset(mdata.obsp, newmu.obsp, i, j)
    copy_subset(mdata.varp, newmu.varp, i, j)
    copy_subset(mdata.obsmap, newmu.obsmap, i, j)
    copy_subset(mdata.varmap, newmu.varmap, i, j)

    for mod ∈ keys(mdata.mod)
        adjustmap!(reshape(newmu.obsmap[mod], :), i)
        adjustmap!(reshape(newmu.varmap[mod], :), j)
    end
    return newmu
end

getadidx(
    I::Colon,
    ref::AbstractVector{<:Unsigned},
    idx::AbstractIndex{<:AbstractString},
    reduce_memory=false,
) = I
function getadidx(
    I::Union{AbstractVector{<:Integer}, OrdinalRange},
    ref::AbstractVector{<:Unsigned},
    idx::AbstractIndex{<:AbstractString},
    reduce_memory=false,
)
    J = filter(>(0x0), ref[I])
    if reduce_memory && length(J) > 0
        diff = J[end] - J[1]
        if abs(diff) + 1 == length(J)
            return diff >= 0 ? (J[1]:J[end]) : (J[end]:-1:J[1])
        end
        diffs = unique(J[2:end] .- J[1:(end - 1)])
        if length(diffs) == 1
            return diffs[1] == 1 ? (J[1]:J[end]) : (J[1]:diffs[1]:J[end])
        end
    end
    return J
end
getadidx(
    I::Number,
    ref::AbstractVector{<:Unsigned},
    idx::Index{<:AbstractString},
    reduce_memory=false,
) = getadidx([I], ref, idx, reduce_memory)

adjustmap!(map::AbstractVector{<:Unsigned}, I::Colon) = map
function adjustmap!(
    map::AbstractVector{<:Unsigned},
    I::Union{Integer, AbstractVector{<:Integer}, OrdinalRange},
)
    nz = findall(map .> 0x0)
    map[nz] .= 1:length(nz)
    return map
end

function Base.show(io::IO, mdata::AbstractMuData)
    compact = get(io, :compact, false)
    repr = """$(typeof(mdata).name.name) object $(size(mdata)[1]) \u2715 $(size(mdata)[2])"""
    for (name, adata) ∈ mdata.mod
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
        col ∈ names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
    ]
    globaldata = getproperty(mdata, attr)[!, globalcols]
    namesattr = Symbol(string(attr) * "_names")
    mattr = Symbol(string(attr) * "m")
    mapattr = Symbol(string(attr) * "map")
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

    dupidx = Dict(mod => index_duplicates(getproperty(ad, namesattr)) for (mod, ad) ∈ mdata.mod)
    old_dupidx = index_duplicates(old_rownames)
    for (mod, dups) ∈ dupidx, (mod2, ad) ∈ mdata.mod
        if mod != mod2 && any(
            unique(getproperty(mdata.mod[mod], namesattr)[dups .> 0]) .∈
            (getproperty(ad, namesattr),),
        )
            @warn "Duplicated $(string(namesattr)) should not be present in different modalities due to the ambiguity that leads to."
            break
        end
    end

    if !isempty(mdata.mod)
        if length(mdata.mod) > 1
            if join_common
                commoncols = intersect((names(getproperty(ad, attr)) for ad ∈ values(mdata.mod))...)
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
                    ) for (mod, ad) ∈ mdata.mod
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
                        ) for (mod, ad) ∈ mdata.mod
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
                    ) for (mod, ad) ∈ mdata.mod
                )
                data_mod =
                    axis == 0 ? vcat(dfs..., cols=:union) :
                    outerjoin(dfs..., on=[idxcol, dupidxcol])
            end

            # reorder based on individual dataframes
            if axis == 1
                newidx = DataFrame(union([zip(df[:, idxcol], df[:, dupidxcol]) for df ∈ dfs]...))
                rename!(newidx, [idxcol, dupidxcol])
                if size(globaldata, 1) > 0
                    mask = newidx[!, idxcol] .∈ (old_rownames,)
                    newidx = vcat(newidx[mask, :], newidx[.~mask, :])
                end
                newidx[!, rowcol] = 1:size(newidx, 1)
                oldsize = size(data_mod, 1)
                data_mod = innerjoin(data_mod, newidx, on=[idxcol, dupidxcol])
                if oldsize != size(data_mod, 1)
                    error(
                        "Something went wrong when reordering the global data frame (global data frame had $oldsize rows, but only $(size(data_mod, 1)) rows after reordering). Please report a bug.",
                    )
                end
                data_mod = select!(data_mod[sortperm(data_mod[!, rowcol]), :], Not(rowcol))
            end
        elseif length(mdata.mod) == 1
            mod, ad = iterate(mdata.mod)[1]
            data_mod = insertcols!(
                rename!(x -> mod * ":" * x, select(getproperty(ad, attr), :, copycols=false)),
                idxcol => getproperty(ad, namesattr),
                dupidxcol => 0,
                mod * ":" * rowcol => 1:length(getproperty(ad, namesattr)),
            )
        end

        # this occurs when join_common=true and we already have a global data frame, e.g. after reading from HDF5
        if join_common
            sharedcols = intersect(names(data_mod), names(globaldata))
            rename!(globaldata, [col => "global:$col" for col ∈ sharedcols])
        end

        globaljoincols = Vector{String}()
        for mod ∈ intersect(keys(getproperty(mdata, mapattr)), keys(mdata.mod))
            colname = mod * ":" * rowcol
            globaldata[!, colname] = reshape(getproperty(mdata, mapattr)[mod], :)
            push!(globaljoincols, colname)
        end
        for col ∈ globaljoincols
            data_mod[!, col] = coalesce.(data_mod[!, col], 0x0)
        end

        if length(globaljoincols) == 0 && size(globaldata, 2) > 1 && any(old_dupidx .> 0)
            @warn "$namesattr is not unique, global $attr is present, and $mapattr is empty. The update() is not well-defined, verify if global $attr map to the correct modality-specific $attr."
            insertcols!(globaldata, dupidxcol => old_dupidx)
            data_mod[!, dupidxcol] = index_duplicates(data_mod[!, idxcol])
            push!(globaljoincols, dupidxcol)
        end

        data_mod = leftjoin(
            data_mod,
            insertcols!(globaldata, idxcol => old_rownames),
            on=[idxcol, globaljoincols...],
        )
        rownames = data_mod[!, idxcol]
        select!(data_mod, Not([idxcol, dupidxcol]))

        if join_common
            for col ∈ sharedcols
                gcol = "global:$col"
                if data_mod[col] == data_mod[gcol]
                    select!(data_mod, Not(gcol))
                else
                    @warn "Column $col was present in $attr but is also a common column in all modalities, and their contents differ. $attr.$col was renamed to $attr.$gcol."
                end
            end
        end

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

    setproperty!(mdata, namesattr, Index(rownames))
    for (mod, ad) ∈ mdata.mod
        colname = mod * ":" * rowcol
        adindices = data_mod[!, colname]
        select!(data_mod, Not(colname))
        replace!(adindices, missing => 0x0)
        map = convert(Vector{minimum_unsigned_type_for_n(maximum(adindices))}, adindices)
        getproperty(mdata, mapattr)[mod] = map
        getproperty(mdata, mattr)[mod] = map .> 0
    end
    setproperty!(mdata, attr, disallowmissing!(data_mod, error=false))

    keep_index = rownames .∈ (old_rownames,)
    @inbounds if sum(keep_index) != length(old_rownames)
        for (k, v) ∈ getproperty(mdata, mattr)
            if k ∉ keys(mdata.mod)
                getproperty(mdata, mattr)[k] = v[keep_index, :]
            end
        end

        pattr = Symbol(string(attr) * "p")
        for (k, v) ∈ getproperty(mdata, pattr)
            if k ∉ keys(mdata.mod)
                getproperty(mdata, pattr)[k] = v[keep_index, keep_index]
            end
        end
    end

    return mdata
end

update_obs!(mdata::MuData) = _update_attr!(mdata, :obs, 1)
update_var!(mdata::MuData) = _update_attr!(mdata, :var, 0, true)
function update!(mdata::MuData)
    update_obs!(mdata)
    update_var!(mdata)
    return mdata
end

function shrink_attr(mdata::AbstractMuData, attr::Symbol)
    globalcols = [
        col for
        col ∈ names(getproperty(mdata, attr)) if !any(startswith.(col, keys(mdata.mod) .* ":"))
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
    obsmap::StrAlignedMappingView{Tuple{1 => 1}}

    var::SubDataFrame
    var_names::SubIndex{<:AbstractString}
    varm::StrAlignedMappingView{Tuple{1 => 2}}
    varp::StrAlignedMappingView{Tuple{1 => 2, 2 => 2}}
    varmap::StrAlignedMappingView{Tuple{1 => 2}}

    uns::Dict{<:AbstractString, <:Any}
end

function Base.view(mu::MuData, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j = convertidx(I, mu.obs_names), convertidx(J, mu.var_names)
    mod = Dict(
        m => view(
            ad,
            getadidx(i, reshape(mu.obsmap[m], :), mu.obs_names, true),
            getadidx(j, reshape(mu.varmap[m], :), mu.var_names, true),
        ) for (m, ad) ∈ mu.mod
    )
    return MuDataView(
        mu,
        i,
        j,
        FrozenDict(mod),
        view(mu.obs, nrow(mu.obs) > 0 ? i : (:), :),
        view(mu.obs_names, i),
        view(mu.obsm, i),
        view(mu.obsp, i, i),
        view(mu.obsmap, i),
        view(mu.var, nrow(mu.var) > 0 ? j : (:), :),
        view(mu.var_names, j),
        view(mu.varm, j),
        view(mu.varp, j, j),
        view(mu.varmap, j),
        mu.uns,
    )
end
function Base.view(mu::MuDataView, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j =
        Base.reindex(parentindices(mu), (convertidx(I, mu.obs_names), convertidx(J, mu.var_names)))
    return view(parent(mu), i, j)
end

function Base.getindex(mu::MuDataView, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j =
        Base.reindex(parentindices(mu), (convertidx(I, mu.obs_names), convertidx(J, mu.var_names)))
    return getindex(parent(mu), i, j)
end

Base.parent(mu::MuData) = mu
Base.parent(mu::MuDataView) = mu.parent
Base.parentindices(mu::MuData) = axes(mu)
Base.parentindices(mu::MuDataView) = (mu.I, mu.J)
file(mu::MuDataView) = file(parent(mu))
