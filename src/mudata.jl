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

    axis::UInt8

    function MuData(file::Union{HDF5.File, HDF5.Group, ZGroup}, backed=false, checkversion=true)
        if checkversion
            if !has_attribute(file, "encoding-type")
                @warn "This file was not created by muon, we can't guarantee that everything will work correctly"
            elseif read_attribute(file, "encoding-type") != "MuData"
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
        mdata.obsm =
            StrAlignedMapping{Tuple{1 => 1}}(mdata, haskey(file, "obsm") ? read_dict_of_mixed(file["obsm"]) : nothing)
        mdata.obsp = StrAlignedMapping{Tuple{1 => 1, 2 => 1}}(
            mdata,
            haskey(file, "obsp") ? read_dict_of_matrices(file["obsp"]) : nothing,
        )
        mdata.obsmap = StrAlignedMapping{Tuple{1 => 1}}(
            mdata,
            haskey(file, "obsmap") ? read_dict_of_matrices(file["obsmap"]) : nothing,
        )

        # Variables
        mdata.varm =
            StrAlignedMapping{Tuple{1 => 2}}(mdata, haskey(file, "varm") ? read_dict_of_mixed(file["varm"]) : nothing)
        mdata.varp = StrAlignedMapping{Tuple{1 => 2, 2 => 2}}(
            mdata,
            haskey(file, "varp") ? read_dict_of_matrices(file["varp"]) : nothing,
        )
        mdata.varmap = StrAlignedMapping{Tuple{1 => 2}}(
            mdata,
            haskey(file, "varmap") ? read_dict_of_matrices(file["varmap"]) : nothing,
        )

        # unstructured
        mdata.uns = haskey(file, "uns") ? read_dict_of_mixed(file["uns"], separate_index=false) : Dict{String, Any}()

        mdata.axis = has_attribute(file, "axis") ? read_attribute(file, "axis") + 0x1 : 0x1

        # Modalities
        mdata.mod = OrderedDict{String, AnnData}()
        mods = keys(file["mod"])

        if has_attribute(file["mod"], "mod-order")
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
        obsm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
        varm::Union{AbstractDict{<:AbstractString, <:Union{<:AbstractArray{<:Number}, DataFrame}}, Nothing}=nothing,
        obsp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        varp::Union{AbstractDict{<:AbstractString, <:AbstractMatrix{<:Number}}, Nothing}=nothing,
        obsmap::Union{AbstractDict{<:AbstractString, <:AbstractVector{<:Integer}}, Nothing}=nothing,
        varmap::Union{AbstractDict{<:AbstractString, <:AbstractVector{<:Integer}}, Nothing}=nothing,
        uns::Union{AbstractDict{<:AbstractString, <:Any}, Nothing}=nothing,
        axis::Union{Integer, Nothing}=nothing,
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
        mdata.axis = isnothing(axis) ? 0x1 : axis

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
            write(hfile, "MuData (format-version=$MUDATAVERSION;creator=$NAME;creator-version=$VERSION)")
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
    write_attr(parent, "obs", disallowmissing(mudata.obs, error=false), index=mudata.obs_names, compress=compress)
    write_attr(parent, "obsm", mudata.obsm, index=mudata.obs_names, compress=compress)
    write_attr(parent, "obsp", mudata.obsp, compress=compress)
    write_attr(parent, "obsmap", mudata.obsmap, compress=compress)
    write_attr(parent, "var", disallowmissing(mudata.var, error=false), index=mudata.var_names, compress=compress)
    write_attr(parent, "varm", mudata.varm, index=mudata.var_names, compress=compress)
    write_attr(parent, "varp", mudata.varp, compress=compress)
    write_attr(parent, "varmap", mudata.varmap, compress=compress)
    write_attr(parent, "uns", mudata.uns, compress=compress)
    write_attribute(parent, "axis", mudata.axis - 0x1)
end

# FileIO support
load(f::File{format"h5mu"}; backed::Bool=false) = readh5mu(filename(f), backed=backed)
save(f::File{format"h5mu"}, data::AbstractMuData; compress::UInt8=0x9) = writeh5mu(filename(f), data, compress=compress)

Base.size(mdata::AbstractMuData) = (length(mdata.obs_names), length(mdata.var_names))
Base.size(mdata::AbstractMuData, d::Integer) = size(mdata)[d]

Base.getindex(mdata::AbstractMuData, modality::Symbol) = mdata.mod[string(modality)]
Base.getindex(mdata::AbstractMuData, modality::AbstractString) = mdata.mod[modality]
Base.setindex!(mdata::MuData, ad::AnnData, key::AbstractString) = setindex!(mdata.mod, ad, key)
Base.setindex!(mdata::MuData, ad::AnnData, key::Symbol) = setindex!(mdata.mod, ad, string(key))
function Base.getindex(
    mdata::MuData,
    I::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
    J::Union{OrdinalRange, Colon, AbstractVector{<:Integer}, AbstractVector{<:AbstractString}, Number, AbstractString},
)
    @boundscheck checkbounds(mdata, I, J)
    i, j = convertidx(I, mdata.obs_names), convertidx(J, mdata.var_names)
    newmu = MuData(
        mod=OrderedDict{String, AnnData}(
            k => ad[
                getadidx(i, vec(mdata.obsmap[k]), mdata.obs_names),
                getadidx(j, vec(mdata.varmap[k]), mdata.var_names),
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
        adjustmap!(vec(newmu.obsmap[mod]), i)
        adjustmap!(vec(newmu.varmap[mod]), j)
    end
    return newmu
end

getadidx(I::Colon, ref::AbstractVector{<:Unsigned}, idx::AbstractIndex{<:AbstractString}, reduce_memory=false) = I
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
getadidx(I::Number, ref::AbstractVector{<:Unsigned}, idx::Index{<:AbstractString}, reduce_memory=false) =
    getadidx([I], ref, idx, reduce_memory)

adjustmap!(map::AbstractVector{<:Unsigned}, I::Colon) = map
function adjustmap!(map::AbstractVector{<:Unsigned}, I::Union{Integer, AbstractVector{<:Integer}, OrdinalRange})
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

function _verify_global_df(mdata::MuData, attr::Symbol, axis::UInt8)
    globaldata = copy(getproperty(mdata, attr), copycols=false)
    if nrow(globaldata) == 0 && size(mdata, axis) > 0 # no data, but possibly some columns. insertcols throws an exception in this case since the new column is longer than the dataframe
        if ncol(globaldata) > 0
            @warn "mdata.$attr is empty, but has columns. You probably tried to broadcast a scalar to an empty .$attr. Dropping empty columns..."
        end
        globaldata = DataFrame() # no data, so dropping the empty columns
    elseif (n_missing_rows = size(mdata, axis) - nrow(globaldata)) > 0
        @warn "mdata.$attr has less rows than it should. You probably assigned a column with the wrong length to an empty .$attr. Filling up with missing..."
        newdf = DataFrame((colname => Fill(missing, n_missing_rows) for colname ∈ names(globaldata))...)
        globaldata = vcat(globaldata, newdf)
    elseif n_missing_rows < 0
        @warn "mdata.$attr has more rows than it should. You probably assigned a column with the wrong length to an empty .$attr. Subsetting to the first $(size(mdata, axis)) rows..."
        keepat!(globaldata, 1:size(mdata, axis))
    end
    return globaldata
end

function _update_attr!(mdata::MuData, attr::Symbol, axis::UInt8)
    namesattr = Symbol("$(attr)_names")
    mattr = Symbol("$(attr)m")
    mapattr = Symbol("$(attr)map")

    map = getproperty(mdata, mapattr)

    dupidx = Dict(mod => index_duplicates(getproperty(ad, namesattr)) for (mod, ad) ∈ mdata.mod)
    have_duplicates = false
    for (mod, dups) ∈ dupidx, (mod2, ad) ∈ mdata.mod
        if mod != mod2
            mask = dups .> 0
            have_duplicates = any(mask)
            if any(getproperty(mdata.mod[mod], namesattr)[mask] .∈ (getproperty(ad, namesattr),))
                @warn "Duplicated $namesattr should not be present in different modalities due to the ambiguity that leads to."
                break
            end
        end
    end
    if have_duplicates
        @warn "$namesattr are not unique. To make them unique, call $(namesattr)_make_unique!"
        if mdata.axis == 0
            @warn "Behavior is not defined with axis=0, $namesattr need to be made unique first."
        end
    end

    old_rownames = getproperty(mdata, namesattr)
    idxcol, dupidxcol, rowcol = find_unique_colnames(mdata, attr, 0x3)
    globaldata = _verify_global_df(mdata, attr, axis)
    globaldata = insertcols!(globaldata, idxcol => old_rownames)

    dfs = (
        DataFrame(
            idxcol => getproperty(ad, namesattr),
            (have_duplicates ? (dupidxcol => dupidx[mod],) : ())...,
            mod * ":" * rowcol => 1:length(getproperty(ad, namesattr)),
        ) for (mod, ad) ∈ mdata.mod
    )

    if length(dfs) == 1
        data_mod = iterate(dfs)[1]
    else
        data_mod =
            axis == mdata.axis || mdata.axis == 0x0 ?
            outerjoin(dfs..., on=have_duplicates ? [idxcol, dupidxcol] : idxcol, order=:left, makeunique=false) :
            vcat(dfs..., cols=:union)
    end
    have_duplicates && select!(data_mod, Not(dupidxcol))
    globalcols = [idxcol]
    for (mod, ad) ∈ mdata.mod
        if haskey(map, mod) && sum(map[mod] .> 0) == length(getproperty(ad, namesattr))
            colname = mod * ":" * rowcol
            globaldata[!, colname] = vec(map[mod])
            push!(globalcols, colname)
        end
    end
    if have_duplicates && ncol(globaldata) > 1 + length(globalcols)
        if length(globalcols) == 0
            global_dupidx = index_duplicates(old_rownames)
            if any(global_dupidx .> 0)
                @warn "$namesattr is not unique, global $attr is present, and $mapattr is empty.\
                        The update() is not well-defined, verify if global $attr map to the correct modality-specific $attr."
                globaldata[!, dupidxcol] = global_dupidx
                data_mod[!, dupidxcol] = index_duplicates(data_mod[!, idxcol])
                push!(globalcols, dupidxcol)
            end
        end
    end
    transform!(data_mod, (mod * ":" * rowcol => x -> coalesce.(x, 0x0) for mod ∈ keys(mdata.mod))..., renamecols=false)
    data_mod = leftjoin(data_mod, globaldata, on=globalcols, order=:right) # right order to keep the old ordering as much as possible

    rownames = Index(data_mod[!, idxcol])
    setproperty!(mdata, namesattr, rownames)
    empty!(map)
    attrm = getproperty(mdata, mattr)
    for mod ∈ keys(mdata.mod)
        colname = mod * ":" * rowcol
        adindices = data_mod[!, colname]
        cmap = convert(Vector{minimum_unsigned_type_for_n(maximum(adindices))}, adindices)
        map[mod] = cmap
        attrm[mod] = cmap .> 0
        select!(data_mod, Not(colname))
    end
    select!(data_mod, names(data_mod) .∉ ((idxcol, dupidxcol),))
    setproperty!(mdata, attr, disallowmissing!(data_mod, error=false))

    new_index = rownames .∈ (old_rownames,)
    old_index = old_rownames .∈ (rownames,)
    n_kept_rows = sum(new_index)
    n_new_rows = length(rownames) - n_kept_rows
    # kept rows have the same ordering as before => no need to reorder anything
    if n_kept_rows != length(old_rownames)
        @inbounds for (k, v) ∈ attrm
            if !haskey(mdata.mod, k)
                ellipsis::Union{Colon, Ellipsis} = ..
                chunk::Union{Fill{Missing, ndims(v)}, DataFrame} = Fill(missing, n_new_rows, size(v)[2:end]...)
                if isa(v, DataFrame)
                    chunk = DataFrame(chunk, names(v))
                    ellipsis = (:) # EllipsisNotation.jl doesn't work with DataFrames
                end
                attrm[k] = vcat(view(v, keep_index, ellipsis), chunk)
            end
        end

        attrp = getproperty(mdata, Symbol("$(attr)p"))
        @inbounds for (k, v) ∈ attrp
            topright = Fill(missing, n_kept_rows, n_new_rows, size(v)[3:end]...)
            bottomleft = Fill(missing, n_new_rows, n_kept_rows, size(v)[3:end]...)
            bottomright = Fill(missing, n_new_rows, n_new_rows, size(v)[3:end]...)
            attrp[k] = hvcat(2, view(v, keep_index, keep_index), topright, bottomleft, bottomright)
        end
    end
    return mdata
end

@enum ModColumnClass::UInt8 mod_column_common mod_column_unique mod_column_nonunique
function _pull_attr!(
    mdata::MuData,
    attr::Symbol,
    axis::UInt8,
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing;
    common::Bool=true,
    join_common::Bool,
    nonunique::Bool=false,
    join_nonunique::Bool=false,
    unique::Bool=true,
    prefix_unique::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M}
    if !isnothing(columns) && isa(columns, AbstractString)
        columns = (columns,)
    end
    if (axis == mdata.axis || mdata.axis == 0) && (join_common || join_nonunique)
        throw(ArgumentError("Cannot join columns with the same name for shared $(attr)_names"))
    end
    if !isnothing(mods)
        if isa(mods, AbstractString)
            mods = (mods,)
        end
        if !all(haskey.((mdata.mod,), mods))
            throw(ArgumentError("Not all requested modalities are present in the MuData object"))
        end
    end
    if only_drop
        drop = true
    end

    cols = Dict{String, Vector{String}}()
    fullcntr = counter(String)
    filteredcntr = counter(String)
    for (mod, ad) ∈ mdata.mod
        modcols = String[]
        for colname ∈ names(getproperty(ad, attr))
            inc!(fullcntr, colname)
            if (isnothing(mods) || mod ∈ mods) && (isnothing(columns) || "$mod:$colname" ∈ columns || colname ∈ columns)
                push!(modcols, colname)
                inc!(filteredcntr, colname)
            end
        end
        cols[mod] = modcols
    end

    attrmap = getproperty(mdata, Symbol("$(attr)map"))
    globaldf = _verify_global_df(mdata, attr, axis)
    fake_col = find_unique_colnames(mdata, attr, 0x1)[1]
    if isempty(globaldf)
        globaldf[:, fake_col] = Fill(missing, size(mdata, axis))
    end
    for (mod, ad) ∈ mdata.mod
        if isnothing(mods) || mod ∈ mods
            modcols = map(col -> (name=col, class=if fullcntr[col] == length(mdata.mod)
                mod_column_common
            elseif fullcntr[col] == 1
                mod_column_unique
            else
                mod_column_nonunique
            end), cols[mod])
            filter!(
                col ->
                    common && col.class == mod_column_common ||
                    nonunique && col.class == mod_column_nonunique ||
                    unique && col.class == mod_column_unique,
                modcols,
            )
            modmap = vec(attrmap[mod])
            mask = modmap .> 0
            if drop
                select!(getproperty(mod, attr), Not(modcols))
            end
            if !only_drop && length(modcols) > 0
                moddf = select(getproperty(ad, attr), map(x -> x.name, modcols), copycols=false)[
                    sortperm(view(modmap, mask)),
                    :,
                ]
                rename!(
                    moddf,
                    [
                        (
                            join_common && col.class == mod_column_common ||
                            join_nonunique && col.class == mod_column_nonunique ||
                            !prefix_unique && col.class == mod_column_unique
                        ) && filteredcntr[col.name] == fullcntr[col.name] ? col.name : "$mod:$(col.name)" for
                        col ∈ modcols
                    ],
                )

                globalsubdf = @view globaldf[mask, :]
                for (colname, col) ∈ zip(names(moddf), eachcol(moddf))
                    if columnindex(globalsubdf, colname) > 0
                        globalsubdf[:, colname] .= coalesce.(col, globalsubdf[!, colname])
                    else
                        globalsubdf[:, colname] = col
                    end
                end
            end
        end
    end
    select!(globaldf, names(globaldf) .!= fake_col)
    setproperty!(mdata, attr, globaldf)
    return mdata
end

function _push_attr!(
    mdata::MuData,
    attr::Symbol,
    axis::UInt8,
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing;
    common::Bool=true,
    prefixed::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M}
    args = Base.@locals
    if !isnothing(columns)
        for arg ∈ (:common, :prefixed)
            if args[arg]
                @warn "$arg=true, but columns given. $arg will be ignored."
            end
        end
        if isa(columns, AbstractString)
            columns = (columns,)
        end
    end
    if !isnothing(mods)
        if isa(mods, AbstractString)
            mods = (mods,)
        end
        if !all(haskey.((mdata.mod,), mods))
            throw(ArgumentError("Not all requested modalities are present in the MuData object."))
        end
    end
    if only_drop
        drop = true
    end

    df = _verify_global_df(mdata, attr, axis)
    cols = Vector{@NamedTuple{name::String, derived_name::String, prefix::Union{String, Nothing}}}()
    for colname ∈ names(df)
        splitname = split(colname, ":", limit=2)
        if (length(splitname) == 1 || !haskey(mdata.mod, splitname[1])) &&
           (isnothing(columns) && common || splitname ∈ columns)
            push!(cols, (name=colname, derived_name=colname, prefix=nothing))
        elseif (isnothing(mods) || splitname[1] ∈ mods) && (isnothing(columns) && prefixed || splitname[2] ∈ columns)
            push!(cols, (name=colname, derived_name=splitname[2], prefix=splitname[1]))
        end
    end

    if length(cols) == 0
        return mdata
    end

    seen = Set{String}()
    for col ∈ cols
        if col.derived_name ∈ seen && col.derived_name == col.name
            throw(
                ArgumentError(
                    "Cannot push multiple columns with the same anme $(col.derived_name) with and without a modality prefix. \
                     You might have to explicitly specify the columns to push. \
                     In case there are columns with the same name with and without a modality prefix, \
                     this has to be resolved first.",
                ),
            )
        end
        push!(seen, col.derived_name)
    end

    attrmap = getproperty(mdata, Symbol("$(attr)map"))
    for (mod, ad) ∈ mdata.mod
        if !only_drop && (isnothing(mods) || mod ∈ mods)
            modmap = vec(attrmap[mod])
            mask = modmap .> 0
            cdf = df[mask, :]
            select!(cdf, (col.name => col.derived_name for col ∈ cols if isnothing(col.prefix) || col.prefix == mod)...)
            cdf = cdf[sortperm(view(modmap, mask)), :]

            moddf = getproperty(ad, attr)
            for (colname, col) ∈ zip(names(cdf), eachcol(cdf))
                if columnindex(moddf, colname) > 0
                    moddf[:, colname] .= coalesce.(col, moddf[!, colname])
                else
                    moddf[!, colname] = col
                end
            end
        end
    end
    if drop
        select!(df, Not(map(x -> x.name, cols)))
    end
    setproperty!(mdata, attr, df)
    return mdata
end

update_obs!(mdata::MuData) = _update_attr!(mdata, :obs, 0x1)
update_var!(mdata::MuData) = _update_attr!(mdata, :var, 0x2)
function update!(mdata::MuData)
    update_obs!(mdata)
    update_var!(mdata)
    return mdata
end

pull_obs!(
    mdata::MuData;
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing,
    common::Bool=true,
    join_common::Union{Bool, Nothing}=nothing,
    nonunique::Bool=true,
    join_nonunique::Bool=false,
    unique::Bool=true,
    prefix_unique::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M} = _pull_attr!(
    mdata,
    :obs,
    0x1,
    columns,
    mods,
    common=common,
    join_common=isnothing(join_common) ? mdata.axis == 0x2 : join_common,
    nonunique=nonunique,
    join_nonunique=join_nonunique,
    unique=unique,
    prefix_unique=prefix_unique,
    drop=drop,
    only_drop=only_drop,
)
pull_var!(
    mdata::MuData;
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing,
    common::Bool=true,
    join_common::Union{Bool, Nothing}=nothing,
    nonunique::Bool=true,
    join_nonunique::Bool=false,
    unique::Bool=true,
    prefix_unique::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M} = _pull_attr!(
    mdata,
    :var,
    0x2,
    columns,
    mods,
    common=common,
    join_common=isnothing(join_common) ? mdata.axis == 0x1 : join_common,
    nonunique=nonunique,
    join_nonunique=join_nonunique,
    unique=unique,
    prefix_unique=prefix_unique,
    drop=drop,
    only_drop=only_drop,
)

push_obs!(
    mdata::MuData;
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing,
    common::Bool=true,
    prefixed::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M} =
    _push_attr!(mdata, :obs, 0x1, columns, mods, common=common, prefixed=prefixed, drop=drop, only_drop=only_drop)
push_var!(
    mdata::MuData;
    columns::Union{AbstractVector{<:AbstractString}, NTuple{N, <:AbstractString}, AbstractString, Nothing}=nothing,
    mods::Union{AbstractVector{<:AbstractString}, NTuple{M, <:AbstractString}, AbstractString, Nothing}=nothing,
    common::Bool=true,
    prefixed::Bool=true,
    drop::Bool=false,
    only_drop::Bool=false,
) where {N, M} =
    _push_attr!(mdata, :var, 0x2, columns, mods, common=common, prefixed=prefixed, drop=drop, only_drop=only_drop)

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
            getadidx(i, vec(mu.obsmap[m]), mu.obs_names, true),
            getadidx(j, vec(mu.varmap[m]), mu.var_names, true),
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
    i, j = Base.reindex(parentindices(mu), (convertidx(I, mu.obs_names), convertidx(J, mu.var_names)))
    return view(parent(mu), i, j)
end

function Base.getindex(mu::MuDataView, I, J)
    @boundscheck checkbounds(mu, I, J)
    i, j = Base.reindex(parentindices(mu), (convertidx(I, mu.obs_names), convertidx(J, mu.var_names)))
    return getindex(parent(mu), i, j)
end

Base.parent(mu::MuData) = mu
Base.parent(mu::MuDataView) = mu.parent
Base.parentindices(mu::MuData) = axes(mu)
Base.parentindices(mu::MuDataView) = (mu.I, mu.J)
file(mu::MuDataView) = file(parent(mu))

obs_names_make_unique!(mdata::MuData) = attr_make_unique(mdata, :obs, 0x1)
var_names_make_unique!(mdata::MuData) = attr_make_unique(mdata, :obs, 0x2)
function attr_make_unique!(mdata::MuData, attr::Symbol, axis::UInt8)
    namesattr = Symbol(string(attr) * "_names")

    for ad ∈ values(mdata.mod)
        attr_make_unique!(ad, namesattr)
    end
    mods = collect(keys(mdata.mod))
    have_duplicates = false
    for i ∈ 1:(length(mods) - 1), j ∈ (i + 1):length(mods)
        if length(getproperty(mdata.mod[mods[i]], namesattr) ∩ getproperty(mdata.mod[mods[j]], namesattr)) > 0
            @warn "Modality names will be prepended to $(string(namesattr)) since there are identical $(string(namesattr)) in different modalities."
            have_duplicates = true
            break
        end
    end
    if have_duplicates
        for (modname, mod) ∈ mdata.mod
            getproperty(mod, namesattr) .= modname .* ":" .* getproperty(mod, namesattr)
        end
        update_attr!(mdata, attr, axis)
    end
    return mdata
end
