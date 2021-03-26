mutable struct AnnData
    file::Union{HDF5.File, HDF5.Group, Nothing}

    X::Union{AbstractMatrix{<:Number}, Nothing}
    layers::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}

    obs::Union{DataFrame, Nothing}
    obs_names::Union{AbstractVector{<:AbstractString}, Nothing}
    obsm::Union{AbstractDict{<:AbstractString, AbstractArray{<:Number}}, Nothing}
    obsp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}

    var::Union{DataFrame, Nothing}
    var_names::Union{AbstractVector{<:AbstractString}, Nothing}
    varm::Union{AbstractDict{<:AbstractString, AbstractArray{<:Number}}, Nothing}
    varp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}

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

    function AnnData(;
        X::AbstractMatrix{<:Number},
        obs::Union{DataFrame, Nothing}=nothing,
        obs_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        var::Union{DataFrame, Nothing}=nothing,
        var_names::Union{AbstractVector{<:AbstractString}, Nothing}=nothing,
        obsm::Union{AbstractDict{<:AbstractString, AbstractArray{<:Number}}, Nothing}=nothing,
        varm::Union{AbstractDict{<:AbstractString, AbstractArray{<:Number}}, Nothing}=nothing,
        obsp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
        varp::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
        layers::Union{AbstractDict{<:AbstractString, AbstractMatrix{<:Number}}, Nothing}=nothing,
    )
        m, n = size(X)
        if !isnothing(obs) && size(obs, 1) != m
            throw(DimensionMismatch("X has $n rows, but obs has $(size(obs, 1)) rows"))
        end

        if isnothing(obs_names)
            obs_names = string.(collect(1:m))
        elseif length(obs_names) != m
            throw(DimensionMismatch("X has $m rows, but $(length(obs_names)) obs_names given"))
        end

        if !isnothing(var) && size(var, 1) != n
            throw(DimensionMismatch("X has $n columns, but var has $(size(var, 1)) rows"))
        end

        if isnothing(var_names)
            var_names = string.(collect(1:n))
        elseif length(var_names) != n
            throw(DimensionMismatch("X has $m columns, but $(length(var_names)) var_names given"))
        end

        # TODO: custom Dict class that verifies shapes upon assignment
        if !isnothing(obsm)
            for (k, v) in obsm
                if size(v, 1) != m
                    throw(DimensionMismatch("X has $m rows, but obsm[$k] has $(size(v, 1)) rows"))
                end
            end
        end
        if !isnothing(varm)
            for (k, v) in varm
                if size(v, 1) != n
                    throw(DimensionMismatch("X has $n columns, but varm[$k] has $(size(v, 1)) rows"))
                end
            end
        end
        if !isnothing(obsp)
            for (k, v) in obsp
                if size(v, 1) != size(v, 2)
                    throw(DimensionMismatch("obsp[$k] is not square"))
                elseif size(v, 1) != m
                    throw(DimensionMismatch("X has $m rows, but obsp[$k] has $(size(v, 1)) rows"))
                end
            end
        end
        if !isnothing(varp)
            for (k, v) in varp
                if size(v, 1) != size(v, 2)
                    throw(DimensionMismatch("varp[$k] is not square"))
                elseif size(v, 1) != n
                    throw(DimensionMismatch("X has $n columns, but varp[$k] has $(size(v, 1)) rows"))
                end
            end
        end
        if !isnothing(layers)
            for (k, v) in layers
                if size(v) != size(X)
                    throw(DimensionMismatch("X has shape $(size(X)), but layers[$k] has shape $(size(v))"))
                end
            end
        end


        return new(nothing, X, layers, obs, obs_names, obsm, obsp, var, var_names, varm, varp)
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
