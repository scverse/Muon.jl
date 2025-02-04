function Base.sort!(idx::AbstractArray, vals::AbstractArray...)
    if !issorted(idx)
        ordering = sortperm(idx)
        permute!(idx, ordering)
        for v in vals
            permute!(v, ordering)
        end
    end
end

backed_matrix(obj::Union{HDF5.File, HDF5.Group}) = SparseDataset(obj)
backed_matrix(obj::HDF5.Dataset) = TransposedDataset(obj)

function hdf5_object_name(obj::Union{HDF5.File, HDF5.Group, HDF5.Dataset})
    name = HDF5.name(obj)
    return name[(last(findlast("/", name)) + 1):end]
end

function find_unique_colnames(mdata::MuData, property::Symbol, ncols::Int)
    nchars = 16
    allunique = false
    local colnames::Vector{String}
    while !allunique
        colnames = [randstring(nchars) for _ in 1:ncols]
        allunique = length(Set(colnames)) == ncols
        nchars *= 2
    end

    it = Iterators.flatten(((mdata,), values(mdata.mod)))
    for i in 1:ncols
        finished = false
        while !finished
            for ad in it
                try
                    names(getproperty(ad, property), colnames[i])
                    colnames[i] = "_" * colnames[i]
                    break
                catch e
                    if !(e isa ArgumentError)
                        rethrow(e)
                    end
                end
            end
            finished = true
        end
    end
    return colnames
end

function index_duplicates(idx::AbstractArray{T}) where T
    counter = Dict{T, UInt8}()
    dup_idx = Vector{UInt8}(undef, length(idx))
    for (i, val) in enumerate(idx)
        if val in keys(counter)
            count = counter[val] + 1
        else
            count = 0
        end
        dup_idx[i] = counter[val] = count
    end
    return dup_idx
end

function minimum_unsigned_type_for_n(n::Number)
    local mintype::Type
    for type in [UInt8, UInt16, UInt32, UInt64, UInt128]
        mval = typemax(type)
        if mval >= n
            mintype = type
            break
        end
    end
    return mintype
end

@inline function convertidx(
    idx::Union{AbstractUnitRange, Colon, AbstractVector{<:Integer}, AbstractVector{Bool}},
    ref::AbstractIndex{<:AbstractString},
)
    return idx
end
@inline function convertidx(idx::Number, ref::AbstractIndex{<:AbstractString})
    return idx:idx
end
@inline convertidx(
    idx::Union{AbstractString, AbstractVector{<:AbstractString}},
    ref::AbstractIndex{<:AbstractString},
) = ref[idx, true]

Base.axes(A::Union{AbstractMuData, AbstractAnnData}) = map(n -> Base.OneTo(n), size(A))
isbacked(ad::Union{AbstractMuData, AbstractAnnData}) = !isnothing(file(ad))

@inline function Base.checkbounds(::Type{Bool}, A::Union{AbstractMuData, AbstractAnnData}, I...)
    Base.checkbounds_indices(
        Bool,
        axes(A),
        Tuple(i isa AbstractString || i isa AbstractVector{<:AbstractString} ? (:) : i for i in I),
    )
end

@inline function Base.checkbounds(A::Union{AbstractMuData, AbstractAnnData}, I...)
    checkbounds(Bool, A, I...) || throw(BoundsError(A, I))
    nothing
end

function Base.summary(A::Union{AbstractMuData, AbstractAnnData})
    s = size(A)
    return "$(typeof(A)) with $(s[1]) observations and $(s[2]) variables"
end

Base.summary(io::IO, A::Union{AbstractMuData, AbstractAnnData}) = print(io, summary(A))

Base.firstindex(A::Union{AbstractMuData, AbstractAnnData}, d::Integer) = 1
Base.lastindex(A::Union{AbstractMuData, AbstractAnnData}, d::Integer) = size(A, d)

Base.copy(d::Union{MuDataView, AnnDataView}) = parent(d)[parentindices(d)...]

"""
    var_names_make_unique!(A::AnnData, join = '-')

Make `A.var_names` unique by appending `join` and sequential numbers
(1, 2, 3 etc) to duplicate elements, leaving the first unchanged.
"""
function var_names_make_unique!(A::AnnData, join='-')
    index_make_unique!(A.var_names, join)
end

"""
    obs_names_make_unique!(A::AnnData, join = '-')

Make `A.obs_names` unique by appending `join` and sequential numbers
(1, 2, 3 etc) to duplicate elements, leaving the first unchanged.
"""
function obs_names_make_unique!(A::AnnData, join='-')
    index_make_unique!(A.obs_names, join)
end

function index_make_unique!(index, join)
    duplicates = duplicateindicies(index)

    if isempty(duplicates)
        @info "var names are already unique, doing nothing"
        return nothing
    end

    example_colliding_names = []
    set = Set(index)

    for (name, positions) in duplicates
        i = 1
        for pos in Iterators.rest(positions, 2)
            while true
                potential = string(index[pos], join, i)
                i += 1
                if potential in set
                    if length(example_colliding_names) <= 5
                        push!(example_colliding_names, potential)
                    end
                else
                    index[pos] = potential
                    push!(set, potential)
                    break
                end
            end
        end
    end

    if !isempty(example_colliding_names)
        @warn """
              Appending $(join)[1-9...] to duplicates caused collision with another name.
              Example(s): $example_colliding_names
              This may make the names hard to interperet.
              Consider setting a different delimiter with `join={delimiter}`
              """
    end
end

function duplicateindicies(v::Muon.Index{T, I}) where {T <: AbstractString, I <: Integer}
    varnames = Dict{T, Vector{Int64}}()

    for i in eachindex(v)
        if haskey(varnames, v[i])
            push!(varnames[v[i]], i)
        else
            varnames[v[i]] = [i]
        end
    end

    filter!(x -> length(last(x)) > 1, varnames)
    varnames
end
