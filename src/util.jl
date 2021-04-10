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
    finished = false
    it = Iterators.flatten(((mdata,), values(mdata.mod)))
    while !finished
        for i in 1:ncols
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
        end
        finished = true
    end
    return colnames
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
    idx::Union{AbstractUnitRange, Colon, AbstractVector{<:Integer}},
    ref::Index{<:AbstractString},
)
    return idx
end
@inline function convertidx(idx::Number, ref::Index{<:AbstractString})
    return idx:idx
end
@inline convertidx(
    idx::Union{AbstractString, AbstractVector{<:AbstractString}},
    ref::Index{<:AbstractString},
) = ref[idx, true]

Base.axes(A::Union{MuData, AnnData}) = map(n -> Base.OneTo(n), size(A))

@inline function Base.checkbounds(::Type{Bool}, A::Union{MuData, AnnData}, I...)
    Base.checkbounds_indices(
        Bool,
        axes(A),
        Tuple(i isa AbstractString || i isa AbstractVector{<:AbstractString} ? (:) : i for i in I),
    )
end

@inline function Base.checkbounds(A::Union{MuData, AnnData}, I...)
    checkbounds(Bool, A, I...) || throw(BoundsError(A, I))
    nothing
end

function Base.summary(A::Union{MuData, AnnData})
    s = size(A)
    return "$(typeof(A)) with $(s[1]) observations and $(s[2]) variables"
end

Base.summary(io::IO, A::Union{MuData, AnnData}) = print(io, summary(A))

Base.firstindex(A::Union{MuData, AnnData}, d::Integer) = 1
Base.lastindex(A::Union{MuData, AnnData}, d::Integer) = size(A, d)
