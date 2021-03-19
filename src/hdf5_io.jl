function read_dataframe(tablegroup::HDF5.Group)
    columns = read_attribute(tablegroup, "column-order")

    havecat = false
    if haskey(tablegroup, "__categories")
        havecat = true
        catcols = tablegroup["__categories"]
    end

    if haskey(attributes(tablegroup), "_index")
        indexdsetname = read_attribute(tablegroup, "_index")
        rownames = read(tablegroup[indexdsetname])
    else
        rownames = nothing
    end

    df = DataFrame()

    for col in columns
        column = read(tablegroup, col)
        if havecat && haskey(catcols, col)
            cats = read(catcols, col)
            column = CategoricalArray(map(x -> cats[x + 1], column))
        end
        df[!, col] = column
    end

    return df, rownames
end

read_matrix(f::HDF5.Dataset) = read(f)

function read_matrix(f::HDF5.Group)
    enctype = read_attribute(f, "encoding-type")
    shape = read_attribute(f, "shape")
    if enctype == "csc_matrix"
        return SparseMatrixCSC(
            shape[1],
            shape[2],
            read(f, "indptr") .+ 1,
            read(f, "indices") .+ 1,
            read(f, "data"),
        )
    elseif enctype == "csr_matrix"
        return
        SparseMatrixCSC(
            shape[2],
            shape[1],
            read(f, "indptr") .+ 1,
            read(f, "indices") .+ 1,
            read(f, "data"),
        )'
    else
        throw("unknown encoding $enctype")
    end
end
