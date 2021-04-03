# AnnData and MuData

To put it briefly, `AnnData` objects represent annotated datasets with the main data as a matrix and with rich annotations that might include tables and arrays. `MuData` objects represent collections of `AnnData` objects focusing on, but not limited to, scenarios with different `AnnData` objects representing different sets of _features_ profiled for the same _samples_.

Originally, both [AnnData objects](https://github.com/theislab/anndata) and [MuData objects](https://github.com/gtca/muon) have been implemented in Python.

## AnnData

`AnnData` implementation in `Muon.jl` tries to mainly follow the [reference implmentation](https://anndata.readthedocs.io/), albeit there are some differences in how these objects are implemented and behave.

`AnnData` objects can be [stored in and read from](@ref io_anndata) `.h5ad` files.

## MuData

The basic idea behind a multimodal object is _key &rarr; value_ relationship where _keys_ represent the unique names of individual modalities and _values_ are `AnnData` objects that contain the correposnding data. Similarly to `AnnData` objects, `MuData` objects can also contain rich multimodal annotations.
