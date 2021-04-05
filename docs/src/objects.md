# AnnData and MuData

To put it briefly, `AnnData` objects represent annotated datasets with the main data as a matrix and with rich annotations that might include tables and arrays. `MuData` objects represent collections of `AnnData` objects focusing on, but not limited to, scenarios with different `AnnData` objects representing different sets of _features_ profiled for the same _samples_.

Originally, both [AnnData objects](https://github.com/theislab/anndata) and [MuData objects](https://github.com/gtca/muon) have been implemented in Python.

## AnnData

`AnnData` implementation in `Muon.jl` tries to mainly follow the [reference implmentation](https://anndata.readthedocs.io/), albeit there are some differences in how these objects are implemented and behave.

`AnnData` objects can be [stored in and read from](@ref io_anndata) `.h5ad` files.

### Creating AnnData objects

A simple 2D array is already enough to initialize an annotated data object:

```@example 1
import Random # hide
Random.seed!(1) # hide
import Muon.AnnData # hide
x = rand(10, 2) * rand(2, 5);
ad = AnnData(X=x)
```

Observations correpond to the rows of the matrix and have unique names:

```@example 1
ad.obs_names = "obs_" .* ad.obs_names
```

Corresponding arrays for the observations are stored in the `.obsm` slot:

```@example 1
import LinearAlgebra.svd, LinearAlgebra.Diagonal # hide
f = svd(x);
ad.obsm["X_svd"] = f.U * Diagonal(f.S);
```

When data is assigned, it is verified first that the dimensions match:

```@example 1
ad.obsm["X_Vt"] = f.Vt  # won't work
# => DimensionMismatch
```

## MuData

The basic idea behind a multimodal object is _key_ ``\rightarrow`` _value_ relationship where _keys_ represent the unique names of individual modalities and _values_ are `AnnData` objects that contain the correposnding data. Similarly to `AnnData` objects, `MuData` objects can also contain rich multimodal annotations.

```@example 1
import Distributions.Binomial # hide
import Muon.MuData # hide
ad2 = AnnData(X=rand(Binomial(1, 0.3), (10, 7)))

md = MuData(mod=Dict("view_rand" => ad, "view_binom" => ad2))
```

