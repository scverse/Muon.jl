# AnnData and MuData

To put it briefly, `AnnData` objects represent annotated datasets with the main data as a matrix and with rich annotations that might include tables and arrays. `MuData` objects represent collections of `AnnData` objects focusing on, but not limited to, scenarios with different `AnnData` objects representing different sets of _features_ profiled for the same _samples_.

Originally, both [AnnData objects](https://github.com/theislab/anndata) and [MuData objects](https://github.com/gtca/muon) have been implemented in Python.

## AnnData

`AnnData` implementation in `Muon.jl` tries to mainly follow the [reference implementation](https://anndata.readthedocs.io/), albeit there are some differences in how these objects are implemented and behave due to how different languages are designed and opeate.

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
ad.obs_names .= "obs_" .* ad.obs_names
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

### Slicing AnnData objects

Just as simple arrays, AnnData objects can be subsetted with slicing operations, with the first dimension corresponding to _observations_ and the second dimension corresponding to _variables_:

```@example 1
obs_sub = "obs_" .* string.(collect(1:3))
ad_sub = ad[obs_sub,:]
```

Since the dimensions are labelled, using names is a natural way to subset these objects but boolean and integer arrays can be used as well:

```@example 1
# both return the same subset
ad_sub[[true,false,true],:]
ad_sub[[1,3],:]
```

## MuData

The basic idea behind a multimodal object is _key_ ``\rightarrow`` _value_ relationship where _keys_ represent the unique names of individual modalities and _values_ are `AnnData` objects that contain the correposnding data. Similarly to `AnnData` objects, `MuData` objects can also contain rich multimodal annotations.

```@example 1
import Distributions.Binomial # hide
import Muon.MuData # hide
ad2 = AnnData(X=rand(Binomial(1, 0.3), (10, 7)),
              obs_names="obs_" .* string.(collect(1:10)))

md = MuData(mod=Dict("view_rand" => ad, "view_binom" => ad2))
```

Features are considered unique to each modality.

### Slicing MuData objects

Slicing now works across all modalities:

```@example 1
md[["obs_1", "obs_9"],:]
```

### Multimodal annotation

We can store annotation at the multimodal level, that includes multidimensional arrays:

```@example 1
md.obsm["X_svd"] = f.U * Diagonal(f.S);
md.obsm
```
