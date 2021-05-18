![Muon.jl](https://user-images.githubusercontent.com/32863903/112323914-864a0f80-8cb2-11eb-91ae-375cdb61cd1b.png)

# Muon for Julia

Muon is originally [a Python library to work with multimodal data](https://github.com/gtca/muon). `Muon.jl` brings the ability to work with the same data structures to Julia.

`Muon.jl` implements I/O for `.h5mu` and `.h5ad` files as well as basic operations on the multimodal objects.

## Introduction

Datasets can usually be represented as matrices with values for the _variables_ measured in different samples, or _observations_. Variables and observations tend to have annotations attached to them, a typical example would be metadata annotating samples. Such a dataset with the matrix in its centre and different kinds of annotations associated with it can be stored conveniently in an [annotated data](https://anndata.readthedocs.io/en/latest/) object, `AnnData` for short.

Multimodal datasets are characterised by the variables coming from different generative processes. Each of these _modalities_ is an annotated dataset by itself, but they can be managed and analyzed together within a `MuData` object.

## Examples

`MuData` objects can be created from `.h5mu` files:

```julia
using Muon

mdata = readh5mu("pbmc10k.h5mu");
```

Individual modalities can be accessed directly by their name:

```julia
mdata["rna"]
# => AnnData object 10110 ✕ 101001
```

Low-dimensional representations of the data can be plotted with the plotting library of choice:

```julia
using DataFrames
using GLMakie
using AlgebraOfGraphics

df = DataFrame(LF1 = mdata.obsm["X_umap"][1,:],
               LF2 = mdata.obsm["X_umap"][2,:]);

data(df) * mapping(:LF1, :LF2) * visual(Scatter) |> draw
```

