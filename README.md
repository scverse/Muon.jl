# Muon for Julia

Muon is [a Python library to work with multimodal data](https://github.com/gtca/muon). `Muon.jl` brings the ability to work with the same data structures to Julia.

`Muon.jl` implements I/O for `.h5mu` and `.h5ad` files as well as basic operations on the multimodal objects.

## Examples

```julia
using Muon

mdata = readh5mu("pbmc10k.h5mu");

using DataFrames
using GLMakie
using AlgebraOfGraphics

df = DataFrame(LF1 = mdata.obsm["X_umap"][1,:],
               LF2 = mdata.obsm["X_umap"][2,:]);

data(df) * mapping(:LF1, :LF2) * visual(Scatter) |> draw
```

Individual modalities can be accessed directly by their name:

```julia
mdata["rna"]
# => AnnData object 10110 ✕ 101001
```
