# I/O

## Reading .h5mu files

To read multimodal HDF5 files, which have the `.h5mu` extension, there is `readh5mu`:

```julia
m = readh5mu("multimodal_dataset.h5mu")
# => MuData object 10101 x 101010
```

## Writing .h5mu files

To save the multimodal object on disk, there is [`writeh5mu`](@ref):

```julia
writeh5mu("multimodal_dataset.h5mu", m)
```

## .h5ad files I/O

For serializing and deserializing AnnData objects, there are [`writeh5ad`](@ref) and [`readh5ad`](@ref):

```julia
writeh5ad("dataset.h5ad", ad)
ad = readh5ad("dataset.h5ad")
```
