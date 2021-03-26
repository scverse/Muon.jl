using Muon, Test

@testset "HDF5 sparse matrix" begin
    include("sparse_hdf5.jl")
end
@testset "AnnData" begin
    include("anndata.jl")
end
