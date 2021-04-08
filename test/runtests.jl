using Muon, Test

@testset "Index" begin
    include("index.jl")
end

@testset "HDF5 sparse matrix" begin
    include("sparse_hdf5.jl")
end
@testset "AnnData" begin
    include("anndata.jl")
end
