using Muon, Test

@testset "Index" begin
    include("index.jl")
end
@testset "Elementwise IO" begin
    include("elementwise_io.jl")
end
@testset "HDF5 backed matrix" begin
    include("backed_matrix.jl")
end
@testset "aligned mappings" begin
    include("alignedmapping.jl")
end
@testset "AnnData" begin
    include("anndata.jl")
end
@testset "MuData" begin
    include("mudata.jl")
end
