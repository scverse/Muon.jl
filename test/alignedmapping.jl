using HDF5
file = h5open(tempname(), "w")

arr1 = rand((10 for i in 1:5)...)
arr2 = rand(Int, 5, (10 for i in 1:4)...)
arr3 = rand(Float32, 10, 10, 11, 10, 10, 10)
arr4 = rand(UInt8, 10, 10, 10, 20, 10)

make_init_args(::Type{Muon.AlignedMapping}, d::Dict) = (d,)
function make_init_args(::Type{Muon.BackedAlignedMapping}, d::Dict)
    path = "test"
    Muon.write_attr(file, path, d)
    return (file, path)
end

@testset "$name" for (name, T) in (
    ("AlignedMapping", Muon.AlignedMapping),
    ("BackedAlignedMapping", Muon.BackedAlignedMapping),
)
    @testset "ref: $ndims dimensions" for ndims in 1:5
        ref = Array{Bool}(undef, (10 for i in 1:ndims)...)
        @test_throws DimensionMismatch T{Tuple{2 => 1, 1 => ndims, 3 => ndims}}(
            ref,
            make_init_args(T, Dict("test" => arr2))...,
        )
        @test_throws DimensionMismatch T{Tuple{2 => 1, 1 => ndims, 3 => ndims}}(
            ref,
            make_init_args(T, Dict("test" => arr3))...,
        )
        mapping = T{Tuple{2 => 1, 1 => ndims, 3 => ndims}}(
            ref,
            make_init_args(T, Dict("arr1" => arr1))...,
        )
        mapping["arr4"] = arr4
        @test_throws DimensionMismatch mapping["arr2"] = arr2
        @test_throws DimensionMismatch mapping["arr3"] = arr3
        @test mapping["arr1"] == arr1
        @test mapping["arr4"] == arr4
    end
end
