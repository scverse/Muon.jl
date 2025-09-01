using PooledArrays
using HDF5
using Zarr

h5file = h5open(tempname(), "w")
zarrfile = zgroup(tempname())

@testset "$backend" for (file, backend) ∈ ((h5file, "HDF5"), (zarrfile, "Zarr"))
    @testset "encoding types" begin
        function test_encoding(data, expected_type, expected_version)
            Muon.write_attr(file, "encoding_types", data)
            dset = file["encoding_types"]
            @test Muon.has_attribute(dset, "encoding-type")
            @test Muon.read_attribute(dset, "encoding-type") == expected_type
            @test Muon.has_attribute(dset, "encoding-version")
            @test Muon.read_attribute(dset, "encoding-version") == expected_version
        end

        enc_tests = [
            ("hello world", "string", "0.2.0"),
            ([1, 2, 3], "array", "0.2.0"),
            ([true, false, true], "array", "0.2.0"),
            (["hello", "world"], "string-array", "0.2.0"),
            (1, "numeric-scalar", "0.2.0"),
            (1.0, "numeric-scalar", "0.2.0"),
            (true, "numeric-scalar", "0.2.0"),
            (Dict("a" => 1), "dict", "0.1.0"),
            ([1, 2, missing, 3], "nullable-integer", "0.1.0"),
            ([true, false, missing, true], "nullable-boolean", "0.1.0"),
            (BitVector([true, false, true]), "array", "0.2.0"),
            (BitMatrix([true false true; false true false]), "array", "0.2.0"),
            (PooledArray(["a", "b", "a", "a"]), "categorical", "0.2.0"),
            (PooledArray([1, 1, 2, 1]), "categorical", "0.2.0"),
        ]

        for args ∈ enc_tests
            test_encoding(args...)
        end
    end

    @testset "roundtrip" begin
        outdata = Dict(
            "a" => [1, 2, 3],
            "b" => BitVector([true, false, true]),
            "c" => ["a", "b", "c"],
            "d" => 1,
            "e" => true,
            "f" => "a",
            "g" => PooledArray(["a", "b", "a", "a"]; compress=true, signed=true),
            "h" => PooledArray([1, 1, 2, 1]),
            "i" => [1, 2, missing, 3],
            "k" => [true, false, missing, true],
        )

        Muon.write_attr(file, "roundtrip", outdata)
        indata = Muon.read_dict_of_mixed(file["roundtrip"])

        @test keys(outdata) == keys(indata)
        for k ∈ keys(outdata)
            @test typeof(outdata[k]) == typeof(indata[k])

            # special case to deal with missing
            if isa(outdata[k], AbstractArray) && Missing <: eltype(outdata[k])
                @test all(outdata[k] .=== indata[k])
            else
                @test outdata[k] == indata[k]
            end
        end

        # Bool arrays don't survive round trip, but will be read as BitArrays
        outdata = Dict("a" => [true, false, true, true])
        Muon.write_attr(file, "test_bool", outdata)
        indata = Muon.read_dict_of_mixed(file["test_bool"])

        @test keys(outdata) == keys(indata)
        for k ∈ keys(outdata)
            @test eltype(outdata[k]) == eltype(indata[k])
            @test all(outdata[k] .=== indata[k])
        end
    end
end
