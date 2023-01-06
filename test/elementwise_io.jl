

using CategoricalArrays
using HDF5

tmp = mktempdir()

@testset "encoding_types" begin
    tempfile1 = joinpath(tmp, "tmp_encoding_types.h5ad")
    output = h5open(tempfile1, "w")
    g = create_group(output, "test")

    function test_encoding(data, expected_type, expected_version)
        Muon.write_attr(g, "test", data)
        attrs = attributes(g["test"])
        @test haskey(attrs, "encoding-type")
        @test read(attrs["encoding-type"]) == expected_type
        @test haskey(attrs, "encoding-version")
        @test read(attrs["encoding-version"]) == expected_version
    end

    enc_tests = [
        ("hello world",      "string",         "0.2.0"),
        ([1,2,3],            "array",          "0.2.0"),
        (["hello", "world"], "string-array",   "0.2.0"),
        (1,                  "numeric-scalar", "0.2.0"),
        (1.0,                "numeric-scalar", "0.2.0"),
        (true,               "numeric-scalar", "0.2.0"),
        (Dict("a" => 1),     "dict",           "0.1.0"),
        (CategoricalArray(["a", "b", "a", "a"]), "categorical", "0.2.0"),
        (CategoricalArray([1, 1, 2, 1]),         "categorical", "0.2.0"),
    ]

    for args in enc_tests
        test_encoding(args...)
    end
end

@testset "roundtrip" begin
    tempfile1 = joinpath(tmp, "tmp_roundtrip.h5ad")
    output = h5open(tempfile1, "w")
    g = create_group(output, "test")

    outdata = Dict(
        "a" => [1,2,3],
        "b" => [true, false, true],
        "c" => ["a", "b", "c"],
        "d" => 1,
        "e" => true,
        "f" => "a",
        "g" => CategoricalArray(["a", "b", "a", "a"]),
        "h" => CategoricalArray([1, 1, 2, 1]))

    Muon.write_attr(g, "test", outdata)
    indata = Muon.read_dict_of_mixed(g["test"])

    @test keys(outdata) == keys(indata)
    for k in keys(outdata)
        @test typeof(outdata[k]) == typeof(indata[k])
        @test outdata[k] == indata[k]
    end
end
