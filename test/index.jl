using Random
Random.seed!(42)
size = rand(100:200)

function make_testvalues(size::Integer)
    testvalues = Vector{String}(undef, size)
    for i in 1:size
        length = rand(50:200)
        testvalues[i] = randstring(length)
    end
    return testvalues
end

testvalues = make_testvalues(size)
idx = Muon.Index(testvalues)
@testset "integer indexing" begin
    @testset "element: $i" for (i, v) in enumerate(testvalues)
        @test idx[i] == v
    end
end

@testset "value indexing" begin
    @testset "element: $i" for (i, v) in enumerate(testvalues)
        @test idx[v] == i
    end
end

testvalues2 = make_testvalues(size)
testvalues3 = copy(testvalues)
@testset "value replacement" begin
    @testset "element: $i" for (i, v) in enumerate(testvalues)
        testvalues3[i] = testvalues2[i]
        idx[v] = testvalues2[i]
        @test idx == testvalues3
    end
end

@testset "index replacement" begin
    @testset "element $i" for (i, v) in enumerate(testvalues2)
        testvalues3[i] = testvalues[i]
        idx[i] = testvalues[i]
        @test idx == testvalues3
    end
end

@testset "presence of items" begin
    @testset "element $i" for (i, v) in enumerate(idx)
        @test v ∈ idx
    end
    @test "test" ∉ idx
end

@testset "duplicates" begin
    idx = Muon.Index(["test1", "test2", "test3", "test4", "test1", "test6"])
    @test idx["test1"] ∈ (1, 5)
    @test sort(idx["test1", true]) == [1, 5]
    idx[5] = "test5"
    @test idx[1] == "test1"
    @test idx[5] == "test5"
    idx[5] = "test2"
    @test idx[5] == "test2"
end
