using Random
Random.seed!(42)
_size = rand(100:200)

make_testvalues(_size::Integer) = [randstring(rand(50:200)) for _ in 1:_size]

testvalues = make_testvalues(_size)
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

testvalues2 = make_testvalues(_size)
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
    dupidx = Muon.Index(["test1", "test2", "test3", "test4", "test1", "test6"])
    @test dupidx["test1"] ∈ (1, 5)
    @test sort(dupidx["test1", true]) == [1, 5]
    dupidx[5] = "test5"
    @test dupidx[1] == "test1"
    @test dupidx[5] == "test5"
    dupidx[5] = "test2"
    @test dupidx[5] == "test2"
end

@testset "views" begin
    subidx1, subidx2 = @view(idx[26:75]), @view(idx[collect(26:75)])
    subidx3 = copy(subidx1)
    @test subidx1 == subidx2 == subidx3 == idx[26:75]
    @test subidx1[3:5] == subidx2[3:5] == subidx3[3:5] == idx[28:30]
    @test subidx1[idx[27]] == subidx2[idx[27]] == subidx3[idx[27]] == 2
    @test subidx1[idx[27], true] == subidx2[idx[27], true] == subidx3[idx[27], true] == [2]
    @test_throws KeyError subidx1["a"]
    @test_throws KeyError subidx2["a"]
    @test_throws KeyError subidx3["a"]
    @test subidx1["a", false, false] == subidx2["a", false, false] == subidx3["a", false, false] == 0
    @test subidx1["a", true, false] == subidx2["a", true, false] == subidx3["a", true, false] == []
    @test @view(subidx1[3:5]) == @view(subidx2[3:5]) == @view(subidx3[3:5]) == idx[28:30]
    @test @view(idx[:]) == idx
end
