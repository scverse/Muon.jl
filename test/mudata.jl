using DataFrames
using HDF5

tmp = mktempdir()

n = 100
d1 = 10
d2 = 20

warn_msg = "Cannot join columns with the same name because var_names are intersecting."
warn_msg2 = "Duplicated obs_names should not be present in different modalities due to the ambiguity that leads to."

function make_ads()
    obs_names = ["obs_$i" for i in 1:n]
    ad1 = AnnData(
        X=rand(n, d1),
        obs_names=copy(obs_names),
        var=DataFrame(testcol=["ad1_$i" for i in 1:d1]),
    )

    obs_names[10] = obs_names[n] = "testobs"
    ad2 = AnnData(
        X=rand(n, d2),
        obs_names=copy(obs_names),
        layers=Dict("testlayer" => rand(UInt16, n, d2)),
        varm=Dict("test" => rand(Int8, d2, 5, 2, 10)),
    )

    obs_names[10] = "obs_10"
    ad3 = AnnData(
        X=rand(n, d1),
        obs_names=copy(obs_names),
        var=DataFrame(commoncol=(d1 + 1):(2d1), ad3col=1:d1),
    )
    return ad1, ad2, ad3
end

ad1, ad2, ad3 = make_ads()
expected_n = 102

@testset "create mudata" begin
    md_single = MuData(mod=Dict("ad1" => ad1))
    global md = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
    md.var[!, :mutestcol] = rand(size(md, 2))
    md.obsm["mdtest"] = rand(size(md, 1), 5, 3)

    @test size(md, 1) == expected_n
    @test size(md, 2) == d1 + d2

    modalities = sort(collect(keys(md.mod)))
    @test modalities == ["ad1", "ad2"]

    ad1.var[!, :commoncol] = ["ad1_$i" for i in 1:size(ad1, 2)]
    ad1.var_names = ["ad1_$i" for i in 1:size(ad1, 2)]
    ad2.var[!, :commoncol] = ["ad2_$i" for i in 1:size(ad2, 2)]
    ad2.var_names = ["ad2_$i" for i in 1:size(ad2, 2)]
    md = MuData(mod=Dict("ad1" => ad1, "ad2" => ad2))
    md.var[!, :mutestcol] = rand(size(md, 2))
    md.obsm["mdtest"] = rand(size(md, 1), 5, 3)

    @test size(md, 1) == expected_n
    @test size(md, 2) == d1 + d2
    @test sort(names(md.var)) == ["ad1:testcol", "commoncol", "mutestcol"]

    a1, a2, _ = make_ads()
    a1.obs = DataFrame(demo=[1 for i in 1:size(ad1, 1)])
    a2.obs = DataFrame(demo=[2 for i in 1:size(ad2, 1)])
    a2.obs_names = copy(a1.obs_names)
    m = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => a1, "ad2" => a2)))
    m.obs[!, "demo"] .= "common"
    @test_logs (:warn, warn_msg) update!(m)
    @test sort(names(m.obs)) == ["ad1:demo", "ad2:demo", "demo"]
end

@testset "insert new AnnData" begin
    ad2.var[!, :testcol] = ["ad2_$i" for i in 1:size(ad2, 2)]
    update!(md)
    @test sort(names(md.var)) == ["commoncol", "mutestcol", "testcol"]

    md["ad3"] = ad3
    @test_logs (:warn, warn_msg2) update!(md)
    @test sort(names(md.var)) ==
          ["ad1:testcol", "ad2:testcol", "ad3:ad3col", "commoncol", "mutestcol", "testcol"]
    @test !any(ismissing.(md.var.commoncol))
    @test sum(ismissing.(md.var.mutestcol[reshape(md.varm["ad1"], :)])) ==
          sum(ismissing.(md.var.mutestcol[reshape(md.varm["ad2"], :)])) ==
          0
    @test all(ismissing.(md.var.mutestcol[reshape(md.varm["ad3"], :)]))
end

function test_row_slice(md, i1, n, d, j=:)
    md1, md2, md3, md4 =
        md[i1, :], md[collect(i1), :], md[unique(md.obs_names[i1]), :], md[1:n .∈ (i1,), :]

    @test size(md1) == size(md2) == size(md4) == (length(i1), d) == size(@test_logs (:warn, warn_msg) MuData(mod=md1.mod))
    @test size(md3) == (length(md.obs_names[unique(md.obs_names[i1]), true]), d)
    @test md1.obs_names == md2.obs_names == md4.obs_names == md.obs_names[i1]
    @test sort(unique(md3.obs_names)) == sort(unique(md.obs_names[i1]))

    ad1_names = filter(in(md["ad1"].obs_names), md.obs_names[i1])
    ad2_names = filter(in(md["ad2"].obs_names), md.obs_names[i1])

    ad1_idx = filter(>(0x0), reshape(md.obsmap["ad1"], :)[i1])
    ad2_idx = filter(>(0x0), reshape(md.obsmap["ad2"], :)[i1])

    ad1, ad2 = parent(md["ad1"])[ad1_idx, j], parent(md["ad2"])[ad2_idx, j]

    @test md1["ad1"].X == md2["ad1"].X == md4["ad1"].X == ad1.X
    @test md1["ad2"].X == md2["ad2"].X == md4["ad2"].X == ad2.X
    @test md1["ad1"].obs_names == md2["ad1"].obs_names == md4["ad1"].obs_names == ad1.obs_names
    @test md1["ad2"].obs_names == md2["ad2"].obs_names == md4["ad2"].obs_names == ad2.obs_names
end

function test_md_slicing(md, n, d, j=:)
    @testset "row slice $(1:i)" for i in 2:n
        test_row_slice(md, 1:i, n, d, j)
    end
    @testset "row slice $(i:(i + 1))" for i in 1:(n - 1)
        test_row_slice(md, i:(i + 1), n, d, j)
    end
end

ad1, ad2, _ = make_ads()
md = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
md.var[!, :mutestcol] = rand(size(md, 2))
md.obsm["mdtest"] = rand(size(md, 1), 5, 3)

@testset "slicing mudata" begin
    test_md_slicing(md, expected_n, d1 + d2)
end

@testset "views" begin
    i, j = (26:75, [string(i) for i in 3:7])
    mdview = @view md[i, j]
    submd = md[i, j]
    viewcp = copy(mdview)

    @test mdview.obs_names == submd.obs_names
    @test mdview.var_names == submd.var_names
    @testset "modality $mod" for (mod, ad) in submd.mod
        @test ad.X == mdview[mod].X == viewcp[mod].X
        @test ad.obs_names == mdview[mod].obs_names == viewcp[mod].obs_names
        @test ad.var_names == mdview[mod].var_names == viewcp[mod].var_names
    end
    test_md_slicing(submd, 50, 10, j)
    test_md_slicing(mdview, 50, 10, j)
end

@testset "readwrite" begin
    tempfile1 = joinpath(tmp, "tmp1.h5mu")
    writeh5mu(tempfile1, md)
    read_md = (@test_logs (:warn, warn_msg) readh5mu(tempfile1, backed=false))
    read_md_backed = (@test_logs (:warn, warn_msg) readh5mu(tempfile1, backed=true))

    @test size(md) == size(read_md) == size(read_md_backed)
    @test md.obs_names == read_md.obs_names == read_md_backed.obs_names
    @test md.var_names == read_md.var_names == read_md_backed.var_names
    @test isequal(md.var, read_md.var)
    @test isequal(md.var, read_md_backed.var)
    @test isequal(md.obs, read_md.obs)
    @test isequal(md.obs, read_md_backed.obs)
    @test md["ad1"].X == read_md["ad1"].X == read_md_backed["ad1"].X
    @test md["ad2"].X == read_md["ad2"].X == read_md_backed["ad2"].X
    @test md["ad1"].var == read_md["ad1"].var == read_md_backed["ad1"].var
    @test md["ad2"].var == read_md["ad2"].var == read_md_backed["ad2"].var
    @test md["ad2"].varm == read_md["ad2"].varm == read_md_backed["ad2"].varm
    @test md["ad2"].layers == read_md["ad2"].layers == read_md_backed["ad2"].layers

    md["ad3"] = ad3
    tempfile2 = joinpath(tmp, "tmp2.h5mu")
    writeh5mu(tempfile2, md)
    @test_logs (:warn, warn_msg2) (:warn, warn_msg) update!(md)
    read_md = (@test_logs (:warn, warn_msg2) (:warn, warn_msg) readh5mu(tempfile2, backed=false))
    read_md_backed = (@test_logs (:warn, warn_msg2) (:warn, warn_msg) readh5mu(tempfile2, backed=true))
    @test size(md) == size(read_md) == size(read_md_backed)
    @test md.obs_names == read_md.obs_names == read_md_backed.obs_names
    @test md.var_names == read_md.var_names == read_md_backed.var_names
    @test isequal(md.var, read_md.var)
    @test isequal(md.var, read_md_backed.var)
    @test isequal(md.obs, read_md.obs)
    @test isequal(md.obs, read_md_backed.obs)
end

@testset "obs_var" begin
    ad1, ad2, _ = make_ads()
    ad2.var_names = ["ad2_$i" for i in 1:size(ad2, 2)]
    md = MuData(mod=Dict("ad1" => ad1, "ad2" => ad2))
    tempfile = joinpath(tmp, "tmp.h5mu")
    @testset "obs_global_columns" begin
        for (m, mod) in md.mod
            mod.obs = DataFrame()
            mod.obs[!, "demo"] = repeat([m], size(mod, 1))
        end
        md.obs[!, "demo"] = repeat(["global"], size(md, 1))
        update!(md)
        @test sort(names(md.obs)) == sort(vcat(["$m:demo" for m in keys(md.mod)], ["demo"]))
        writeh5mu(tempfile, md)
        read_md = readh5mu(tempfile, backed=false)
        @test sort(names(read_md.obs)) == sort(vcat(["$m:demo" for m in keys(md.mod)], ["demo"]))
    end

    @testset "var_global_columns" begin
        for (m, mod) in md.mod
            mod.var = DataFrame()
            mod.var[!, "demo"] = repeat([m], size(mod, 2))
        end
        md.var[!, "global"] = repeat(["global_var"], size(md, 2))
        update!(md)
        @test sort(names(md.var)) == sort(["demo", "global"])
        select!(md.var, Not("global"))
        update!(md)
        @test names(md.var) == ["demo"]
        writeh5mu(tempfile, md)
        read_md = readh5mu(tempfile, backed=false)
        @test names(md.var) == ["demo"]
    end
end
