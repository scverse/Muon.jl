n = 100
d1 = 10
d2 = 20

warn_msg = "Cannot join columns with the same name because var_names are intersecting."
obs_names = ["obs_$i" for i in 1:n]
x = rand(Float64, (n, d1))
ad = AnnData(X=x, obs_names=copy(obs_names))

obs_names[10] = "testobs"
obs_names[n] = "testobs"
expected_n = 102

y = rand(Float64, (n, d2))
ad2 = AnnData(X=y, obs_names=obs_names)

md = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => ad, "ad2" => ad2)))

@testset "create mudata" begin
    @test size(md, 1) == expected_n
    @test size(md, 2) == d1 + d2

    modalities = sort(collect(keys(md.mod)))
    @test modalities == ["ad1", "ad2"]
end

function test_row_slice(md, i1, n, d, j=:)
    md1, md2, md3, md4 =
        md[i1, :], md[collect(i1), :], md[unique(md.obs_names[i1]), :], md[1:n .∈ (i1,), :]

    @test size(md1) ==
          size(md2) ==
          size(md4) ==
          (length(i1), d) ==
          size(@test_logs (:warn, warn_msg) MuData(mod=md1.mod))
    @test size(md3) == (length(md.obs_names[unique(md.obs_names[i1]), true]), d)
    @test md1.obs_names == md2.obs_names == md4.obs_names == md.obs_names[i1]
    @test sort(unique(md3.obs_names)) == sort(unique(md.obs_names[i1]))

    ad1_names = filter(in(md["ad1"].obs_names), md.obs_names[i1])
    ad2_names = filter(in(md["ad2"].obs_names), md.obs_names[i1])

    ad1_idx = filter(>(0x0), md.obsm["ad1"][i1])
    ad2_idx = filter(>(0x0), md.obsm["ad2"][i1])

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
