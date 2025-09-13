using Random
using Logging
using DataFrames

Random.seed!(42)

n = 100
d1 = 10
d2 = 20

function make_ads(; obs_unique::Bool=false, var_unique::Bool=false, obs_subset::Bool=false, var_subset::Bool=false)
    obs_names = ["obs_$i" for i ∈ 1:n]
    ad1 = AnnData(
        X=rand(n, d1),
        obs_names=copy(obs_names),
        obs=DataFrame(unique_col=["ad1_unique_$i" for i ∈ 1:n], common_col=["ad1_$i" for i ∈ 1:n]),
        var=DataFrame(unique_col=["ad1_unique_$i" for i ∈ 1:d1], common_col=["ad1_$i" for i ∈ 1:d1]),
    )

    obs_names[10] = obs_names[n] = "testobs"
    ad2 = AnnData(
        X=rand(n, d2),
        obs_names=copy(obs_names),
        obs=DataFrame(nonunique_col=1:n, common_col=["ad2_$i" for i ∈ 1:n]),
        var=DataFrame(nonunique_col=1:d2, common_col=["ad2_$i" for i ∈ 1:d2]),
        layers=Dict("testlayer" => rand(UInt16, n, d2)),
        varm=Dict("test" => rand(Int8, d2, 5, 2, 10)),
    )

    obs_names[10] = "obs_10"
    ad3 = AnnData(
        X=rand(n, d1),
        obs_names=copy(obs_names),
        obs=DataFrame(nonunique_col=repeat([true, false], n ÷ 2), common_col=["ad3_$i" for i ∈ 1:n]),
        var=DataFrame(nonunique_col=repeat([true, false], d1 ÷ 2), common_col=["ad3_$i" for i ∈ 1:d1]),
    )

    if obs_unique
        ad1.obs_names = ["obs1_$i" for i ∈ 1:size(ad1, 1)]
        ad2.obs_names = ["obs2_$i" for i ∈ 1:size(ad2, 1)]
        ad3.obs_names = ["obs3_$i" for i ∈ 1:size(ad3, 1)]
    end
    if var_unique
        ad1.var_names = ["var1_$i" for i ∈ 1:size(ad1, 2)]
        ad2.var_names = ["var2_$i" for i ∈ 1:size(ad2, 2)]
        ad3.var_names = ["var3_$i" for i ∈ 1:size(ad3, 2)]
    end

    obs_idx = shuffle(1:size(ad3, 1))
    var_idx = shuffle(1:size(ad3, 2))
    if obs_subset
        obs_idx = obs_idx[1:(length(obs_idx) ÷ 2)]
    end
    if var_subset
        var_idx = var_idx[1:(length(var_idx) ÷ 2)]
    end
    ad3 = ad3[obs_idx, var_idx]

    return ad1, ad2, ad3
end

ad1, ad2, ad3 = make_ads()
expected_obs = 102

warn_msg = "obs_names are not unique. To make them unique, call obs_names_make_unique!"
warn_msg2 = "Duplicated obs_names should not be present in different modalities due to the ambiguity that leads to."
warn_msg3 = "mdata.obs is empty, but has columns. You probably tried to broadcast a scalar to an empty .obs. Dropping empty columns..."
warn_msg4 = "mdata.obs has less rows than it should. You probably assigned a column with the wrong length to an empty .obs. Filling up with missing..."
warn_msg5 = "mdata.obs has more rows than it should. You probably assigned a column with the wrong length to an empty .obs. Subsetting to the first $expected_obs rows..."

@testset "create mudata" begin
    md_single = MuData(mod=Dict("ad1" => ad1))
    global md = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
    md.obs[:, :mutestcol] .= 1
    @test_logs (:warn, warn_msg) (:warn, warn_msg3) update!(md)
    md.obs[:, :mutestcol] = rand(size(md, 1) - 1)
    @test_logs (:warn, warn_msg) (:warn, warn_msg4) update!(md)
    md.obs = DataFrame(:mutestcol => rand(size(md, 1) + 1))
    @test_logs (:warn, warn_msg) (:warn, warn_msg5) update!(md)
    md.var[!, :mutestcol] = rand(size(md, 2))

    @test names(md.var) == ["mutestcol"]
    @test size(md, 1) == expected_obs
    @test size(md, 2) == d1 + d2

    modalities = sort(collect(keys(md.mod)))
    @test modalities == ["ad1", "ad2"]

    for (mod, ad) ∈ md.mod
        obsmap = vec(md.obsmap[mod])
        mask = obsmap .> 0
        @test md.obs_names[mask] == ad.obs_names[obsmap[mask]]

        varmap = vec(md.varmap[mod])
        mask = varmap .> 0
        @test md.var_names[mask] == ad.var_names[varmap[mask]]
    end

    a1, a2, _ = make_ads()
    a1.obs = DataFrame(demo=[1 for i ∈ 1:size(ad1, 1)])
    a2.obs = DataFrame(demo=[2 for i ∈ 1:size(ad2, 1)])
    a2.obs_names = copy(a1.obs_names)
    m = (@test_nowarn MuData(mod=Dict("ad1" => a1, "ad2" => a2)))
    m.obs[!, "demo"] = repeat(["common"], size(m, 1))
    @test_nowarn update!(m)
    @test names(m.obs) == ["demo"]
end

@testset "insert new AnnData" begin
    md["ad3"] = ad3
    @test_logs (:warn, warn_msg2) (:warn, warn_msg) update!(md)
    @test sort(names(md.var)) == ["mutestcol"]
    @test sum(ismissing.(md.var.mutestcol[vec(md.varm["ad1"])])) ==
          sum(ismissing.(md.var.mutestcol[vec(md.varm["ad2"])])) ==
          0
    @test all(ismissing.(md.var.mutestcol[vec(md.varm["ad3"])]))
end

function test_row_slice(md, i1, n, d, j=:)
    md1, md2, md3, md4 = md[i1, :], md[collect(i1), :], md[unique(md.obs_names[i1]), :], md[1:n .∈ (i1,), :]

    @test size(md1) == size(md2) == size(md4) == (length(i1), d) == size(MuData(mod=md1.mod))
    @test size(md3) == (length(md.obs_names[unique(md.obs_names[i1]), true]), d)
    @test md1.obs_names == md2.obs_names == md4.obs_names == md.obs_names[i1]
    @test sort(unique(md3.obs_names)) == sort(unique(md.obs_names[i1]))

    ad1_names = filter(in(md["ad1"].obs_names), md.obs_names[i1])
    ad2_names = filter(in(md["ad2"].obs_names), md.obs_names[i1])

    ad1_idx = filter(>(0x0), vec(md.obsmap["ad1"])[i1])
    ad2_idx = filter(>(0x0), vec(md.obsmap["ad2"])[i1])
    ad1, ad2 = md["ad1"][ad1_idx, j], md["ad2"][ad2_idx, j]

    parent_ad1_idx = filter(>(0x0), vec(parent(md).obsmap["ad1"])[parentindices(md)[1][i1]])
    parent_ad2_idx = filter(>(0x0), vec(parent(md).obsmap["ad2"])[parentindices(md)[1][i1]])
    parent_ad1, parent_ad2 = parent(md)["ad1"][parent_ad1_idx, j], parent(md["ad2"])[parent_ad2_idx, j]

    @test md1["ad1"].X == md2["ad1"].X == md4["ad1"].X == ad1.X
    @test md1["ad2"].X == md2["ad2"].X == md4["ad2"].X == ad2.X
    @test md1["ad1"].obs_names == md2["ad1"].obs_names == md4["ad1"].obs_names == ad1.obs_names == parent_ad1.obs_names
    @test md1["ad2"].obs_names == md2["ad2"].obs_names == md4["ad2"].obs_names == ad2.obs_names == parent_ad2.obs_names
end

function test_md_slicing(md, n, d, j=:)
    @testset "row slice $(1:i)" for i ∈ 2:n
        test_row_slice(md, 1:i, n, d, j)
    end
    @testset "row slice $(i:(i + 1))" for i ∈ 1:(n - 1)
        test_row_slice(md, i:(i + 1), n, d, j)
    end
end

ad1, ad2, _ = make_ads()
md = (@test_logs (:warn, warn_msg) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
md.var[!, :mutestcol] = rand(size(md, 2))
md.obsm["mdtest"] = rand(size(md, 1), 5, 3)

@testset "slicing mudata" begin
    test_md_slicing(md, expected_obs, d1 + d2)
end

@testset "views" begin
    i, j = (26:75, [string(i) for i ∈ 3:7])
    mdview = @view md[i, j]
    submd = md[i, j]
    viewcp = copy(mdview)

    @test mdview.obs_names == submd.obs_names
    @test mdview.var_names == submd.var_names
    @testset "modality $mod" for (mod, ad) ∈ submd.mod
        @test ad.X == mdview[mod].X == viewcp[mod].X
        @test ad.obs_names == mdview[mod].obs_names == viewcp[mod].obs_names
        @test ad.var_names == mdview[mod].var_names == viewcp[mod].var_names
    end
    test_md_slicing(submd, 50, 10, j)
    test_md_slicing(mdview, 50, 10, j)
end

@testset "readwrite" begin
    @testset "$backend" for (readfun, writefun, backend) ∈
                            ((readh5mu, writeh5mu, "HDF5"), (readzarrmu, writezarrmu, "Zarr"))
        cmd = md[:, :]
        tempfile1 = tempname()
        writefun(tempfile1, cmd)
        read_md = (@test_logs (:warn, warn_msg) readfun(tempfile1, backed=false))
        read_md_backed = (@test_logs (:warn, warn_msg) readfun(tempfile1, backed=true))

        @test size(cmd) == size(read_md) == size(read_md_backed)
        @test cmd.obs_names == read_md.obs_names == read_md_backed.obs_names
        @test cmd.var_names == read_md.var_names == read_md_backed.var_names
        @test cmd.obsm == read_md.obsm == read_md_backed.obsm
        @test isequal(cmd.var, read_md.var)
        @test isequal(cmd.var, read_md_backed.var)
        @test isequal(cmd.obs, read_md.obs)
        @test isequal(cmd.obs, read_md_backed.obs)
        @test cmd["ad1"].X == read_md["ad1"].X == read_md_backed["ad1"].X
        @test cmd["ad2"].X == read_md["ad2"].X == read_md_backed["ad2"].X
        @test cmd["ad1"].var == read_md["ad1"].var == read_md_backed["ad1"].var
        @test cmd["ad2"].var == read_md["ad2"].var == read_md_backed["ad2"].var
        @test cmd["ad2"].varm == read_md["ad2"].varm == read_md_backed["ad2"].varm
        @test cmd["ad2"].layers == read_md["ad2"].layers == read_md_backed["ad2"].layers

        cmd["ad3"] = ad3
        tempfile2 = tempname()
        writefun(tempfile2, cmd)
        @test_logs (:warn, warn_msg2) (:warn, warn_msg) update!(cmd)
        read_md = (@test_logs (:warn, warn_msg2) (:warn, warn_msg) readfun(tempfile2, backed=false))
        read_md_backed = (@test_logs (:warn, warn_msg2) (:warn, warn_msg) readfun(tempfile2, backed=true))
        @test size(cmd) == size(read_md) == size(read_md_backed)
        @test cmd.obs_names == read_md.obs_names == read_md_backed.obs_names
        @test cmd.var_names == read_md.var_names == read_md_backed.var_names
        @test isequal(cmd.var, read_md.var)
        @test isequal(cmd.var, read_md_backed.var)
        @test isequal(cmd.obs, read_md.obs)
        @test isequal(cmd.obs, read_md_backed.obs)
    end
end

@testset "pull/push multimodal" begin
    for unique ∈ (true, false), subset ∈ (true, false), axis ∈ (0x1, 0x2)
        attrname = axis == 0x1 ? "var" : "obs"
        attr = Symbol(attrname)
        pull_attr! = getproperty(Main, Symbol("pull_$(attr)!"))
        push_attr! = getproperty(Main, Symbol("push_$(attr)!"))

        oaxis = 0x3 - axis
        oattrname = axis == 0x1 ? "obs" : "var"
        oattr = Symbol(oattrname)
        pull_oattr! = getproperty(Main, Symbol("pull_$(oattr)!"))
        push_oattr! = getproperty(Main, Symbol("push_$(oattr)!"))
        @testset "unique=$unique, subset=$subset, attr=$attr" begin
            ad1, ad2, ad3 = make_ads(; (Symbol("$(attr)_unique") => unique, Symbol("$(attr)_subset") => subset)...)
            if unique && axis == 0x2
                md = (@test_nowarn MuData(mod=Dict("ad1" => ad1, "ad2" => ad2, "ad3" => ad3), axis=axis))
            else
                md = with_logger(NullLogger()) do # warning depends on the RNG, and differs between Julia versions
                    MuData(mod=Dict("ad1" => ad1, "ad2" => ad2, "ad3" => ad3), axis=axis)
                end
            end

            @testset "pull_$attr" begin
                pull_attr!(md)
                @test sort(names(getproperty(md, attr))) ==
                      ["ad1:unique_col", "ad2:nonunique_col", "ad3:nonunique_col", "common_col"]

                for (mod, ad) ∈ md.mod
                    map = vec(getproperty(md, Symbol("$(attr)map"))[mod])
                    mask = map .> 0
                    @test all(startswith.(getproperty(md, attr)[mask, :common_col], mod))
                    @test getproperty(md, attr)[mask, :common_col] == getproperty(ad, attr)[map[mask], :common_col]
                end

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, columns="common_col")
                @test names(getproperty(md, attr)) == ["common_col"]
                @test !any(ismissing.(getproperty(md, attr)[!, :common_col]))

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, columns=["common_col"], mods="ad2")
                @test names(getproperty(md, attr)) == ["ad2:common_col"]
                @test sum(.~ismissing.(getproperty(md, attr)[!, "ad2:common_col"])) == size(ad2, oaxis)

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, common=true, nonunique=true, unique=false)
                @test sort(names(getproperty(md, attr))) == ["ad2:nonunique_col", "ad3:nonunique_col", "common_col"]

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, common=false, nonunique=false, unique=true)
                @test names(getproperty(md, attr)) == ["ad1:unique_col"]

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, common=false, unique=false)
                @test sort(names(getproperty(md, attr))) == ["ad2:nonunique_col", "ad3:nonunique_col"]

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, common=false, unique=false, join_nonunique=true)
                @test names(getproperty(md, attr)) == ["nonunique_col"]

                setproperty!(md, attr, DataFrame())
                pull_attr!(md, common=false, nonunique=false, unique=true, prefix_unique=false)
                @test names(getproperty(md, attr)) == ["unique_col"]
            end
            @testset "pull_$oattr" begin
                pull_oattr!(md)
                @test sort(names(getproperty(md, oattr))) == [
                    "ad1:common_col",
                    "ad1:unique_col",
                    "ad2:common_col",
                    "ad2:nonunique_col",
                    "ad3:common_col",
                    "ad3:nonunique_col",
                ]

                setproperty!(md, oattr, DataFrame())
                @test_throws ArgumentError pull_oattr!(md, join_common=true)

                setproperty!(md, oattr, DataFrame())
                @test_throws ArgumentError pull_oattr!(md, join_nonunique=true)
            end
            @testset "push_$attr" begin
                getproperty(md, attr)[:, :pushed] = fill(true, size(md, oaxis))
                getproperty(md, attr)[:, "ad2:ad2_pushed"] .= true
                push_attr!(md)
                for ad ∈ values(md.mod)
                    @test columnindex(getproperty(ad, attr), :pushed) > 0
                end
                @test columnindex(getproperty(ad1, attr), :ad2_pushed) == 0
                @test columnindex(getproperty(ad2, attr), :ad2_pushed) > 0
                @test columnindex(getproperty(ad3, attr), :ad2_pushed) == 0
                @test all(getproperty(ad2, attr)[!, :ad2_pushed])
            end
            @testset "push_$oattr" begin
                getproperty(md, oattr)[:, :pushed] = fill(true, size(md, axis))
                getproperty(md, oattr)[:, "ad2:ad2_pushed"] .= true
                push_oattr!(md)
                for ad ∈ values(md.mod)
                    @test columnindex(getproperty(ad, oattr), :pushed) > 0
                end
                @test columnindex(getproperty(ad1, oattr), :ad2_pushed) == 0
                @test columnindex(getproperty(ad2, oattr), :ad2_pushed) > 0
                @test columnindex(getproperty(ad3, oattr), :ad2_pushed) == 0
                @test all(getproperty(ad2, oattr)[!, :ad2_pushed])
            end
        end
    end
end
