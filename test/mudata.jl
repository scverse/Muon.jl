using Random
using Logging
using DataFrames

Random.seed!(42)

n = 100
d1 = 10
d2 = 20

function make_ads(;
    obs_unique::Bool=false,
    var_unique::Bool=false,
    obs_subset::Bool=false,
    var_subset::Bool=false,
    obs_mod_duplicated::Bool=false,
    var_mod_duplicated::Bool=false,
)
    obs_names = ["obs_$i" for i ∈ 1:n]
    ad1 = AnnData(
        X=rand(n, d1),
        obs_names=obs_names,
        obs=DataFrame(unique_col=["ad1_unique_$i" for i ∈ 1:n], common_col=["ad1_$i" for i ∈ 1:n]),
        var=DataFrame(unique_col=["ad1_unique_$i" for i ∈ 1:d1], common_col=["ad1_$i" for i ∈ 1:d1]),
    )

    ad2 = AnnData(
        X=rand(n, d2),
        obs_names=obs_names,
        obs=DataFrame(nonunique_col=1:n, common_col=["ad2_$i" for i ∈ 1:n]),
        var=DataFrame(nonunique_col=1:d2, common_col=["ad2_$i" for i ∈ 1:d2]),
        layers=Dict("testlayer" => rand(UInt16, n, d2)),
        varm=Dict("test" => rand(Int8, d2, 5, 2, 10)),
        uns=Dict("test" => nothing),
    )

    ad3 = AnnData(
        X=rand(n, d1),
        obs_names=obs_names,
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

    if obs_mod_duplicated
        ad2.obs_names[2] = ad2.obs_names[1] = "testobs"
        ad3.obs_names[2] = "testobs"
    end
    if var_mod_duplicated
        ad2.var_names[2] = ad2.var_names[1] = "testvar"
        ad3.var_names[2] = "testvar"
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

ad1, ad2, ad3 = make_ads(obs_mod_duplicated=true)
expected_obs = 102

warn_msg = "obs_names are not unique. To make them unique, call obs_names_make_unique!"
warn_msg2 = "Duplicated obs_names should not be present in different modalities due to the ambiguity that leads to."
warn_msg3 = "mdata.obs is empty, but has columns. You probably tried to broadcast a scalar to an empty .obs. Dropping empty columns..."
warn_msg4 = "mdata.obs has less rows than it should. You probably assigned a column with the wrong length to an empty .obs. Filling up with missing..."
warn_msg5 = "mdata.obs has more rows than it should. You probably assigned a column with the wrong length to an empty .obs. Subsetting to the first $expected_obs rows..."
warn_msg6 = "var_names are not unique. To make them unique, call var_names_make_unique!"

@testset "create mudata" begin
    md_single = MuData(mod=Dict("ad1" => ad1))
    global md = (@test_logs (:warn, warn_msg) (:warn, warn_msg6) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
    md.obs[:, :mutestcol] .= 1
    @test_logs (:warn, warn_msg3) (:warn, warn_msg) (:warn, warn_msg6) update!(md)
    md.obs[:, :mutestcol] = rand(size(md, 1) - 1)
    @test_logs (:warn, warn_msg4) (:warn, warn_msg) (:warn, warn_msg6) update!(md)
    md.obs = DataFrame(:mutestcol => rand(size(md, 1) + 1))
    @test_logs (:warn, warn_msg5) (:warn, warn_msg) (:warn, warn_msg6) update!(md)
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

    a1, a2, _ = make_ads(obs_mod_duplicated=true)
    a1.obs = DataFrame(demo=[1 for i ∈ 1:size(ad1, 1)])
    a2.obs = DataFrame(demo=[2 for i ∈ 1:size(ad2, 1)])
    a2.obs_names = copy(a1.obs_names)
    m = @test_logs (:warn, warn_msg6) MuData(mod=Dict("ad1" => a1, "ad2" => a2))
    m.obs[!, "demo"] = repeat(["common"], size(m, 1))
    @test_logs (:warn, warn_msg6) update!(m)
    @test names(m.obs) == ["demo"]
end

@testset "insert new AnnData" begin
    md["ad3"] = ad3
    @test_logs (:warn, warn_msg2) (:warn, warn_msg) (:warn, warn_msg6) update!(md)
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

ad1, ad2, ad3 = make_ads(obs_mod_duplicated=true)
md = (@test_logs (:warn, warn_msg) (:warn, warn_msg6) MuData(mod=Dict("ad1" => ad1, "ad2" => ad2)))
md.var[!, :mutestcol] = rand(size(md, 2))
md.obsm["mdtest"] = rand(size(md, 1), 5, 3)
ad2.uns["test"] = nothing
md.uns["mdtest"] = nothing

@testset "slicing mudata" begin
    with_logger(NullLogger()) do
        test_md_slicing(md, expected_obs, d1 + d2)
    end
end

@testset "views" begin
    with_logger(NullLogger()) do
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
end

@testset "readwrite" begin
    @testset "$backend" for (readfun, writefun, backend) ∈
                            ((readh5mu, writeh5mu, "HDF5"), (readzarrmu, writezarrmu, "Zarr"))
        cmd = md[:, :]
        tempfile1 = tempname()
        writefun(tempfile1, cmd)
        read_md = (@test_logs (:warn, warn_msg) (:warn, warn_msg6) readfun(tempfile1, backed=false))
        read_md_backed = (@test_logs (:warn, warn_msg) (:warn, warn_msg6) readfun(tempfile1, backed=true))

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
        @test isnothing(read_md.uns["mdtest"])
        @test isnothing(read_md["ad2"].uns["test"])

        cmd["ad3"] = ad3
        tempfile2 = tempname()
        writefun(tempfile2, cmd)
        @test_logs (:warn, warn_msg2) (:warn, warn_msg) (:warn, warn_msg6) update!(cmd)
        read_md = (@test_logs (:warn, warn_msg2) (:warn, warn_msg) (:warn, warn_msg6) readfun(tempfile2, backed=false))
        read_md_backed =
            (@test_logs (:warn, warn_msg2) (:warn, warn_msg) (:warn, warn_msg6) readfun(tempfile2, backed=true))
        @test size(cmd) == size(read_md) == size(read_md_backed)
        @test cmd.obs_names == read_md.obs_names == read_md_backed.obs_names
        @test cmd.var_names == read_md.var_names == read_md_backed.var_names
        @test isequal(cmd.var, read_md.var)
        @test isequal(cmd.var, read_md_backed.var)
        @test isequal(cmd.obs, read_md.obs)
        @test isequal(cmd.obs, read_md_backed.obs)
    end
end

function make_mdata(ads...; axis)
    modalities = Dict("ad$i" => ad for (i, ad) ∈ enumerate(ads))
    mdata = MuData(mod=modalities, axis=axis)

    mdata.obs[!, "batch"] = rand(("a", "b", "c"), size(mdata, 1))
    mdata.var[!, "batch"] = rand(("d", "e", "f"), size(mdata, 2))

    mdata.obsm["test"] = randn(size(mdata, 1))
    mdata.varm["test"] = randn(size(mdata, 2))

    return mdata
end

@testset "update" begin
    for axis ∈ (0x1, 0x2), obs_mod_duplicated ∈ (false, true), obs_subset ∈ (false, true)
        ad1, ad2, ad3 = make_ads(obs_mod_duplicated=obs_mod_duplicated, obs_subset=obs_subset)

        attr = Symbol(axis == 0x1 ? "obs" : "var")
        oattr = Symbol(axis == 0x1 ? "var" : "obs")
        attrm = Symbol("$(attr)m")
        oattrm = Symbol("$(oattr)m")
        namesattr = Symbol("$(attr)_names")
        onamesattr = Symbol("$(oattr)_names")
        mapattr = Symbol("$(attr)map")
        omapattr = Symbol("$(oattr)map")
        oaxis = 0x3 - axis

        @testset "axis=$axis, obs_mod_duplicated=$obs_mod_duplicated, obs_subset=$obs_subset" begin
            with_logger(NullLogger()) do
                @testset "simple" begin
                    md = make_mdata(ad1, ad2, ad3, axis=axis)

                    # names along non-axis are concatenated
                    @test size(md, oaxis) == sum(size(mod, oaxis) for mod ∈ values(md.mod))
                    @test getproperty(md, onamesattr) ==
                          vcat((getproperty(mod, onamesattr) for mod ∈ values(md.mod))...)

                    # names along axis are unioned
                    axisnames = vcat(
                        getproperty(ad2, namesattr),
                        setdiff(
                            union((getproperty(ad, namesattr) for ad ∈ (ad1, ad3))...),
                            getproperty(ad2, namesattr),
                        ),
                    )
                    @test size(md, axis) == length(axisnames)
                    @test sort(getproperty(md, namesattr)) == sort(axisnames)

                    # test for correct order
                    @test getproperty(md, namesattr)[begin:size(ad1, axis)] == getproperty(ad1, namesattr)
                end
                @testset "add modality" begin
                    md = MuData(mod=Dict("ad1" => ad1), axis=axis)
                    for (modname, ad) ∈ (("ad2", ad2), ("ad3", ad3))
                        old_attrnames = getproperty(md, namesattr)
                        old_oattrnames = getproperty(md, onamesattr)

                        some_obs_names = md.obs_names[begin:2]
                        md.obsm["test"] = randn(size(md, 1))
                        true_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]

                        md.mod[modname] = ad
                        update!(md)

                        test_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]
                        if axis == 0x2
                            @test sum(ismissing.(md.obsm["test"])) == size(ad, 1)
                            @test all(ismissing.(md.obsm["test"][(end - size(ad, 1) + 1):end]))
                            @test all(.!ismissing.(md.obsm["test"][begin:size(ad, 1)]))
                            @test test_obsm_values[.!ismissing.(test_obsm_values)] == true_obsm_values
                        else
                            @test test_obsm_values == true_obsm_values
                        end

                        attrnames = getproperty(md, namesattr)
                        oattrnames = getproperty(md, onamesattr)
                        @test attrnames[begin:length(old_attrnames)] == old_attrnames
                        @test oattrnames[begin:length(old_oattrnames)] == old_oattrnames

                        modattrnames = getproperty(ad, namesattr)
                        @test attrnames == vcat(old_attrnames, modattrnames[modattrnames .∉ (old_attrnames,)])
                        @test oattrnames == vcat(old_oattrnames, getproperty(ad, onamesattr))
                    end
                end
                @testset "delete modality" begin
                    md = make_mdata(ad1, ad2, ad3, axis=axis)
                    modnames = collect(keys(md.mod))

                    fullbatch = getproperty(md, attr)[!, "batch"]
                    fullobatch = getproperty(md, oattr)[!, "batch"]
                    fulltestm = getproperty(md, attrm)["test"]
                    fullotestm = getproperty(md, oattrm)["test"]
                    keptmask =
                        vec(getproperty(md, mapattr)[modnames[2]] .> 0) .|
                        vec(getproperty(md, mapattr)[modnames[3]] .> 0)
                    keptomask =
                        vec(getproperty(md, omapattr)[modnames[2]] .> 0) .|
                        vec(getproperty(md, omapattr)[modnames[3]] .> 0)

                    delete!(md.mod, modnames[1])
                    update!(md)

                    @test size(md, oaxis) == sum(size(ad, oaxis) for ad ∈ values(md.mod))
                    @test getproperty(md, attr)[!, "batch"] == fullbatch[keptmask]
                    @test getproperty(md, oattr)[!, "batch"] == fullobatch[keptomask]
                    @test getproperty(md, attrm)["test"] == fulltestm[keptmask, :]
                    @test getproperty(md, oattrm)["test"] == fullotestm[keptomask, :]

                    fullbatch = getproperty(md, attr)[!, "batch"]
                    fullobatch = getproperty(md, oattr)[!, "batch"]
                    fulltestm = getproperty(md, attrm)["test"]
                    fullotestm = getproperty(md, oattrm)["test"]
                    keptmask = vec(getproperty(md, mapattr)[modnames[2]] .> 0)
                    keptomask = vec(getproperty(md, omapattr)[modnames[2]] .> 0)
                    delete!(md.mod, modnames[3])
                    update!(md)

                    @test size(md, oaxis) == sum(size(ad, oaxis) for ad ∈ values(md.mod))
                    @test getproperty(md, attr)[!, "batch"] == fullbatch[keptmask]
                    @test getproperty(md, oattr)[!, "batch"] == fullobatch[keptomask]
                    @test getproperty(md, attrm)["test"] == fulltestm[keptmask, :]
                    @test getproperty(md, oattrm)["test"] == fullotestm[keptomask, :]
                end
                @testset "update intersecting" begin # same obsnames for all modalities, intersecting var_names which are unique in each modality
                    modnames = ("ad1", "ad2", "ad3")
                    for (modname, ad) ∈ zip(modnames, (ad1, ad2, ad3))
                        getproperty(ad, onamesattr)[2:end] .= ("$(modname)_$(oattr)_$(i)" for i ∈ 2:size(ad, oaxis))
                    end
                    md = MuData(mod=Dict(modname => ad for (modname, ad) ∈ zip(modnames, (ad1, ad2, ad3))), axis=axis)

                    # names along non-axis are concatenated
                    @test size(md, oaxis) == sum(size(ad, oaxis) for ad ∈ values(md.mod))
                    @test getproperty(md, onamesattr) == vcat((getproperty(ad, onamesattr) for ad ∈ values(md.mod))...)

                    # names along axis are unioned
                    axisnames = vcat(
                        getproperty(ad1, namesattr),
                        (
                            getproperty(ad, namesattr)[getproperty(ad, namesattr) .∉ (getproperty(ad1, namesattr),)] for ad ∈ (ad1, ad2)
                        )...,
                    )
                    @test size(md, axis) == length(axisnames)
                    @test getproperty(md, namesattr) == axisnames
                end
                @testset "update after in-place filtering" begin
                    md = make_mdata(ad1, ad2, ad3, axis=axis)
                    old_obsnames = md.obs_names
                    old_varnames = md.var_names

                    some_obs_names = md.obs_names[begin:2]
                    true_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]

                    md.mod["ad3"] = ad3[randsubseq(1:size(ad3, 1), 0.02), :]
                    update!(md)

                    @test sum(ismissing.(md.obs[!, "batch"])) == 0
                    @test md.var_names == old_varnames
                    if axis == 0x1 # check if the order is preserved
                        @test md.obs_names == old_obsnames[old_obsnames .∈ (md.obs_names,)]
                    end

                    test_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]

                    @test test_obsm_values == true_obsm_values
                    @test sum(ismissing.(md.obs[!, "batch"])) == 0
                end
                @testset "update after obs reordered" begin
                    md = make_mdata(ad1, ad2, ad3, axis=axis)
                    some_obs_names = md.obs_names[begin:2]
                    true_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]

                    md.mod["ad1"] = md.mod["ad1"][end:-1:begin, :]
                    update!(md)

                    test_obsm_values = md.obsm["test"][sort(md.obs_names[some_obs_names, true])]

                    @test test_obsm_values == true_obsm_values
                    @test sum(ismissing.(md.obs[!, "batch"])) == 0
                end
            end
        end
    end
end

@testset "pull/push multimodal" begin
    for unique ∈ (true, false), subset ∈ (true, false), axis ∈ (0x1, 0x2)
        attrname = axis == 0x1 ? "var" : "obs"
        attr = Symbol(attrname)
        attrmap = Symbol("$(attrname)map")
        pull_attr! = getproperty(Main, Symbol("pull_$(attr)!"))
        push_attr! = getproperty(Main, Symbol("push_$(attr)!"))

        oaxis = 0x3 - axis
        oattrname = axis == 0x1 ? "obs" : "var"
        oattr = Symbol(oattrname)
        oattrmap = Symbol("$(oattrname)map")
        pull_oattr! = getproperty(Main, Symbol("pull_$(oattr)!"))
        push_oattr! = getproperty(Main, Symbol("push_$(oattr)!"))
        @testset "unique=$unique, subset=$subset, attr=$attr" begin
            ad1, ad2, ad3 = make_ads(; (Symbol("$(attr)_unique") => unique, Symbol("$(attr)_subset") => subset)...)
            if unique && axis == 0x2
                md = (@test_nowarn MuData(mod=Dict("ad1" => ad1, "ad2" => ad2, "ad3" => ad3), axis=axis))
            else
                md = with_logger(NullLogger()) do   # warning depends on the RNG, and differs between Julia versions
                    MuData(mod=Dict("ad1" => ad1, "ad2" => ad2, "ad3" => ad3), axis=axis)
                end
            end

            @testset "pull_$attr" begin
                pull_attr!(md)
                @test sort(names(getproperty(md, attr))) ==
                      ["ad1:unique_col", "ad2:nonunique_col", "ad3:nonunique_col", "common_col"]

                for (mod, ad) ∈ md.mod
                    map = vec(getproperty(md, attrmap)[mod])
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
                odf = getproperty(md, oattr)
                @test sort(names(odf)) == [
                    "ad1:common_col",
                    "ad1:unique_col",
                    "ad2:common_col",
                    "ad2:nonunique_col",
                    "ad3:common_col",
                    "ad3:nonunique_col",
                ]

                common_cols = [odf[!, "$mod:common_col"] for mod ∈ keys(md.mod)]
                mask = mapreduce(x -> .!ismissing.(x), .&, common_cols)
                common_cols = map(col -> getindex.(col, range.(5, lastindex.(col))), getindex.(common_cols, (mask,)))
                @test allequal(common_cols)

                setproperty!(md, oattr, DataFrame())
                @test_throws ArgumentError pull_oattr!(md, join_common=true)

                setproperty!(md, oattr, DataFrame())
                @test_throws ArgumentError pull_oattr!(md, join_nonunique=true)
            end
            @testset "push_$attr" begin
                getproperty(md, attr)[:, :pushed] = rand(Int, size(md, oaxis))
                getproperty(md, attr)[:, "ad3:ad3_pushed"] = rand(Int, size(md, oaxis))
                push_attr!(md)
                for (mod, ad) ∈ md.mod
                    @test columnindex(getproperty(ad, attr), :pushed) > 0

                    map = vec(getproperty(md, attrmap)[mod])
                    mask = map .> 0
                    @test getproperty(md, attr)[!, :pushed][mask] == getproperty(ad, attr)[!, :pushed][map[mask]]
                end
                @test columnindex(getproperty(ad1, attr), :ad3_pushed) == 0
                @test columnindex(getproperty(ad2, attr), :ad3_pushed) == 0
                @test columnindex(getproperty(ad3, attr), :ad3_pushed) > 0

                map = vec(getproperty(md, attrmap)["ad3"])
                mask = map .> 0
                @test getproperty(md, attr)[!, "ad3:ad3_pushed"][mask] ==
                      getproperty(ad3, attr)[!, :ad3_pushed][map[mask]]
            end
            @testset "push_$oattr" begin
                getproperty(md, oattr)[:, :pushed] = rand(Int, size(md, axis))
                getproperty(md, oattr)[:, "ad3:ad3_pushed"] .= rand(Int, size(md, axis))
                push_oattr!(md)
                for (mod, ad) ∈ md.mod
                    @test columnindex(getproperty(ad, oattr), :pushed) > 0

                    map = vec(getproperty(md, oattrmap)[mod])
                    mask = map .> 0
                    @test getproperty(md, oattr)[!, :pushed][mask] == getproperty(ad, oattr)[!, :pushed][map[mask]]
                end
                @test columnindex(getproperty(ad1, oattr), :ad3_pushed) == 0
                @test columnindex(getproperty(ad2, oattr), :ad3_pushed) == 0
                @test columnindex(getproperty(ad3, oattr), :ad3_pushed) > 0

                map = vec(getproperty(md, oattrmap)["ad3"])
                mask = map .> 0
                @test getproperty(md, oattr)[!, "ad3:ad3_pushed"][mask] ==
                      getproperty(ad3, oattr)[!, :ad3_pushed][map[mask]]
            end
        end
    end
end
