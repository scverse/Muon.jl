n = 100
d = 10
x = rand(Float64, (n, d))
obs_names = ["obs_$i" for i in 1:n]
ad = AnnData(X=x, obs_names=obs_names)

@testset "create anndata" begin
    @test size(ad) == (n, d)
end

function test_row_slice(ad, i1, n, d, x)
    ad1, ad2, ad3, ad4 = ad[i1, :], ad[collect(i1), :], ad[ad.obs_names[i1], :], ad[1:n .∈ (i1,), :]

    @test size(ad1) == size(ad2) == size(ad3) == size(ad4) == (length(i1), d)
    @test ad1.X == ad2.X == ad3.X == ad4.X == x[i1, :]
    @test ad1.obs_names == ad2.obs_names == ad3.obs_names == ad4.obs_names == ad.obs_names[i1]
end

function test_ad_slicing(ad, n, d, x)
    @testset "row slice $(1:i)" for i in 2:n
        test_row_slice(ad, 1:i, n, d, x)
    end
    @testset "row slice $(i:(i + 1))" for i in 1:(n - 1)
        test_row_slice(ad, i:(i + 1), n, d, x)
    end
end

@testset "slicing anndata" begin
    test_ad_slicing(ad, n, d, x)
end

@testset "views" begin
    i, j = (26:75, [string(i) for i in 3:7])
    adview = @view ad[i, j]
    subad = ad[i, j]
    @test adview.X == x[26:75, 3:7]
    @test adview.obs_names == subad.obs_names == ad.obs_names[i]
    @test adview.var_names == subad.var_names == ad.var_names[3:7]
    @test copy(adview).X == subad.X
    test_ad_slicing(subad, 50, 5, x[i, 3:7])
end

@testset "unique names" begin
    @test_logs (:info,) var_names_make_unique!(ad)
    @test_logs (:info,) obs_names_make_unique!(ad)
    ad2 = deepcopy(ad)
    ad2.var_names[3] == "10"
    ad2.obs_names[90] == "obs_30"
    var_names_make_unique!(ad2)
    obs_names_make_unique!(ad2)
    @test allunique(ad2.var_names)
    @test allunique(ad2.obs_names)
    ad2.var_names[10] = "10-1"
    ad2.var_names[3] = "10"
    ad2.var_names[4] = "10"
    ad2.obs_names[11] = "obs_10-1"
    ad2.obs_names[10] = "obs_10"
    ad2.obs_names[9] = "obs_10"
    @test_logs (:warn,) var_names_make_unique!(ad2)
    @test_logs (:warn,) obs_names_make_unique!(ad2)
    @test allunique(ad2.var_names)
    @test allunique(ad2.obs_names)
end

@testset "DataFrame conversion" begin
    using DataFrames
    df = DataFrame(ad)
    @test names(df) == ["obs"; ad.var_names]
    @test df.obs == ad.obs_names
    ad.var_names[3] = "10"
    @test_throws ArgumentError DataFrame(ad)
    @test_throws ArgumentError DataFrame(ad, columns=:foo)
    @test_throws ArgumentError DataFrame(ad, layer="doesn't exist")
    df2 = DataFrame(ad, columns=:obs)
    @test names(df2) == ["var"; ad.obs_names]
    @test df2.var == ad.var_names
end
