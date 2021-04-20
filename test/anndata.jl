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
    for i in 2:n
        test_row_slice(ad, 1:i, n, d, x)
    end
    for i in 1:(n - 1)
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
