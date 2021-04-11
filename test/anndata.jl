n = 100
d = 10
x = rand(Float64, (n, d))
ad = AnnData(X=x)

@testset "create anndata" begin
  @test size(ad, 1) == n
end


@testset "slicing anndata" begin
  for i = 2:n
    @test size(ad[collect(1:i),:]) == (i, d)
  end
  for i = 1:(n-1)
    @test size(ad[collect(i:i+1),:]) == (2, d)
  end
  @test size(ad[["1", "2", "3"],:]) == (3, d)
  @test size(ad[fill(true, n),:]) == (n, d)
end
