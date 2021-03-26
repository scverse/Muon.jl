@testset "create adata" begin
  n = 100
  x = rand(Float64, (n, 10))
  ad = AnnData(X=x)
  @test size(ad)[1] == n
end
