@testset "create mudata" begin
  n = 100
  d1 = 10
  d2 = 20
  x = rand(Float64, (n, d1))
  ad = AnnData(X=x)
  y = rand(Float64, (n, d2))
  ad2 = AnnData(X=y)
  md = MuData(mod=Dict("ad1" => ad, "ad2" => ad2))
  @test size(md)[1] == n
  @test size(md)[2] == max(d1, d2)
  
  modalities = collect(keys(md.mod))
  @test modalities == ["ad1", "ad2"]
end
