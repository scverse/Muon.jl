n = 100
d1 = 10
d2 = 20

warn_msg = "Cannot join columns with the same name because var_names are intersecting."

x = rand(Float64, (n, d1))
ad = AnnData(X=x)

y = rand(Float64, (n, d2))
ad2 = AnnData(X=y)

@test_logs (:warn, warn_msg) md = MuData(mod=Dict("ad1" => ad, "ad2" => ad2))

@testset "create mudata" begin
  @test size(md, 1) == n
  @test size(md, 2) == d1 + d2

  modalities = sort(collect(keys(md.mod)))
  @test modalities == ["ad1", "ad2"]
end

@testset "slicing mudata" begin
  d = d1 + d2
  for i = 2:n
    @test (@test_logs (:warn, warn_msg) size(md[1:i,:])) == (i, d)
  end
  for i = 1:(n-1)
    @test (@test_logs (:warn, warn_msg) size(md[i:i+1,:])) == (2, d)
  end
  @test (@test_logs (:warn, warn_msg) size(md[["1", "2", "3"],:])) == (3, d)
  @test (@test_logs (:warn, warn_msg) size(md[fill(true, n),:])) == (n, d)
end
