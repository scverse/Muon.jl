using HDF5
using SparseArrays
using Random

file = h5open(tempname(), "w")
Random.seed!(42)
spmatrix = sprand(20, 20, 0.3)
spmatrix_t = spmatrix'
Muon.write_impl(file, "mat", spmatrix)
Muon.write_impl(file, "mat_t", spmatrix_t)

@testset "sparse matrix read" begin
    @test Muon.read_matrix(file["mat"]) == spmatrix
    @test Muon.read_matrix(file["mat_t"]) == spmatrix_t
end

@testset "sparse dataset" begin
    spdset = Muon.SparseDataset(file["mat"])
    spdset_t = Muon.SparseDataset(file["mat_t"])

    @testset "full read" begin
        @test read(spdset) == spmatrix
        @test read(spdset_t) == spmatrix_t
    end

    # double nesting to have more compact output in case someting fails
    @testset "single element access" begin
        @testset "index: $i,$j" for i in 1:size(spmatrix, 1), j in 1:size(spmatrix, 2)
            @test spdset[i, j] == spmatrix[i, j]
            @test spdset_t[i, j] == spmatrix_t[i, j]
        end
    end
    @testset "row access" begin
        @testset "row: $i" for i in 1:size(spmatrix, 1)
            @test spdset[i, :] == spmatrix[i, :]
            @test spdset_t[i, :] == spmatrix_t[i, :]
        end
    end
    @testset "column access" begin
        @testset "column: $j" for j in 1:size(spmatrix, 2)
            @test spdset[:, j] == spmatrix[:, j]
            @test spdset_t[:, j] == spmatrix_t[:, j]
        end
    end

    @testset "range access" begin
        @testset "range: $rowstart:$(rowstart + 4), $colstart:$(colstart + 4)" for (
            rowstart,
            colstart,
        ) in zip(
            1:(size(spmatrix, 1) - 4),
            1:(size(spmatrix, 2) - 4),
        )
            @test spdset[rowstart:(rowstart + 4), colstart:(colstart + 4)] ==
                  spmatrix[rowstart:(rowstart + 4), colstart:(colstart + 4)]
            @test spdset_t[rowstart:(rowstart + 4), colstart:(colstart + 4)] ==
                  spmatrix_t[rowstart:(rowstart + 4), colstart:(colstart + 4)]
        end
    end

    @testset "row range access" begin
        @testset "row: $i, column range: $colstart:$(colstart + 4)" for i in 1:size(spmatrix, 1),
            colstart in 1:(size(spmatrix, 2) - 4)

            @test spdset[i, colstart:(colstart + 4)] == spmatrix[i, colstart:(colstart + 4)]
            @test spdset_t[i, colstart:(colstart + 4)] == spmatrix_t[i, colstart:(colstart + 4)]
        end
    end
    @testset "column range access" begin
        @testset "column: $j, row range: $rowstart:$(rowstart + 4)" for j in 1:size(spmatrix, 2),
            rowstart in 1:(size(spmatrix, 1) - 4)

            @test spdset[rowstart:(rowstart + 4), j] == spmatrix[rowstart:(rowstart + 4), j]
            @test spdset_t[rowstart:(rowstart + 4), j] == spmatrix_t[rowstart:(rowstart + 4), j]
        end
    end

    @testset "single element modification" begin
        I, J, V = findnz(spmatrix)
        ci = 1
        for j in 1:size(spmatrix, 2)
            while j > J[ci] && ci < length(I)
                ci += 1
            end
            for i in 1:size(spmatrix, 1)
                while i > I[ci] && j >= J[ci] && ci < length(I)
                    ci += 1
                end

                @testset "index: $i, $j" begin
                    if i == I[ci] && j == J[ci]
                        spdset[i, j] = spmatrix[i, j] = spdset_t[j, i] = spmatrix_t[j, i] = π
                        @test spdset == spmatrix
                        @test spdset_t == spmatrix_t
                    else
                        @test_throws KeyError spdset[i, j] = π
                        @test_throws KeyError spdset_t[j, i] = π
                    end
                end
            end
        end
    end

    @testset "range modification" begin
        @testset "range: $rowstart:$(rowstart + 4), $colstart:$(colstart + 4)" for (
            rowstart,
            colstart,
        ) in zip(
            1:(size(spmatrix, 1) - 4),
            1:(size(spmatrix, 2) - 4),
        )
            rI, rJ = rowstart:(rowstart + 4), colstart:(colstart + 4)
            submat = spmatrix[rI, rJ]
            nonzeros(submat) .= rand(eltype(submat), nnz(submat))

            spdset[rI, rJ] = spmatrix[rI, rJ] = submat
            spdset_t[rJ, rI] = spmatrix_t[rJ, rI] = submat'

            @test spdset == spmatrix
            @test spdset_t == spmatrix_t

            @test_throws KeyError spdset[rI, rJ] = rand(eltype(spdset), (length(rI), length(rJ)))
            @test_throws KeyError spdset_t[rJ, rI] =
                rand(eltype(spdset_t), (length(rJ), length(rI)))
        end
    end
end

close(file)
