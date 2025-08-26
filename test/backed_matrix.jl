using HDF5
using Zarr
using SparseArrays
using Random

h5file = h5open(tempname(), "w")
zarrfile = zgroup(tempname())
Random.seed!(42)
spmatrix = sprand(20, 20, 0.3)
matrix = rand(20, 20)

@testset "$backend" for (file, backend) ∈ ((h5file, "HDF5"), (zarrfile, "Zarr"))
    for (mat, T, name, issparse) ∈ (
        (spmatrix, Muon.SparseDataset, "sparse", true),
        (matrix, Muon.TransposedDataset, "transposed", false),
    )
        mat_t = mat'

        Muon.write_attr(file, "mat", mat)
        Muon.write_attr(file, "mat_t", mat_t)

        @testset "$name matrix read" begin
            @test Muon.read_matrix(file["mat"]) == mat
            @test Muon.read_matrix(file["mat_t"]) == mat_t
        end

        @testset "$name dataset" begin
            dset = T(file["mat"])
            dset_t = T(file["mat_t"])

            @testset "full read" begin
                @test read(dset) == mat
                @test read(dset_t) == mat_t
            end

            # double nesting to have more compact output in case someting fails
            @testset "single element access" begin
                @testset "index: $i,$j" for i ∈ 1:size(mat, 1), j ∈ 1:size(mat, 2)
                    @test dset[i, j] == mat[i, j]
                    @test dset_t[i, j] == mat_t[i, j]
                end
            end
            @testset "row access" begin
                @testset "row: $i" for i ∈ 1:size(mat, 1)
                    @test dset[i, :] == mat[i, :]
                    @test dset_t[i, :] == mat_t[i, :]
                end
            end
            @testset "column access" begin
                @testset "column: $j" for j ∈ 1:size(mat, 2)
                    @test dset[:, j] == mat[:, j]
                    @test dset_t[:, j] == mat_t[:, j]
                end
            end

            @testset "range access" begin
                @testset "range: $rowstart:$(rowstart + 4), $colstart:$(colstart + 4)" for (
                    rowstart,
                    colstart,
                ) ∈ zip(
                    1:(size(mat, 1) - 4),
                    1:(size(mat, 2) - 4),
                )
                    @test dset[rowstart:(rowstart + 4), colstart:(colstart + 4)] ==
                          mat[rowstart:(rowstart + 4), colstart:(colstart + 4)]
                    @test dset_t[rowstart:(rowstart + 4), colstart:(colstart + 4)] ==
                          mat_t[rowstart:(rowstart + 4), colstart:(colstart + 4)]
                end
            end

            @testset "row range access" begin
                @testset "row: $i, column range: $colstart:$(colstart + 4)" for i ∈ 1:size(mat, 1),
                    colstart ∈ 1:(size(mat, 2) - 4)

                    @test dset[i, colstart:(colstart + 4)] == mat[i, colstart:(colstart + 4)]
                    @test dset_t[i, colstart:(colstart + 4)] == mat_t[i, colstart:(colstart + 4)]
                end
            end
            @testset "column range access" begin
                @testset "column: $j, row range: $rowstart:$(rowstart + 4)" for j ∈ 1:size(mat, 2),
                    rowstart ∈ 1:(size(mat, 1) - 4)

                    @test dset[rowstart:(rowstart + 4), j] == mat[rowstart:(rowstart + 4), j]
                    @test dset_t[rowstart:(rowstart + 4), j] == mat_t[rowstart:(rowstart + 4), j]
                end
            end

            @testset "single element modification" begin
                if issparse
                    I, J, V = findnz(mat)
                    ci = 1
                    for j ∈ 1:size(mat, 2)
                        while j > J[ci] && ci < length(I)
                            ci += 1
                        end
                        for i ∈ 1:size(mat, 1)
                            while i > I[ci] && j >= J[ci] && ci < length(I)
                                ci += 1
                            end

                            @testset "index: $i, $j" begin
                                if i == I[ci] && j == J[ci]
                                    dset[i, j] = mat[i, j] = dset_t[j, i] = mat_t[j, i] = π
                                    @test dset == mat
                                    @test dset_t == mat_t
                                else
                                    @test_throws KeyError dset[i, j] = π
                                    @test_throws KeyError dset_t[j, i] = π
                                end
                            end
                        end
                    end
                else
                    @testset "index: $i,$j" for i ∈ 1:size(mat, 1), j ∈ 1:size(mat, 2)
                        dset[i, j] = mat[i, j] = dset_t[j, i] = mat_t[j, i] = π
                        @test dset == mat
                        @test dset_t == mat_t
                    end
                end
            end

            @testset "range modification" begin
                @testset "range: $rowstart:$(rowstart + 4), $colstart:$(colstart + 4)" for (
                    rowstart,
                    colstart,
                ) ∈ zip(
                    1:(size(mat, 1) - 4),
                    1:(size(mat, 2) - 4),
                )
                    rI, rJ = rowstart:(rowstart + 4), colstart:(colstart + 4)
                    submat = mat[rI, rJ]
                    issparse ? nonzeros(submat) .= rand(eltype(submat), nnz(submat)) :
                    submat .= rand(eltype(submat), size(submat)...)

                    dset[rI, rJ] = mat[rI, rJ] = submat
                    dset_t[rJ, rI] = mat_t[rJ, rI] = submat'

                    @test dset == mat
                    @test dset_t == mat_t

                    if issparse
                        @test_throws KeyError dset[rI, rJ] =
                            rand(eltype(dset), (length(rI), length(rJ)))
                        @test_throws KeyError dset_t[rJ, rI] =
                            rand(eltype(dset_t), (length(rJ), length(rI)))
                    end
                end
            end
        end
    end
end
close(h5file)
