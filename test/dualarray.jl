using Test
using ForwardDiff2: Dual, DualArray, tagtype, npartials, value, partials, dualrun
using StaticArrays: SVector

@testset "DualArray" begin
    dualrun() do
        a = rand(2)
        d1 = DualArray(a)
        @test tagtype(d1) === Nothing
        d2 = DualArray{2,2}(a)
        @test tagtype(d2) === 2

        _x = rand(3)
        x = DualArray(hcat(_x, ones(3)))
        @test npartials(x) === 1
        dx = similar(x)
        dx .= sin.(x)
        @test_broken sin.(x) == dx
        @test sin.(_x) == value.(dx)
        @test cos.(_x) == first.(partials.(dx))
    end
end

@testset "indexing" begin
    dualrun() do
        a = reshape(collect(1:60), (5,4,3))
        da = DualArray(a)

        @test da[9]  == Dual(9, SVector(29, 49))
        @test da[10] == Dual(10, SVector(30, 50))

        da[10] = Dual(2, SVector(2, 2))
        @test da[10] === Dual(2, SVector(2, 2))
    end
end
