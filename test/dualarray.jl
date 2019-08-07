using Test
using ForwardDiff2: DualArray, tagtype, npartials, value, partials
using ForwardDiff: value, partials

@testset "DualArray" begin
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
    @test sin.(x) == dx
    @test sin.(_x) == value.(dx)
    @test cos.(_x) == partials.(dx, 1)
end

@testset "indexing" begin
    a = reshape(1:60, (5,4,3))
    da = DualArray(a)

    @test da[9]  == Dual(9, 29, 49)
    @test da[10] == Dual(10, 30, 50)

    @test da[10] = Dual(2,2,2)
    @test da[10] === Dual(2,2,2)
end
