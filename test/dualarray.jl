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

    @test all(iszero, partials(similar(x), 1))
end
