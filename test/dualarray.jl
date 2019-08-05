using Test
using ForwardDiff2: DualArray, tagname, ncols
using ForwardDiff: value, partials

@testset "DualArray" begin
    a = rand(2)
    d1 = DualArray(a)
    @test tagname(d1) === Nothing
    d2 = DualArray{2,2}(a)
    @test tagname(d2) === 2
    @test ncols(d2) === 2

    _x = rand(3)
    x = DualArray(hcat(_x, ones(3)))
    dx = similar(x)
    for i in eachindex(x)
        dx[i] = sin(x[i])
    end
    @test sin.(_x) == dx.data[:, 1]
    @test cos.(_x) == dx.data[:, 2]
end
