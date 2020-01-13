using Test
using ForwardDiff2: D
using StaticArrays

@testset begin
    @test D(sin)(1.0) == cos(1.0)
    # Gradient
    @test D(x->[x, x^2])(3) == [1, 6]
    @test D(sum)([1,2,3]) == ones(3)'
    # Jacobian
    @test D(x->@SVector([x[1]^x[2], x[3]^3, x[3]*x[2]*x[1]]))(@SVector[1,2,3.]) === @SMatrix [2.0 0 0; 0 0 27; 6 3 2]
    @test D(cumsum)(@SVector([1,2,3])) == @SMatrix [1 0 0; 1 1 0; 1 1 1]
    @test D(cumsum)([1,2,3]) == [1 0 0; 1 1 0; 1 1 1]
    @test D(x->@SVector([x[1], x[2]]))(@SVector([1,2,3])) === @SMatrix [1 0 0; 0 1 0]
    # Hessian
    @test D(D(x->x[1]^x[2] + x[3]^3 + x[3]*x[2]*x[1]))(@SVector[1,2,3]) === @SMatrix [2 4 2; 4 0 1; 2 1 18.]
    @test D(D(x->x[1]^x[2] + x[3]^3 + x[3]*x[2]*x[1]))([1,2,3]) == [2 4 2; 4 0 1; 2 1 18.]
    # inference
    @inferred D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x)(1)
    # broken due to `Core._apply`
    @test_broken @inferred D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x + log(cos(x)) + sec(pi*x) - angle(x) + one(x) / log1p(sin(x)))(1)
end
