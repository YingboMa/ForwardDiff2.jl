using Test
using ForwardDiff2: D, ForwardDiff2
using StaticArrays
using OrdinaryDiffEq, ForwardDiff
using LinearAlgebra: norm

@testset "Derivative, Gradient, Jacobian, and Hessian" begin
    @test D(sin)(1.0) === cos(1.0)
    @test D(float)(1) === 1
    @test D(AbstractFloat)(1) === 1
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
end

@testset "Inference" begin
    @inferred D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x)(1)
    # broken due to `Core._apply`
    @test_broken @inferred D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x + log(cos(x)) + sec(pi*x) - angle(x) + one(x) / log1p(sin(x)))(1)
end

@testset "Differentiate an ODE solver" begin
    # TODO: drop derivative API
    OrdinaryDiffEq.constvalue(t::Type{<:ForwardDiff2.Dual}) = ForwardDiff2.value(t)
    function lorenz(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end
    function odesolver(u0, value, tf)
        tspan = (0.0, tf)
        prob = ODEProblem(lorenz, u0, tspan)
        sol = solve(prob, Tsit5(), internalnorm=(u, t) -> value(norm(u)))
        return sol[end]
    end

    u0 = [1.0, 0.0, 0.0]
    for tf in [1, 10, 100.0]
        fd2 = D(x->odesolver(x, ForwardDiff2.value, tf))(u0)
        fd = ForwardDiff.jacobian(x->odesolver(x, ForwardDiff.value, tf), u0)
        @test fd2 ≈ fd rtol=1e-12 # Wow, chaos!
    end
end
