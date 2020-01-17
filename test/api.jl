using Test
using ForwardDiff2: D, ForwardDiff2
using StaticArrays
using OrdinaryDiffEq, ForwardDiff
using LinearAlgebra

@testset "Derivative, Gradient, Jacobian, and Hessian" begin
    @test D(sin)(1) * true === cos(1.0)
    @test D(float)(1) * 1 === 1
    @test D(AbstractFloat)(1) * 1 === 1
    @test D(x->[x, x^2])(3) * 1 == [1, 6]
    # Gradient
    @test D(sum)([1,2,3]) * I == ones(3)'
    # Jacobian
    @test D(x->@SVector([x[1]^x[2], x[3]^3, x[3]*x[2]*x[1]]))(@SVector[1,2,3.]) * I === @SMatrix [2.0 0 0; 0 0 27; 6 3 2]
    dcumsum = D(cumsum)
    j2 = dcumsum(@SVector([1,2,3])) * I
    tj2 = @MMatrix [1 0 0; 1 1 0; 1 1 1]
    @test j2 == tj2
    @test typeof(j2) == typeof(tj2)
    @test dcumsum([1,2,3]) * I == [1 0 0; 1 1 0; 1 1 1]
    @test D(x->@SVector([x[1], x[2]]))(@SVector([1,2,3])) * I === @SMatrix [1 0 0; 0 1 0]
    # Hessian
    dh = D(x -> D(x->x[1]^x[2] + x[3]^3 + x[3]*x[2]*x[1])(x) * I)
    @test dh(@SVector[1,2,3]) * I === @SMatrix [2 4 2; 4 0 1; 2 1 18.]
    @test dh([1,2,3]) * I == [2 4 2; 4 0 1; 2 1 18.]
end

@testset "Inference" begin
    simple_D = @inferred D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x)
    @inferred simple_D(1) * 1
    @inferred simple_D(1) * true

    complicated_D = D(x->exp(x) + x^x + cos(x) + tan(x) + 2^x + log(cos(x)) + sec(pi*x) - angle(x) + one(x) / log1p(sin(x)))
    # broken due to `Core._apply`
    @test_broken @inferred complicated_D(1) * 1
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
        fd2 = D(x->odesolver(x, ForwardDiff2.value, tf))(u0) * I
        fd = ForwardDiff.jacobian(x->odesolver(x, ForwardDiff.value, tf), u0)
        @test fd2 ≈ fd rtol=1e-12 # Wow, chaos!
    end
end

@testset "No perturbation confusion" begin
    @test D(x -> x * D(y -> x + y)(1))(1) * 1 === 1
end
