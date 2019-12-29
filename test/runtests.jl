using SafeTestsets
@time @safetestset "Nested Differentiation Tests" begin include("nested.jl") end
@time @safetestset "Dual Tests" begin include("dualtest.jl") end
@time @safetestset "DualArray Tests" begin include("dualarray.jl") end
@time @safetestset "Jacobian Tests" begin include("jacobian.jl") end
