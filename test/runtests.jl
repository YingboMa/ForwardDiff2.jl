using SafeTestsets
@time @safetestset "API Tests" begin include("api.jl") end
@time @safetestset "Nested Differentiation Tests" begin include("nested.jl") end
@time @safetestset "Dual Tests" begin include("dualtest.jl") end
