# ForwardDiff2

[![Build Status](https://travis-ci.org/YingboMa/ForwardDiff2.jl.svg?branch=master)](https://travis-ci.org/YingboMa/ForwardDiff2.jl)
[![codecov](https://codecov.io/gh/YingboMa/ForwardDiff2.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/ForwardDiff2.jl)

`ForwardDiff2` = `ForwardDiff.jl` + `ChainRules.jl` + Struct of arrays

### Warning!!!: This package is still work-in-progress

User API:
```julia
julia> using ForwardDiff2: D

julia> v = rand(2)
2-element Array{Float64,1}:
 0.22260830987887537
 0.6397089507287486

julia> D(prod)(v) # gradient
1×2 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.639709  0.222608

julia> D(cumsum)(v) # Jacobian
2×2 Array{Float64,2}:
 1.0  0.0
 1.0  1.0

julia> D(D(prod))(v) # Hessian
2×2 LinearAlgebra.Adjoint{Float64,Array{Float64,2}}:
 0.0  1.0
 1.0  0.0
```

Note that `ForwardDiff2.jl` also works with `ModelingToolkit.jl`:
```julia
julia> using ModelingToolkit

julia> @variables v[1:2]
(Operation[v₁, v₂],)

julia> D(prod)(v) # gradient
1×2 LinearAlgebra.Adjoint{Operation,Array{Operation,1}}:
 conj(1v₂ + v₁ * identity(0))  conj(identity(0) * v₂ + v₁ * 1)

julia> D(cumsum)(v) # Jacobian
2×2 Array{Expression,2}:
     Constant(1)      identity(0)
 identity(0) + 1  1 + identity(0)

julia> D(D(prod))(v) # Hessian
2×2 LinearAlgebra.Adjoint{Operation,Array{Operation,2}}:
 conj((1 * identity(0) + v₁ * 0) + (1 * identity(0) + v₂ * 0))  conj((identity(0) * identity(0) + v₁ * 0) + (1 * 1 + v₂ * 0))
 conj((1 * 1 + v₁ * 0) + (identity(0) * identity(0) + v₂ * 0))  conj((identity(0) * 1 + v₁ * 0) + (identity(0) * 1 + v₂ * 0))
```

Planned features:

- works both on GPU and CPU
- [Dual cache](http://docs.juliadiffeq.org/latest/basics/faq.html#I-get-Dual-number-errors-when-I-solve-my-ODE-with-Rosenbrock-or-SDIRK-methods...?-1)
- user-extensible scalar and tensor derivative definitions
- in-place function
- sparsity exploitation (color vector support)
- complex differentiation (use `ChainRules.jl`)
- plays nicely with Zygote

The `ForwardDiff2` source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
