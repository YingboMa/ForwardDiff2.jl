# ForwardDiff2

[![Build Status](https://travis-ci.org/YingboMa/ForwardDiff2.jl.svg?branch=master)](https://travis-ci.org/YingboMa/ForwardDiff2.jl)
[![codecov](https://codecov.io/gh/YingboMa/ForwardDiff2.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/ForwardDiff2.jl)

`ForwardDiff2` = `ForwardDiff.jl` + `ChainRules.jl` + Struct of arrays

### Warning!!!: This package is still work-in-progress

User API:

```julia
julia> using ForwardDiff2: D; using LinearAlgebra

julia> D(sin, 10) * 11 === cos(10) * 11
true

julia> v = rand(2)
2-element Array{Float64,1}:
 0.03215572910208908
 0.0939539318769742

julia> D(prod, v) * I # gradient
1×2 Adjoint{Float64,SArray{Tuple{2},Float64,1,2}} with indices SOneTo(1)×SOneTo(2):
 0.0939539  0.0321557

julia> D(cumsum, v) * I # Jacobian
2×2 Adjoint{Float64,Array{Float64,2}}:
 1.0  0.0
 1.0  1.0

julia> D(x->D(prod, x) * I, v) * I # Hessian
2×2 SArray{Tuple{2,2},Float64,2,4} with indices SOneTo(2)×SOneTo(2):
 0.0  1.0
 1.0  0.0
```

Note that `ForwardDiff2.jl` also works with `ModelingToolkit.jl`:
```julia
julia> using ModelingToolkit

julia> @variables v[1:2]
(Operation[v₁, v₂],)

julia> D(sin, v[1]) * 11
cos(v₁) * 11

julia> D(prod, v) * I # gradient
1×2 Adjoint{Operation,SArray{Tuple{2},Operation,1,2}} with indices SOneTo(1)×SOneTo(2):
 conj(identity(1) * v₂ + v₁ * identity(0))  conj(identity(0) * v₂ + v₁ * 1)

julia> D(cumsum, v) * I # Jacobian
2×2 Adjoint{Operation,Array{Expression,2}}:
               conj(1)      conj(identity(0))
 conj(identity(0) + 1)  conj(1 + identity(0))

julia> D(x->D(prod, x) * I, v) * I # Hessian
2×2 SArray{Tuple{2,2},Operation,2,4} with indices SOneTo(2)×SOneTo(2):
 conj((identity(1) * identity(identity(0)) + v₁ * 0) + (identity(1) * identity(0) + v₂ * 0))  …  conj((identity(0) * identity(identity(0)) + v₁ * 0) + (identity(1) * identity(1) + v₂ * 0))
 conj((identity(1) * identity(1) + v₁ * 0) + (identity(identity(0)) * identity(0) + v₂ * 0))     conj((identity(0) * identity(1) + v₁ * 0) + (identity(identity(0)) * identity(1) + v₂ * 0))
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
