# ForwardDiff2

[![Build Status](https://travis-ci.org/YingboMa/ForwardDiff2.jl.svg?branch=master)](https://travis-ci.org/YingboMa/ForwardDiff2.jl)
[![codecov](https://codecov.io/gh/YingboMa/ForwardDiff2.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/ForwardDiff2.jl)

`ForwardDiff2` = `ForwardDiff.jl` + `ChainRules.jl` + Struct of arrays

### Warning!!!: This package is still work-in-progress

User API:
`D(f)(x)` returns a lazy representation of the derivative. 

`D(f)(x) * v` computes `df(x)/dx ⋅ v`, taking advantage of the laziness in `D(f)(x)`.


`DI(f)(x)` is a convenience function to materialize the derivative, gradient or
Jacobian of `f` at `x`.

```julia
julia> using Random; Random.seed!(123);

julia> using ForwardDiff2: D, DI; using LinearAlgebra

julia> D(sin)(10) * 11 === cos(10) * 11
true

julia> x = rand(3);

julia> v = rand(3);

julia> D(prod)(x) # lazy gradient
D(prod)([0.768448, 0.940515, 0.673959])

julia> D(prod)(x) * I # materialize the gradient by multiplying by I
1×3 Adjoint{Float64,SArray{Tuple{3},Float64,1,3}} with indices SOneTo(1)×SOneTo(3):
 0.633868  0.517902  0.722737

julia> D(cumsum)(x) * I # Jacobian
3×3 Adjoint{Float64,Array{Float64,2}}:
 1.0  0.0  0.0
 1.0  1.0  0.0
 1.0  1.0  1.0

julia> D(cumsum)(x) * v # Jacobian-vector product
3-element Array{Float64,1}:
 0.3954531123351086
 0.7086970681426272
 1.3712518846162807

julia> DI(DI(prod))(x) # Hessian
3×3 SArray{Tuple{3,3},Float64,2,9} with indices SOneTo(3)×SOneTo(3):
 0.0       0.673959  0.940515
 0.673959  0.0       0.768448
 0.940515  0.768448  0.0

julia> DI(DI(prod))(x) * v # Hessian-vector product
3-element SArray{Tuple{3},Float64,1,3} with indices SOneTo(3):
 0.8342562312269415
 0.775657771761718
 0.6126411738403423
```

Note that `ForwardDiff2.jl` also works with `ModelingToolkit.jl`:
```julia
julia> using ModelingToolkit

julia> @variables x[1:3] v[1:3]
(Operation[x₁, x₂, x₃], Operation[v₁, v₂, v₃])

julia> D(sin)(x[1]) * 11
cos(x₁) * 11

julia> D(prod)(x) * I # gradient
1×3 Adjoint{Operation,StaticArrays.SArray{Tuple{3},Operation,1,3}} with indices SOneTo(1)×SOneTo(3):
 conj((identity(1) * x₂ + x₁ * identity(0)) * x₃ + (x₁ * x₂) * identity(0))  …  conj((identity(0) * x₂ + x₁ * identity(0)) * x₃ + (x₁ * x₂) * 1)

julia> D(cumsum)(x) * I # Jacobian
3×3 Adjoint{Operation,Array{Expression,2}}:
                               conj(1)                      conj(identity(0))                      conj(identity(0))
                 conj(identity(0) + 1)                  conj(1 + identity(0))        conj(identity(0) + identity(0))
 conj((identity(0) + identity(0)) + 1)  conj((identity(0) + 1) + identity(0))  conj((1 + identity(0)) + identity(0))

julia> D(cumsum)(x) * [1, 2, 3] # Jacobian-vector product
3-element Array{Int64,1}:
 1
 3
 6

julia> D(cumsum)(x) * v # Jacobian-vector product
3-element Array{Operation,1}:
             v₁
        v₂ + v₁
 (v₃ + v₂) + v₁

julia> DI(DI(prod))(x) # Hessian
3×3 StaticArrays.SArray{Tuple{3,3},Operation,2,9} with indices SOneTo(3)×SOneTo(3):
           conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(1) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(1) * identity(identity(0)) + x₁ * 0) + (identity(1) * identity(0) + x₂ * 0))))  …            conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(1) * x₂ + x₁ * identity(0)) * identity(1) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(1) * identity(0) + x₂ * 0))))
           conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * 1) * identity(0) + x₃ * ((identity(1) * identity(1) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0))))               conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * 1) * identity(1) + x₃ * ((identity(0) * identity(1) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0))))
 conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(1) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(1) * identity(identity(0)) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0))))     conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(1) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * identity(0)) * identity(1) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0))))

julia> DI(DI(prod))(x) * v # Hessian-vector product
3-element StaticArrays.SArray{Tuple{3},Operation,1,3} with indices SOneTo(3):
                               (conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(1) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(1) * identity(identity(0)) + x₁ * 0) + (identity(1) * identity(0) + x₂ * 0)))) * v₁ + conj(((identity(0) * x₂ + x₁ * 1) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(1) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(1) * identity(1) + x₂ * 0)))) * v₂) + conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(1) * x₂ + x₁ * identity(0)) * identity(1) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(1) * identity(0) + x₂ * 0)))) * v₃
                               (conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * 1) * identity(0) + x₃ * ((identity(1) * identity(1) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0)))) * v₁ + conj(((identity(0) * x₂ + x₁ * 1) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * 1) * identity(0) + x₃ * ((identity(0) * identity(1) + x₁ * 0) + (identity(identity(0)) * identity(1) + x₂ * 0)))) * v₂) + conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(identity(0)) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * 1) * identity(1) + x₃ * ((identity(0) * identity(1) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0)))) * v₃
 (conj(((identity(1) * x₂ + x₁ * identity(0)) * identity(1) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(1) * identity(identity(0)) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0)))) * v₁ + conj(((identity(0) * x₂ + x₁ * 1) * identity(1) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * identity(0)) * identity(0) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(identity(0)) * identity(1) + x₂ * 0)))) * v₂) + conj(((identity(0) * x₂ + x₁ * identity(0)) * identity(1) + (x₁ * x₂) * 0) + ((identity(identity(0)) * x₂ + x₁ * identity(0)) * identity(1) + x₃ * ((identity(0) * identity(identity(0)) + x₁ * 0) + (identity(identity(0)) * identity(0) + x₂ * 0)))) * v₃
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
