# ForwardDiff2

[![Build Status](https://travis-ci.org/YingboMa/ForwardDiff2.jl.svg?branch=master)](https://travis-ci.org/YingboMa/ForwardDiff2.jl)
[![codecov](https://codecov.io/gh/YingboMa/ForwardDiff2.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/ForwardDiff2.jl)

`ForwardDiff2` = `ForwardDiff.jl` + `ChainRules.jl` + Struct of arrays + `DualCache`

Planned features:

- works both on GPU and CPU
- scalar forward mode AD
- vectorized forward mode AD
- [Dual cache](http://docs.juliadiffeq.org/latest/basics/faq.html#I-get-Dual-number-errors-when-I-solve-my-ODE-with-Rosenbrock-or-SDIRK-methods...?-1)
- nested differentiation
- hyper duals (?)
- user-extensible scalar and tensor derivative definitions
- in-place function
- sparsity exploitation (color vector support)
- complex differentiation (use `ChainRules.jl`)
- plays nicely with Zygote

The `ForwardDiff2` source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
