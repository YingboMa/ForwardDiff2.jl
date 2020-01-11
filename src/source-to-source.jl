# Template
g(x) = tan(x) * 2x + 10x
f(x) = sin(cos(x)) * g(x)

function dg(Δ, x)
    dself = Zero()
    x1, Δ1 = frule(dself, tan, x, Δ)
    x2, Δ2 = frule(dself, *, 2, x, Zero(), Δ)
    x3, Δ3 = frule(dself, *, x1, x2, Δ1, Δ2)
    x4, Δ4 = frule(dself, *, 10, x, Zero(), Δ)
    x5, Δ5 = frule(dself, +, x3, x4, Δ3, Δ4)
    return x5, Δ5
end

function df(Δ, x)
    dself = Zero()
    x1, Δ1 = frule(dself, cos, x, Δ)
    x2, Δ2 = frule(dself, sin, x1, Δ1)
    x3, Δ3 = frule(dself, tan, x2, Δ2)
    x4, Δ4 = dg(x, x3, Δ3)
    x5, Δ5 = frule(dself, *, x3, x4, Δ3, Δ4)
    return x5, Δ5
end
