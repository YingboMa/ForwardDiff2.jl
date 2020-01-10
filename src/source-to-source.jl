f(x) = sin(cos(x))

f(x)
y, d = applyfrule(f, x, d)
y, d = applyfrule(f, x, d)
y, d = applyfrule(f, x, d)
y, d = applyfrule(f, x, d)
function f'(x, Δ)
    x1 = cos(x)
    dx1 = -Δ*sin(x)
    x2 = sin(x1)
    dx2 = dx1*cos(x2)
    return x2, dx2
end

f'(Δx) = (-cos(cos(x)) * sin(x)) * Δx
