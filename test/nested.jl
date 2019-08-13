using Test
using ForwardDiff2: Dual, partials, dualrun
function D(f, x)
    xx = Dual(x, one(x))
    partials(dualrun(f, xx), 1)
end

@test D(x -> x * D(y -> x + y, 1), 1) === 1