using Test
using ForwardDiff2: Dual, partials, dualrun
function D(f, x)
    dualrun() do
        xx = Dual(x, one(x))
        partials(f(xx), 1)
    end
end

@test D(x -> x * D(y -> x + y, 1), 1) === 1
