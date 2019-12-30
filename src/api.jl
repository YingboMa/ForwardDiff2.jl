using ForwardDiff2: dualrun, partials, Dual

tilt(xs::AbstractArray{<:Number}) = xs'
tilt(xs) = hcat(xs...)

function D(f)
    # grad
    function deriv(arg::AbstractArray)
        # no chunking
        grads = map(arg, eachindex(arg)) do _, i
            dualized = map(arg, eachindex(arg)) do x, j
                dualrun(()->Dual(x, i == j ? one(x) : zero(x)))
            end
            res = dualrun(()->f(dualized))
            return map(partials, res)
        end
        # TODO: fix performance of this:
        tilt(grads)
    end
    # scalar
    function deriv(arg)
        dualrun() do
            dualized = map(x->Dual(x, one(x)), arg)
            res = f(dualized)
            return map(partials, res)
        end
    end
    deriv
end

#=
# scalar case: f: R -> something
D(sin)(1.0)
D(x->[x, x^2])(3)

# gradient case: f: R^n -> R
D(sum)([1,2,3])

# Jacobian case: f: R^n -> R^m
=#
