using StaticArrays: StaticArray, SMatrix, SVector
using LinearAlgebra: I

extract_diffresult(xs::AbstractArray{<:Number}) = xs
# need to optimize
extract_diffresult(xs) = hcat(xs...)'
function extract_diffresult(xs::StaticArray{<:Any,<:StaticArray})
    tup = reduce((x,y)->tuple(x..., y...), map(x->x.data, xs.data))
    SMatrix{length(xs[1]), length(xs)}(tup)'
end
extract_diffresult(xs::AbstractMatrix{<:Number}) = xs
extract_diffresult(xs::AbstractVector{<:Number}) = xs'

allpartials(xs) = map(partials, xs)

function seed(v::SVector{N}) where N
    SMatrix{N,N,eltype(v)}(I)
end

function _seed(v, ij)
    i, j = Tuple(ij)
    vi = v[i]
    return (i==j) ? one(vi) : zero(vi)
end

function seed(v)
    vv = vec(v)
    ax = axes(vv, 1)
    return _seed.(Ref(vv), CartesianIndices((ax, ax)))
end

function D(f)
    # grad
    function deriv(arg::AbstractArray)
        # always chunk
        arg_partial = seed(arg)
        darr = dualrun(()->DualArray(arg, arg_partial))
        res = dualrun(()->f(darr))
        diffres = extract_diffresult(allpartials(res))
        return diffres
    end
    # scalar
    function deriv(arg)
        dualrun() do
            dualized = map(x->Dual(x, one(x)), arg)
            res = f(dualized)
            return map(partials, res)
        end
    end
    return deriv
end

#=
# scalar case: f: R -> something
D(sin)(1.0)
D(x->[x, x^2])(3)

# gradient case: f: R^n -> R
D(sum)([1,2,3])

# Jacobian case: f: R^n -> R^m
=#
