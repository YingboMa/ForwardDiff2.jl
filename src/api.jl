using StaticArrays: StaticArray, SMatrix, SVector
using LinearAlgebra: Diagonal, I

tilt(xs::AbstractArray{<:Number}) = xs'
tilt(xs) = hcat(xs...)
tilt(xs::StaticArray{<:Number}) = xs'
function tilt(xs::StaticArray)
    # Jacobian
    tup = reduce((x,y)->tuple(x..., y...), map(x->x.data, xs.data))
    SMatrix{length(xs), length(xs[1])}(tup)'
end

allpartials(xs) = map(partials, xs)

function seed(v, i)
    SVector(map((e, j) -> j == i ? one(e) : zero(e), v, eachindex(v))...)
end

function seed(v::SVector{N}) where N
    SMatrix{N,N,eltype(v)}(I)
end

function seed(v)
    Matrix(Diagonal(map(one, v)))
end

function D(f)
    # grad
    function deriv(arg::AbstractArray)
        # always chunk
        res = dualrun() do
            #darr = DualArray(arg, seed(arg))
            darr = map((x, i)->Dual(x, seed(arg, i)), arg, eachindex(arg))
            @show darr
            f(darr)
        end
        tilt(allpartials(res))
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
