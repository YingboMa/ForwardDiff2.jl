using StaticArrays: StaticArray, SMatrix, SVector
using LinearAlgebra

extract_diffresult(xs::AbstractArray{<:Number}) = xs
# need to optimize
extract_diffresult(xs) = hcat(xs...)'
function extract_diffresult(xs::StaticArray{<:Any,<:StaticArray})
    tup = reduce((x,y)->tuple(x..., y...), map(x->x.data, xs.data))
    SMatrix{length(xs[1]), length(xs)}(tup)'
end
extract_diffresult(xs::AbstractMatrix{<:Number}) = xs'
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

###
### Derivative object:
###

"""
    D(f, x)

`D(f, x) * v` computes ``df/dx * v``
"""
struct D{T,F}
    f::F
    x::T
end

# WARNING: It assume that the number type is commutative
Base.:*(v::Number, dd::D) = dd * v
function Base.:*(dd::D{<:Number}, v::Number)
    ps, derivative_dual = dualrun() do
        dual = Dual(dd.x, v)
        partials(dual), dd.f(dual)
    end
    derivative = map(derivative_dual) do a
        pa = partials(a)
        pa isa Zero ? zero(ps) : pa
    end
    return derivative
end

function Base.:*(dd::D{<:AbstractArray}, V::Union{AbstractArray,UniformScaling})
    # always chunk
    xx_partial = V isa UniformScaling ? seed(dd.x) : V
    J_dual = dualrun() do
        dualarray = DualArray(dd.x, xx_partial)
        dd.f(dualarray)
    end
    J = extract_diffresult(allpartials(J_dual))
    return J
end

# pretty printing
function Base.show(io::IO, dd::D)
    print(io, "D(", nameof(dd.f), ", ")
    show(IOContext(io, :compact => true, :limit => true), dd.x)
    print(io, ')')
end
