using StaticArrays: StaticArray, SMatrix, SVector
using LinearAlgebra

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
    D(f)

`D(f)(x) * v` computes ``\\frac{df}{dx}(x) ⋅ v``
"""
struct D{T,F}
    f::F
    x::T

    D(f) = new{Nothing,typeof(f)}(f, nothing)
    (dd::D{<:Nothing,F})(x::T) where {T,F} = new{T,F}(dd.f, x)
end

"""
    DI(f)

`DI(f)(x)` is a convenient function to compute the derivative, gradient or
Jacobian of `f` at `x`.

It is equivalent to

```julia
D(f)(x) * I
```

where `I` is the multiplicative identity of ``\\frac{df}{dx}(x)``.
"""
DI(f) = x->D(f)(x) * mul_identity(x)

mul_identity(x::AbstractArray) = I
mul_identity(x) = one(x)

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

unwrap_adj(x::Union{Transpose,Adjoint}) = unwrap_adj(parent(x))
unwrap_adj(x) = x

function Base.:*(dd::D{<:AbstractArray}, V::Union{AbstractArray,UniformScaling})
    checkinput(dd.x, V)
    xx_partial = V isa UniformScaling ? seed(dd.x) : V'
    duals = dualrun() do
        dualarray = DualArray(dd.x, xx_partial)
        dd.f(dualarray)
    end

    J_dual = unwrap_adj(duals)
    J_sz = xx_partial isa AbstractVector ?
               (length(J_dual), ) :
               (length(J_dual), size(xx_partial, 2))

    if J_dual isa AbstractArray # Jacobian
        if J_dual isa DualArray
            ps = allpartials(J_dual)
            return isjvp(J_dual) ? ps : ps'
        elseif J_dual isa StaticArray
            return extract_diffresult(J_dual, J_sz)
        else
            # `f: R^n -> R^m` so the Jacobian is `m × n`
            J = similar(J_dual, valtype(eltype(J_dual)), J_sz)
            extract_diffresult!(J, J_dual)
            return J
        end
    else # gradient
        ps = partials(J_dual)
        if V isa AbstractVector
            return ps isa Zero ? zero(J_dual) : first(ps)
        else
            return ps isa Zero ? (false * V[1, :])' : ps'
        end
    end
end

@noinline throwinput() = throw(ArgumentError("In `D(f)(x) * v`, `x` and `v` must have the same axes."))
checkinput(x, v) = (all(i->axes(x, i) == axes(v, i), 1:ndims(x)) || throwinput(); nothing)
checkinput(_, ::UniformScaling) = nothing

function extract_diffresult(xs, mn::Tuple)
    isjvp = mn isa Tuple{<:Integer}
    m, n = isjvp ? (first(mn), 1) : mn
    tup = mapreduce((x,y)->tuple(x..., y...), xs.data) do x
        if x isa Zero
            ntuple(_->false, n)
        else
            partials(x).data
        end
    end
    return isjvp ? SVector{m}(tup) : SMatrix{n, m}(tup)'
end

function extract_diffresult!(J, ds::AbstractArray)
    @inbounds for (d, i) in zip(ds, axes(J, 1))
        if partials(d) isa Zero
            J[i, :] .= false
        else
            J[i, :] .= partials(d)
        end
    end
    return nothing
end

@inline Cassette.overdub(ctx::TaggedCtx, f::typeof(extract_diffresult!), args...) = f(args...)
@inline Cassette.overdub(ctx::TaggedCtx, f::typeof(extract_diffresult), args...) = f(args...)

# pretty printing
function Base.show(io::IO, dd::D{T}) where T
    print(io, "D(", nameof(dd.f), ')')
    if !(T <: Nothing)
        print(io, '(')
        show(IOContext(io, :compact => true, :limit => true), dd.x)
        print(io, ')')
    end
    return nothing
end
