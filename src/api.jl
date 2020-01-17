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
    D(f, x)

`D(f)(x) * v` computes ``df/dx * v``
"""
struct D{T,F}
    f::F
    x::T

    D(f) = new{Nothing,typeof(f)}(f, nothing)
    (dd::D{<:Nothing,F})(x::T) where {T,F} = new{T,F}(dd.f, x)
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

unwrap_adj(x::Union{Transpose,Adjoint}) = unwrap_adj(parent(x))
unwrap_adj(x) = x

function Base.:*(dd::D{<:AbstractArray}, V::Union{AbstractArray,UniformScaling})
    # always chunk
    xx_partial = V isa UniformScaling ? seed(dd.x) : V
    duals = dualrun() do
        dualarray = DualArray(dd.x, xx_partial)
        dd.f(dualarray)
    end

    J_dual = unwrap_adj(duals)
    J_sz = (length(J_dual), length(dd.x))

    if J_dual isa AbstractArray # Jacobian
        if J_dual isa DualArray
            return allpartials(J_dual)'
        elseif J_dual isa StaticArray
            return extract_diffresult(J_dual, J_sz)
        else
            # `f: R^n -> R^m` so the Jacobian is `m × n`
            J = similar(dd.x, J_sz)
            extract_diffresult!(J, J_dual)
            return J
        end
    else # gradient
        return partials(J_dual)'
    end
end

function extract_diffresult(xs, (m, n))
    tup = mapreduce((x,y)->tuple(x..., y...), xs.data) do x
        if x isa Zero
            ntuple(_->false, n)
        else
            partials(x).data
        end
    end

    return SMatrix{n, m}(tup)'
end

function extract_diffresult!(J, ds::AbstractArray)
    @inbounds for (d, i) in zip(ds, axes(J, 2))
        if partials(d) isa Zero
            J[i, :] .= false
        else
            J[i, :] .= partials(d)
        end
    end
    return nothing
end

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
