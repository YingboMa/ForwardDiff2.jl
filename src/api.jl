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

    @show typeof(J_dual)
    if J_dual isa Adjoint{<:Any,<:DualArray}
        J_dual = DualArray{Tag{Nothing}}((J_dual').data', (J_dual').partials')
    end

    # TODO: fix
    # julia> D(x -> @SVector[x[1],x[2],x[3]]', @SVector[1,2,3]) * I
    # and handle Zero here, too
    #  i.e D(x -> @SVector[1,2,3]', @SVector[1,2,3]) * I
    if J_dual isa AbstractArray # Jacobian
        if J_dual isa DualArray
            return allpartials(J_dual)'
        elseif J_dual isa StaticArray
            return extract_diffresult(J_dual)
        else
            # `f: R^n -> R^m` so the Jacobian is `m × n`
            J = similar(dd.x, length(J_dual), length(dd.x))
            extract_diffresult!(J, J_dual)
            return J
        end
    else # gradient
        return partials(J_dual)'
    end
end

function extract_diffresult(xs)
    xs = map(partials, xs)
    tup = reduce((x,y)->tuple(x..., y...), map(x->x.data, xs.data))
    return SMatrix{length(xs[1]), length(xs)}(tup)'
end

function extract_diffresult!(J, ds::AbstractArray)
    @inbounds for (d, i) in zip(ds, axes(J, 2))
        J[i, :] .= partials(d)
    end
    return nothing
end

# pretty printing
function Base.show(io::IO, dd::D)
    print(io, "D(", nameof(dd.f), ", ")
    show(IOContext(io, :compact => true, :limit => true), dd.x)
    print(io, ')')
end
