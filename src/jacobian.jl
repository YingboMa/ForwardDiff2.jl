function seed!(partials)
    # todo check squareness
    stride = sum(i->prod(j->size(partials, j), 1:i),
                 1:ndims(partials)-1) + 1

    partials[1:stride:length(partials)] .= one(eltype(partials))
end

function jacobian(f, x::AbstractVector)
    da = DualArray(similar(x, (length(x), 1+length(x))))
    value(da) .= x
    partials(da) .= zero(eltype(x))
    seed!(partials(da))
    result = f(da)
    if result isa DualArray
        return copy(partials(result))
    else eltype(result) <: Dual
        parts = partials.(result)
        J = similar(x, eltype(eltype(parts)), (length(x), length(x)))
        for j in axes(J, 2), i in axes(J, 1)
            J[i, j] = parts[i][j]
        end
        return J
    end
end
