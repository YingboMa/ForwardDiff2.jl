using ChainRules

# f=hypot
function Base.hypot(dx::Dual{T}, dy::Dual{T}) where {T}
    x, y = value(dx), value(dy)
    val, df = frule(hypot, x, y)
    Dual{T}(val, df(partials(dy, 1), partials(dx, 1)))
end
