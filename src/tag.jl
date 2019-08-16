import ForwardDiff: Dual

struct Tag{Parent} end

_find_dual(ctx::Type{T}, i) where {T} = 0
_find_dual(ctx::Type{T}, i, x::Type{<:Dual{T}}, xs...) where {T} = i
_find_dual(ctx::Type{T}, i, x, xs...) where {T} = _find_dual(ctx, i-1, xs...)

innertagtype(::Type{Tag{T}}) where T = T
@inline @generated function find_dual(::Type{T}, xs...) where {T<:Tag}
    idx = _find_dual(T, length(xs), reverse(xs)...)
    idx === 0 ?
    _find_dual(innertagtype(T), length(xs), reverse(xs)...) : idx
end

# Base case where T is not a Tag
@inline @generated find_dual(T, xs...) = 0
