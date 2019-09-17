import ForwardDiff: Dual

struct Tag{Parent} end

@inline _find_dual(tag, l, i) = 0
@inline _find_dual(::Tag{T}, l, i, x::Dual{Tag{T}}, xs...) where {T} = l-i
@inline _find_dual(tag, l, i, x, xs...) where {T} = _find_dual(tag, l, i+1, xs...)

@inline innertagtype(::Tag{T}) where T = T()
@inline innertagtype(::Tag{Nothing}) where T = nothing

Base.@pure @inline function find_dual(T::Tag, xs...)
    idx = _find_dual(T, length(xs), 0, xs...)
    idx === 0 ?
    _find_dual(innertagtype(T), length(xs), 0, xs...) : idx
end

# Base case where T is not a Tag
Base.@pure @inline find_dual(T, xs...) = 0
