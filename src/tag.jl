struct Tag{Parent} end

@inline _find_dual(tag::T, l, i) where {T} = 0
@inline _find_dual(tag::T, l, i, x::Dual{T}, xs...) where {T} = i
@inline _find_dual(tag::T, l, i, x, xs...) where {T} = _find_dual(tag, l, i+1, xs...)

@inline innertagtype(::Tag{Tag{T}}) where T = Tag{T}()
@inline innertagtype(::Tag{T}) where T = nothing
@inline function find_dual(T::TT, xs...) where {TT<:Tag}
    idx = _find_dual(T, length(xs), 1, xs...)
    idx === 0 ? find_dual(innertagtype(T), xs...) : idx
end

# Base case where T is not a Tag
@inline find_dual(T, xs...) = 0
