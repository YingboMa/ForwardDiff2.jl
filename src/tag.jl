struct Tag{Parent} end

@inline _find_dual(tag::T, l, i) where {T} = 0
@inline _find_dual(tag::T, l, i, x::Dual{T}, xs...) where {T} = i
@inline _find_dual(tag::T, l, i, x, xs...) where {T} = _find_dual(tag, l, i+1, xs...)

@inline oldertag(::Tag{Tag{T}}) where T = Tag{T}()
@inline oldertag(::Tag{T}) where T = nothing

@inline function find_dual(T::TT, xs...) where {TT<:Tag}
    _find_dual(T, length(xs), 1, xs...)
end

# Base case where T is not a Tag
@inline find_dual(T, xs...) = 0
