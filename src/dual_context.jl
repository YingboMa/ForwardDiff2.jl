using Cassette

import Cassette: overdub
using ChainRules

Cassette.@context DualContext

using ForwardDiff: Dual, value, partials, tagtype, ≺

@inline _dominant_dual(tag, maxi, i) = maxi

@inline function _dominant_dual(T, maxi, i, x::Dual{S}, tail...) where {S}
    if T === nothing || (T !== S && T ≺ S)
        _dominant_dual(S, i, i-1, tail...)
    else
        _dominant_dual(T, maxi, i-1, tail...)
    end
end

_dominant_dual(tag, maxi, i, x, tail...) = _dominant_dual(tag, maxi, i-1, tail...)

@inline dominant_dual(xs...) = _dominant_dual(nothing, 0, length(xs), reverse(xs)...)

@inline _value(::Val{T}, x) where T = x
@inline _value(::Val{T}, d::Dual{T}) where T = value(d)
@inline function _value(::Val{T}, d::Dual{S}) where {T,S}
    if S ≺ T
        d
    else
        throw(DualMismatchError(T,S))        
    end
end

@inline Base.@propagate_inbounds _partials(::Val{T}, x, i...) where T = partials(x, i...)
@inline Base.@propagate_inbounds _partials(::Val{T}, d::Dual{T}, i...) where T = partials(d, i...)
@inline function _partials(::Val{T}, d::Dual{S}, i...) where {T,S}
    if S ≺ T
        partials(zero(d))
    else
        throw(DualMismatchError(T,S))
    end
end

@inline function overdub(ctx::DualContext,
                         f,
                         args...)
    idx = dominant_dual(args...)
    if idx === 0
        Cassette.recurse(ctx, f, args...)
    else
        dtag = tagtype(args[idx])
        res = overdub(ctx, frule, f,
                      map(x->_value(Val{dtag}(), x), args)...)
        if res === nothing
            Cassette.recurse(ctx, f, args...)
        else
          # @show dtag
          # @show args
            as = map(x->_value(Val{dtag}(), x), args)
          # @show as
            ps = map(x->_partials(Val{dtag}(), x), args)
          # @show ps
            val, ∂ = res
            Dual{dtag}(val,
                       overdub(ctx,
                               ∂,
                               map(x->_partials(Val{dtag}(), x),
                                   args)...))
        end
    end
end

function dualrun(f, args...)
    ctx = DualContext()
    Cassette.overdub(ctx, f, args...)
end
