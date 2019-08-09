using Cassette

import Cassette: overdub
using ChainRules

Cassette.@context DualContext

using ForwardDiff: Dual, value, partials, Partials,
                   tagtype, ≺, DualMismatchError

@inline _dominant_dual(tag, maxi, i) = maxi

Base.@pure @inline function _dominant_dual(::Val{T}, maxi, i, x::Dual{S}, tail...) where {S, T}
    if T === nothing || (T !== S && T ≺ S)
        _dominant_dual(Val{S}(), i, i-1, tail...)
    else
        _dominant_dual(Val{T}(), maxi, i-1, tail...)
    end
end

Base.@pure @inline function _dominant_dual(tag, maxi, i, x, tail...)
    _dominant_dual(tag, maxi, i-1, tail...)
end

@inline function dominant_dual(xs...)
    _dominant_dual(Val{nothing}(), 0, length(xs), reverse(xs)...)
end

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
        zero(d)
    else
        throw(DualMismatchError(T,S))
    end
end

using ChainRules
using ChainRulesCore
import ChainRules: Wirtinger

ChainRules.Wirtinger(primal::Partials, conjugate::Union{Number, ChainRulesCore.AbstractDifferential}) = Partials(map(p->Wirtinger(p, conjugate), primal.values))


@inline overdub(ctx::DualContext, f, a...) = Cassette.recurse(ctx, f, a...)
@inline overdub(ctx::DualContext, f, a) = _overdub(ctx, f, a)
@inline overdub(ctx::DualContext, f, a, b) = _overdub(ctx, f, a, b)
@inline overdub(ctx::DualContext, f, a, b, c) = _overdub(ctx, f, a, b, c)
@inline overdub(ctx::DualContext, f, a, b, c, d) = _overdub(ctx, f, a, b, c, d)

@inline function _overdub(ctx, f, args...)
    # find the position of the dual number with the highest
    # precedence (dominant) tag
    idx = dominant_dual(args...)
    if idx === 0
        # none of the arguments are dual
        Cassette.recurse(ctx, f, args...)
    else
        # most dominant tag on the duals
        dtag = tagtype(args[idx])

        # call ChainRules.frule to execute `f` and
        # get a function that computes the partials
        res = overdub(ctx, frule, f,
                      map(x->_value(Val{dtag}(), x), args)...)

        if res === nothing
            # this means there is no frule (majority of all calls)
            return Cassette.recurse(ctx, f, args...)
        else
            # this means a result and one or more partial function
            # was computed
            vals, ∂s = res
            ps = map(x->_partials(Val{dtag}(), x), args)

            if !(∂s isa Tuple)
                # a single function scalar output
                return Dual{dtag}(vals, overdub(ctx, ∂s, ps...))
            else
                # many partial functions (as many as outputs)
                return map(vals, ∂s) do val, ∂
                    Dual{dtag}(val, overdub(ctx, ∂, ps...))
                end
            end
        end
    end
end

function dualrun(f, args...)
    ctx = DualContext()
    Cassette.overdub(ctx, f, args...)
end

function Base.show(io::IO, ::Type{<:DualContext})
    Base.printstyled(io, "DualContext", color=:light_yellow)
end
