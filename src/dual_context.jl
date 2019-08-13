using Cassette

import Cassette: overdub, Context, nametype
using ChainRules

Cassette.@context DualContext

struct Tag{Parent} end

const TaggedCtx{T} = Context{nametype(DualContext), T}

@inline function overdub(ctx::C, ::typeof(ForwardDiff.dualtag)) where {T, C <: TaggedCtx{T}}
    Tag{T}()
end

function dualcontext()
    Cassette.disablehooks(DualContext(metadata=ForwardDiff.dualtag()))
end

using ForwardDiff: Dual, value, partials, Partials,
                   tagtype, ≺, DualMismatchError

_find_dual(ctx::TaggedCtx{T}, i) where {T} = 0
_find_dual(ctx::TaggedCtx{T}, i, x::Dual{T}, xs...) where {T} = i
_find_dual(ctx::TaggedCtx{T}, i, x, xs...) where {T} = _find_dual(ctx, i-1, xs...)

@inline function find_dual(ctx::TaggedCtx, xs...)
    _find_dual(ctx, length(xs), reverse(xs)...)
end

@inline _value(::Val{T}, x) where T = x
@inline _value(::Val{T}, d::Dual{T}) where T = value(d)


@inline Base.@propagate_inbounds _partials(::Val{T}, x, i...) where T = partials(x, i...)
@inline Base.@propagate_inbounds _partials(::Val{T}, d::Dual{T}, i...) where T = partials(d, i...)
@inline Base.@propagate_inbounds _partials(::Val{T}, x::Dual{S}, i...) where {T,S} = zero(x) # FIXME: Shouldn't this be zero(Dual{T}) ?

using ChainRules
using ChainRulesCore
import ChainRulesCore: Wirtinger, mul_zero, Zero

function Wirtinger(primal::Partials, conjugate::Union{Number, ChainRulesCore.AbstractDifferential})
    Partials(map(p->Wirtinger(p, conjugate), primal.values))
end
function Wirtinger(primal::Partials, conjugate::Partials)
    Partials(map((p, c)->Wirtinger(p, c), primal.values, conjugate.values))
end
ChainRulesCore.mul_zero(::Zero, p::Partials) = zero(p)
ChainRulesCore.mul_zero(p::Partials, ::Zero) = zero(p)


@inline function overdub(ctx::TaggedCtx{T}, f, args...) where {T}
    # find the position of the dual number with the highest
    # precedence (dominant) tag
    idx = find_dual(ctx, args...)
    if idx === 0
        # none of the arguments are dual
        Cassette.recurse(ctx, f, args...)
    else
        # call ChainRules.frule to execute `f` and
        # get a function that computes the partials
        res = overdub(ctx, frule, f,
                      map(x->_value(Val{T}(), x), args)...)

        if res === nothing
            # this means there is no frule (majority of all calls)
            return Cassette.recurse(ctx, f, args...)
        else
            # this means a result and one or more partial function
            # was computed
            vals, ∂s = res
            ps = map(x->_partials(Val{T}(), x), args)

            if !(∂s isa Tuple)
                # a single function scalar output
                d = overdub(ctx, ∂s, ps...)
                return Dual{T}(vals, d)
            else
                # many partial functions (as many as outputs)
                return map(vals, ∂s) do val, ∂
                    Dual{T}(val, overdub(ctx, ∂, ps...))
                end
            end
        end
    end
end

function dualrun(f, args...)
    ctx = dualcontext()
    Cassette.overdub(ctx, f, args...)
end

function Base.show(io::IO, ::Type{<:DualContext})
    Base.printstyled(io, "DualContext", color=:light_yellow)
end
