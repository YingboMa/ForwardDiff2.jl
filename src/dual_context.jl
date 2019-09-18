using Cassette
using ChainRules

import Cassette: overdub, Context, nametype
import ForwardDiff: Dual, value, partials, Partials, tagtype, dualtag

Cassette.@context DualContext

include("tag.jl")

const TaggedCtx{T} = Context{nametype(DualContext), T}

function dualcontext()
    Cassette.disablehooks(DualContext(metadata=dualtag()))
end

# Calls to `dualtag` are aware of the current context.
# Note that the tags produced in the current context are Tag{T} where T is the metadata type of the context
@inline function overdub(ctx::C, ::typeof(dualtag)) where {T, C <: TaggedCtx{T}}
    Tag{T}()
end

@inline _value(::Any, x) = x
@inline _value(::Tag{T}, d::Dual{Tag{T}}) where T = value(d)


@inline _partials(::Any, x, i...) = partials(x, i...)
@inline _partials(::Tag{T}, d::Dual{Tag{T}}, i...) where T = partials(d, i...)
@inline _partials(::Tag{T}, x::Dual{S}, i...) where {T,S} = partials(zero(Dual{Tag{T}}), i...)

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

@inline _values(S, xs) = map(x->_value(S, x), xs)
@inline _partialss(S, xs) = map(x->_partials(S, x), xs)

@inline function _frule_overdub(ctx, tag::T, f, args...) where T
    res = Cassette.recurse(ctx, frule, f, _values(tag, args)...)

    if res === nothing
        # this means there is no frule (majority of all calls)
        return Cassette.recurse(ctx, f, args...)
    else
        # this means a result and one or more partial function
        # was computed
        vals, ∂s = res
        ps = _partialss(tag, args)

        if !(∂s isa Tuple)
            # a single function (scalar output)
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

@inline overdub(ctx::TaggedCtx, ::typeof(find_dual), args...) = find_dual(args...)

@inline function overdub(ctx::TaggedCtx{T}, f, args...) where {T}
    # find the position of the dual number with the current
    # context's tag or a child tag.
    idx = find_dual(Tag{T}(), args...)
    if idx === 0
        # none of the arguments are dual
        return Cassette.recurse(ctx, f, args...)
    else

        # We may now start operating for a completely
        # different tag -- this is OK.
        tag = tagtype(fieldtype(typeof(args), idx))()
        # call ChainRules.frule to execute `f` and
        # get a function that computes the partials
        _frule_overdub(ctx, tag, f, args...)
    end
end

function dualrun(f, args...)
    ctx = dualcontext()
    Cassette.overdub(ctx, f, args...)
end
