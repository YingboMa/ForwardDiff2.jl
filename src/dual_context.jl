using Cassette
using ChainRules
using ChainRulesCore
import ChainRulesCore: Wirtinger, Zero

using Cassette: overdub, Context, nametype
import ForwardDiff: Dual, value, partials, Partials, tagtype, dualtag

Cassette.@context DualContext

const TaggedCtx{T} = Context{nametype(DualContext),T}

function dualcontext()
    Cassette.disablehooks(DualContext(metadata=dualtag(), pass=CustomDispatchPass))
end

# Calls to `dualtag` are aware of the current context. Note that the tags
# produced in the current context are `Tag{T} where T` is the metadata type of
# the context.
@inline Cassette.overdub(ctx::C, ::typeof(dualtag)) where {T,C<:TaggedCtx{T}} = Tag{T}()

@inline Cassette.overdub(ctx::TaggedCtx, ::typeof(find_dual), args...) = find_dual(args...)

@inline _value(::Any, x) = x
@inline _value(::Tag{T}, d::Dual{Tag{T}}) where T = value(d)


@inline _partials(::Any, x, i...) = partials(x, i...)
@inline _partials(::Tag{T}, d::Dual{Tag{T}}, i...) where T = partials(d, i...)
@inline _partials(::Tag{T}, x::Dual{S}, i...) where {T,S} = Zero()


function Wirtinger(primal::Partials, conjugate::Union{Number,ChainRulesCore.AbstractDifferential})
    return Partials(map(p->Wirtinger(p, conjugate), primal.values))
end
function Wirtinger(primal::Partials, conjugate::Partials)
    return Partials(map((p, c)->Wirtinger(p, c), primal.values, conjugate.values))
end
@inline _values(S, xs) = map(x->_value(S, x), xs)
@inline _partialss(S, xs) = map(x->_partials(S, x), xs)

@inline function _frule_overdub(ctx::TaggedCtx{S}, tag::T, f, args...) where {T,S}
    vs = _values(tag, args)
    res = alternative(ctx, frule, f, vs...)

    if res === nothing
        # this means there is no frule (majority of all calls)
        if f === frule # PSYCH!!
            # do not do frule of frule!
            return alternative(ctx, f, vs...)
        else
            return Cassette.recurse(ctx, f, args...)
        end
    else
        # this means a result and one or more partial function
        # was computed
        vals, pushfwd = res
        ps = _partialss(tag, args)

        if !(pushfwd isa Tuple)
            # a single function (scalar output)
            d = overdub(ctx, pushfwd, Zero(), ps...)
            return Dual{T}(vals, d)
        else
            # many partial functions (as many as outputs)
            return map(vals, pushfwd) do val, ∂
                Dual{T}(val, overdub(ctx, ∂, Zero(), ps...))
            end
        end
    end
end

@inline anydual(x, xs...) = anydual(xs...)
@inline anydual(x::Dual, xs...) = true
@inline anydual() = false

# necessary special cases:

# this makes splatting work...
@inline isinteresting(ctx::TaggedCtx, f::typeof(Core._apply), g, xs...) = Core._apply(isinteresting, (ctx, g), xs...)
@inline function alternative(ctx::TaggedCtx{T}, f::typeof(Core._apply), g, args...) where {T}
    Core._apply(alternative, (ctx, g), args...)
end

# this makes `log` work by making throw_complex_domainerror inferable, but not really sure why
@inline isinteresting(ctx::TaggedCtx, f::typeof(Core.throw), xs) = true
@inline alternative(ctx, f::typeof(Core.throw), arg) = throw(arg)

# actually interesting:

@inline isinteresting(ctx::TaggedCtx, f, a) = anydual(a)
@inline isinteresting(ctx::TaggedCtx, f, a, b) = anydual(a, b)
@inline isinteresting(ctx::TaggedCtx, f, a, b, c) = anydual(a, b, c)
@inline isinteresting(ctx::TaggedCtx, f, a, b, c, d) = anydual(a, b, c, d)
@inline isinteresting(ctx::TaggedCtx, f, args...) = false

@inline function alternative(ctx::TaggedCtx{T}, f, args...) where {T}
    idx = find_dual(Tag{T}(), args...)
    if idx === 0
        # none of the arguments are dual
        return overdub(ctx, f, args...)
    else
        tag = tagtype(fieldtype(typeof(args), idx))()
        # call ChainRules.frule to execute `f` and
        # get a function that computes the partials
        return _frule_overdub(ctx, tag, f, args...)
    end
end

function dualrun(f, args...)
    ctx = dualcontext()
    overdub(ctx, f, args...)
end
