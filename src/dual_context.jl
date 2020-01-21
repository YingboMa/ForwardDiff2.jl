using Cassette
using SpecializeVarargs: @specialize_vararg
using ChainRules
using ChainRulesCore
import ChainRulesCore: Zero

using Cassette: overdub, Context, nametype, similarcontext

Cassette.@context DualContext

const TaggedCtx{T} = Context{nametype(DualContext),T}

function dualcontext()
    # Note that the `dualtag()` is not of the same type as that of the
    # Duals constructed in this context, because it is called in the older context
    # New duals are of type Tag{typeof(dualtag())}
    Cassette.disablehooks(DualContext(metadata=dualtag(), pass=CustomDispatchPass))
end

# Calls to `dualtag` are aware of the current context. Note that the tags
# produced in the current context are `Tag{T} where T` is the metadata type of
# the context.
@inline Cassette.overdub(ctx::TaggedCtx{T}, ::typeof(dualtag)) where {T} = Tag{T}()

@inline Cassette.overdub(ctx::TaggedCtx, ::typeof(find_dual), args...) = find_dual(args...)

@inline _value(::Any, x) = x
@inline _value(::Tag{T}, d::Dual{Tag{T}}) where T = value(d)


@inline _partials(::Any, x) = Zero()
@inline _partials(::Tag{T}, d::Dual{Tag{T}}) where T = d.partials

@inline _values(S, xs) = map(x->_value(S, x), xs)
@inline _partialss(S, xs) = map(x->_partials(S, x), xs)

@inline anydual(x, xs...) = anydual(xs...)
@inline anydual(x::Dual, xs...) = true
@inline anydual() = false

# necessary special cases:

# this makes splatting work...
@inline isinteresting(ctx::TaggedCtx, f::typeof(Core._apply), g, xs...) = Core._apply(isinteresting, (ctx, g), xs...)
@inline function alternative(ctx::TaggedCtx{T}, f::typeof(Core._apply), g, args...) where {T}
    Core._apply(alternative, (ctx, g), args...)
end

# actually interesting:
@inline isinteresting(ctx::TaggedCtx, f, a) = anydual(a)
@inline isinteresting(ctx::TaggedCtx, f, a, b) = anydual(a, b)
@inline isinteresting(ctx::TaggedCtx, f, a, b, c) = anydual(a, b, c)
@inline isinteresting(ctx::TaggedCtx, f, a, b, c, d) = anydual(a, b, c, d)
@inline isinteresting(ctx::TaggedCtx, f, args...) = anydual(args...)
@inline isinteresting(ctx::TaggedCtx, f::Core.Builtin, args...) = false
@inline isinteresting(ctx::TaggedCtx, f::Union{typeof(ForwardDiff2.find_dual),
                                               typeof(ForwardDiff2.anydual)}, args...) = false

@specialize_vararg 4 @inline function _frule_overdub2(ctx::TaggedCtx{T}, f::F, args...) where {T,F}
    # Here we can assume that one or more `args` is a Dual with tag
    # of type T.

    tag = Tag{T}()
    # unwrap only duals with the tag T.
    vs = _values(tag, args)

    # extract the partials only for the current tag
    # so we can pass them to the pushforward
    ps = _partialss(tag, args)

    # default `dself` to `Zero()`
    dself = Zero()

    # call frule to see if there is a rule for this call:
    if ctx.metadata isa Tag
        ctx1 = similarcontext(ctx, metadata=oldertag(ctx.metadata))

        # we call frule with an older context because the Dual numbers may
        # themselves contain Dual numbers that were created in an older context
        frule_result = overdub(ctx1, frule, f, vs..., dself, ps...)
    else
        frule_result = frule(f, vs..., dself, ps...)
    end

    if frule_result === nothing
        # this means there is no frule
        # We can't just do f(args...) here because `f` might be
        # a closure which closes over a Dual number, hence we call
        # recurse. Recurse overdubs the calls inside `f` and not `f` itself
        return Cassette.overdub(ctx, f, args...)
    else
        # this means there exists an frule for this specific call.
        # frule_result is then a tuple (val, pushforward) where val
        # is the primal result. (Note: this may be Dual numbers but only
        # with an older tag)
        val, ∂s = frule_result

        return if ∂s isa Tuple
            map(val, ∂s) do v, ∂
                ∂ isa Zero ? v : Dual{Tag{T}}(v, ∂)
            end
        else
            ∂s isa Zero ? val : Dual{Tag{T}}(val, ∂s)
        end
    end
end

@specialize_vararg 4 @inline function alternative(ctx::TaggedCtx{T}, f::F, args...) where {T,F}
    # This method only executes if `args` contains at least 1 Dual
    # the question is what is its tag

    tag = Tag{T}()

    idx = find_dual(tag, args...)
    if f === Dual
        return overdub(ctx, f, args...)
    elseif idx === 0
        # This is the base case for the recursion in this function which
        # tries to do the alternative with successively older contexts
        # the oldest context is not a `Tag`
        if !(T<:Tag)
            return Cassette.fallback(ctx, f, args...)
        end
        # none of the arguments have the same tag as the context
        # try with the parent context
        ctx1 = similarcontext(ctx, metadata=oldertag(ctx.metadata))
        return overdub(ctx1, f, args...)
    else
        # call ChainRules.frule to execute `f` and
        # get a function that computes the partials
        return _frule_overdub2(ctx, f, args...)
    end
end

function dualrun(f, args...)
    ctx = dualcontext()
    return overdub(ctx, f, args...)
end

const BINARY_PREDICATES = Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=)]

for pred in BINARY_PREDICATES
    @eval function alternative(ctx::TaggedCtx, ::typeof($(pred)), x, y)
        vx, vy = value(x), value(y)
        return isinteresting(ctx, $pred, vx, vy) ?
        alternative(ctx, $pred, vx, vy) :
        $pred(vx, vy)
    end
end

###
### Promotion
###
Cassette.overdub(::TaggedCtx, ::typeof(Base.promote_rule), ::Type{R}, ::Type{Dual{T,V,P}}) where {R,T,V,P} = Dual{T,promote_type(R, V),P}
Cassette.overdub(ctx::TaggedCtx, f::typeof(Base.promote_rule), a::Type{Dual{T,V,P}}, b::Type{R}) where {R,T,V,P} = overdub(ctx, f, b, a)
function Cassette.overdub(ctx::TaggedCtx, ::typeof(Base.promote_rule), ::Type{Dual{T,V1,P}}, ::Type{Dual{T,V2,P}}) where {T,V1,V2,P}
    V3 = overdub(ctx, promote_type, V1, V2)
    return Dual{T,V3,P}
end
# TODO: use multiple dispatch
function Cassette.overdub(ctx::TaggedCtx{S}, ::typeof(Base.promote_rule), ::Type{Dual{T1,V1,P1}}, ::Type{Dual{T2,V2,P2}}) where {T1,T2,V1,V2,P1,P2,S}
    ctx1 = S === Nothing ? ctx : similarcontext(ctx, metadata=oldertag(ctx.metadata))
    V3 = overdub(ctx1, promote_type, V1, V2)
    if Tag{S} === T1
        return Dual{T1,V3,P1}
    elseif Tag{S} === T2
        return Dual{T2,V3,P2}
    end
    promote_error()
end
@noinline promote_error() = error("this cannot happen")

#### recursion early termination condition
@inline Cassette.overdub(ctx::TaggedCtx, f::Core.Builtin, args...) = f(args...)
@inline Cassette.overdub(ctx::TaggedCtx{T}, f::Union{typeof(value),typeof(partials)}, d::Dual{T}) where {T<:Tag{Nothing}} = f(d)
@inline Cassette.overdub(ctx::TaggedCtx{T}, f::typeof(allpartials), d::AbstractArray{<:Dual{T}})  where {T<:Tag{Nothing}} = f(d)
@inline Cassette.overdub(ctx::TaggedCtx, f::typeof(Base.Broadcast.check_broadcast_axes), args...) = f(args...)
@inline Cassette.overdub(ctx::TaggedCtx, f::typeof(Base.Broadcast.preprocess_args), args...) = f(args...)

##### Inference Hacks
@inline isinteresting(ctx::TaggedCtx, f::Union{typeof(Base.print_to_string),typeof(hash)}, args...) = false
@noinline Cassette.overdub(ctx::TaggedCtx, f::Union{typeof(Base.print_to_string),typeof(hash),typeof(Core.throw)}, args...) = f(args...)
