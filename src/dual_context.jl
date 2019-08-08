using Cassette
using ChainRules

Cassette.@context DualContext

using ForwardDiff: Dual, value, partials, tagtype
@inline _first_dual(i) = 0
@inline _first_dual(i, x, tail...) = x isa Dual ? i : _first_dual(i-1, tail...)

@inline last_dual(xs...) = _first_dual(length(xs), reverse(xs)...)

@inline function Cassette.overdub(ctx::DualContext,
                          f,
                          args...)
    idx = last_dual(args...)
    if idx === 0
        Cassette.recurse(ctx, f, args...)
    else
        res = frule(f, map(value, args)...)
        if res === nothing
            Cassette.recurse(ctx, f, args...)
        else
            val, ∂ = res
            Dual{tagtype(args[idx])}(val, ∂(map(partials, args)...))
        end
    end
end

