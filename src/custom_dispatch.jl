using Core: SSAValue
using Cassette

function isinteresting end
function alternative end

# Must return 8 exprs
function rewrite_call(ctx, stmt, extraslot, i)
    exprs = Any[]
    cond = stmt.args[1]        # already an SSAValue

    # order of expressions we pre-assign
    # SSA for each expression here

    ISINTERESTING,
    GOTOIFNOTINTERESTING,
    ALTERNATIVE,
    ALTERNATIVE_IN_SLOT,
    GOTO_LAST,
    ORIGINAL,
    ORIGINAL_IN_SLOT,
    LAST = i:i+20


    # ISINTERESTING
    push!(exprs, Expr(:call,
                      Expr(:nooverdub, isinteresting),
                      Expr(:contextslot),
                      stmt.args...))

    # GOTOIFNOTINTERESTING
    push!(exprs, Expr(:gotoifnot, SSAValue(ISINTERESTING), ORIGINAL))

    # ALTERNATIVE
    push!(exprs, Expr(:call,
                      Expr(:nooverdub, alternative),
                      Expr(:contextslot),
                      stmt.args...))

    # ALTERNATIVE_IN_SLOT
    push!(exprs, Expr(:(=), extraslot, SSAValue(ALTERNATIVE)))

    # GOTO_LAST
    push!(exprs, Core.GotoNode(LAST))

    # ORIGINAL
    push!(exprs, stmt)

    # ORIGINAL_IN_SLOT
    push!(exprs, Expr(:(=), extraslot, SSAValue(ORIGINAL)))

    # LAST
    push!(exprs, extraslot)

    exprs
end

function newslot!(ir)
    extraslot = gensym("tmp")
    push!(ir.slotnames, extraslot)
    push!(ir.slotflags, 0x00)
    Core.SlotNumber(length(ir.slotnames))
end

function rewrite_ir(ctx, ref)
    # turn
    #   f(x...)
    # into
    #   %i = interesting(ctx, f, x...)
    #   gotoifnot %i #g
    #   myslot = alternative(ctx, f, x...)
    #   goto #g+1
    #   #g: myslot = f(x...)

    ir = ref.code_info
    ir = copy(ir)

    Cassette.insert_statements!(ir.code, ir.codelocs,
        (stmt, i) -> Base.Meta.isexpr(stmt, :call) ? 8 : nothing,
        (stmt, i) -> (s = newslot!(ir); rewrite_call(ctx, stmt, s, i)))

    ir.ssavaluetypes = length(ir.code)

    # Core.Compiler.validate_code(ir)
    return ir
end

const CustomDispatchPass = Cassette.@pass rewrite_ir


## Test
#=

using Cassette

Cassette.@context AnyRational

c = AnyRational(pass=CPASS.CustomDispatchPass)

@inline function CPASS.isinteresting(ctx::AnyRational, f, args...)
    return any(x->x isa Rational, args)
end

@inline function CPASS.alternative(ctx::AnyRational, f, args...)
    Cassette.overdub(ctx, f, args...)
end

=#
