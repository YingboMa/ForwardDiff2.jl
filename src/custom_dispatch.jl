using Core: SSAValue
using Cassette

function isinteresting end
function alternative end

function ir_element(x, code::Vector)
    while isa(x, Core.SSAValue)
        x = code[x.id]
    end
    return x
end

##
# The Valentin Pass:
#
# Forces inlining on everything that is not marked `@noinline`
# avoids overdubbing of pure functions
# avoids overdubbing of IntrinsicFunctions and Builtins 
##
function valentin_pass(ctx, ref)
    CI = ref.code_info
    noinline = any(@nospecialize(x) ->
                       Core.Compiler.isexpr(x, :meta) &&
                       x.args[1] == :noinline,
                   CI.code)
    CI.inlineable = !noinline

    # don't overdub pure functions
    if CI.pure
        n_method_args = Int(ref.method.nargs)
        if ref.method.isva
            Cassette.insert_statements!(CI.code, CI.codelocs,
                (x, i) -> i == 1 ?  3 : nothing,
                (x, i) -> i == 1 ? [
                    # this could run into troubles when the function is @pure f(x...) since then n_method_args==2, but this seems to work sofar.
                    Expr(:call, Expr(:nooverdub, GlobalRef(Core, :tuple)), (Core.SlotNumber(i) for i in 2:(n_method_args-1))...),
                    Expr(:call, Expr(:nooverdub, GlobalRef(Core, :_apply)), Core.SlotNumber(1), Core.SSAValue(i), Core.SlotNumber(n_method_args)),
                    Expr(:return, Core.SSAValue(i+1))] : nothing)
        else
            Cassette.insert_statements!(CI.code, CI.codelocs,
                (x, i) -> i == 1 ?  2 : nothing,
                (x, i) -> i == 1 ? [
                    Expr(:call, Expr(:nooverdub, Core.SlotNumber(1)), (Core.SlotNumber(i) for i in 2:n_method_args)...)
                    Expr(:return, Core.SSAValue(i))] : nothing)
        end
        CI.ssavaluetypes = length(CI.code)
        return CI
    end

    # overdubbing IntrinsicFunctions removes our ability to profile code
    newstmt = (x, i) -> begin
        isassign = Base.Meta.isexpr(x, :(=))
        stmt = isassign ? x.args[2] : x
        if Base.Meta.isexpr(stmt, :call)
            applycall = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply), CI.code)
            if applycall
                f = stmt.args[2]
            else
                f = stmt.args[1]
            end
            f = ir_element(f, CI.code)
            if f isa GlobalRef
                mod = f.mod
                name = f.name
                if Base.isbindingresolved(mod, name) && Base.isdefined(mod, name)
                    ff = getfield(f.mod, f.name)
                    if ff isa Core.IntrinsicFunction || ff isa Core.Builtin
                        if applycall
                            stmt.args[2] = Expr(:nooverdub, f)
                        else
                            stmt.args[1] = Expr(:nooverdub, f)
                        end
                    end
                end
            end
        end
        return [x]
    end

    Cassette.insert_statements!(CI.code, CI.codelocs, (x, i) -> 1, newstmt)
    CI.ssavaluetypes = length(CI.code)
    # Core.Compiler.validate_code(CI)
    return CI
end

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

    ir = valentin_pass(ctx, ref)
    ir = ref.code_info
    ir = copy(ir)

    Cassette.insert_statements!(ir.code, ir.codelocs,
        (stmt, i) -> Base.Meta.isexpr(stmt, :call) ? 8 : nothing,
        (stmt, i) -> (s = newslot!(ir); rewrite_call(ctx, stmt, s, i)))

    Cassette.insert_statements!(ir.code, ir.codelocs,
                                (stmt, i) -> Base.Meta.isexpr(stmt, :(=)) && stmt.args[1] isa Core.SlotNumber && stmt.args[2] isa Expr && stmt.args[2].head == :call ? 8 : nothing,
                                (stmt, i) -> rewrite_call(ctx, stmt.args[2], stmt.args[1], i))

    ir.ssavaluetypes = length(ir.code)

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
