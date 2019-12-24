module DualTest

using Test
using Random
using ForwardDiff
using ForwardDiff: Partials, Dual, value, partials, tagtype

using Cassette

using NaNMath, SpecialFunctions
using DiffRules

using ForwardDiff2
import ForwardDiff2: dualrun

test_dualctx = ForwardDiff2.DualContext()

macro dtest(expr)
    :(@test Cassette.overdub(test_dualctx, ()->$(esc(expr))))
end

macro dtest_broken(expr)
    :(@test_broken Cassette.overdub(test_dualctx, ()->$(esc(expr))))
end

import Calculus

samerng() = MersenneTwister(1)

# By lower-bounding the Int range at 2, we avoid cases where differentiating an
# exponentiation of an Int value would cause a DomainError due to reducing the
# exponent by one
intrand(V) = V == Int ? rand(2:10) : rand(V)

dual_isapprox(a, b) = isapprox(a, b)
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T,T3,T4}) where {T,T1,T2,T3,T4} = isapprox(value(a), value(b)) && isapprox(partials(a), partials(b))
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T3,T4,T5}) where {T,T1,T2,T3,T4,T5} = error("Tags don't match")

dual1(primal, partial...) = dualrun(()->Dual(primal, partial))
dual2(primal, partial...) = dualrun(dualrun(()->Dual(primal, partial)))

for N in (0,3), M in (1,4), V in (Int, Float32)
    println("  ...testing Dual{..,$V,$N} and Dual{..,Dual{..,$V,$M},$N}")


    PARTIALS = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL = intrand(V)
    FDNUM = dual1(PRIMAL, PARTIALS)

    PARTIALS2 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL2 = intrand(V)
    FDNUM2 = dual1(PRIMAL2, PARTIALS2)

    PARTIALS3 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL3 = intrand(V)
    FDNUM3 = dual1(PRIMAL3, PARTIALS3)

    M_PARTIALS = Partials{M,V}(ntuple(m -> intrand(V), M))

    M_PARTIALS2 = Partials{M,V}(ntuple(m -> intrand(V), M))

    ################
    # Constructors #
    ################

    @test dual1(PRIMAL, PARTIALS...) === FDNUM

    @test typeof(dual1(widen(V)(PRIMAL), PARTIALS)) === Dual{TestTag(),widen(V),N}
    @test typeof(dual1(widen(V)(PRIMAL), PARTIALS.values)) === Dual{TestTag(),widen(V),N}
    @test typeof(dual1(widen(V)(PRIMAL), PARTIALS...)) === Dual{TestTag(),widen(V),N}

    #############
    # Accessors #
    #############

    @test value(PRIMAL) == PRIMAL
    @test value(FDNUM) == PRIMAL

    @test partials(PRIMAL) == Partials{0,V}(tuple())
    @test partials(FDNUM) == PARTIALS

    for i in 1:N
        @test partials(FDNUM, i) == PARTIALS[i]
    end

    @test ForwardDiff.npartials(FDNUM) == N
    @test ForwardDiff.npartials(typeof(FDNUM)) == N

    @test ForwardDiff.valtype(FDNUM) == V
    @test ForwardDiff.valtype(typeof(FDNUM)) == V

    #####################
    # Generic Functions #
    #####################

    @test FDNUM === copy(FDNUM)

    if V != Int
        @test eps(FDNUM) === eps(PRIMAL)
        @test eps(typeof(FDNUM)) === eps(V)

        @test floor(Int, FDNUM) === floor(Int, PRIMAL)
        @test floor(Int, FDNUM2) === floor(Int, PRIMAL2)

        @test floor(FDNUM) === floor(PRIMAL)
        @test floor(FDNUM2) === floor(PRIMAL2)

        @test ceil(Int, FDNUM) === ceil(Int, PRIMAL)
        @test ceil(Int, FDNUM2) === ceil(Int, PRIMAL2)

        @test ceil(FDNUM) === ceil(PRIMAL)
        @test ceil(FDNUM2) === ceil(PRIMAL2)

        @test trunc(Int, FDNUM) === trunc(Int, PRIMAL)
        @test trunc(Int, FDNUM2) === trunc(Int, PRIMAL2)

        @test trunc(FDNUM) === trunc(PRIMAL)
        @test trunc(FDNUM2) === trunc(PRIMAL2)

        @test round(Int, FDNUM) === round(Int, PRIMAL)
        @test round(Int, FDNUM2) === round(Int, PRIMAL2)

        @test round(FDNUM) === round(PRIMAL)
        @test round(FDNUM2) === round(PRIMAL2)

        @test Base.rtoldefault(typeof(FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
        @dtest dual1(PRIMAL-eps(V), PARTIALS) ≈ FDNUM
    end

    @test hash(FDNUM) === hash(PRIMAL)
    @test hash(FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))

    TMPIO = IOBuffer()
    write(TMPIO, FDNUM)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM)) === FDNUM
    seekstart(TMPIO)
    write(TMPIO, FDNUM2)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM2)) === FDNUM2
    seekstart(TMPIO)
    close(TMPIO)

    @test zero(FDNUM) === dual1(zero(PRIMAL), zero(PARTIALS))
    @test zero(typeof(FDNUM)) === dual1(zero(V), zero(Partials{N,V}))

    @test one(FDNUM) === dual1(one(PRIMAL), zero(PARTIALS))
    @test one(typeof(FDNUM)) === dual1(one(V), zero(Partials{N,V}))

    if V <: Integer
        @test rand(samerng(), FDNUM) == rand(samerng(), value(FDNUM))
    elseif V <: AbstractFloat
        @test rand(samerng(), typeof(FDNUM)) === dual1(rand(samerng(), V), zero(Partials{N,V}))
        @test randn(samerng(), typeof(FDNUM)) === dual1(randn(samerng(), V), zero(Partials{N,V}))
        @test randexp(samerng(), typeof(FDNUM)) === dual1(randexp(samerng(), V), zero(Partials{N,V}))
    end

    # Predicates #
    #------------#

    @test ForwardDiff.isconstant(zero(FDNUM))
    @test ForwardDiff.isconstant(one(FDNUM))
    @test ForwardDiff.isconstant(FDNUM) == (N == 0)


    @test isequal(FDNUM, dual1(PRIMAL, PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(FDNUM, FDNUM2)


    @test FDNUM == dual1(PRIMAL, PARTIALS2)
    @test (PRIMAL == PRIMAL2) == (FDNUM == FDNUM2)

    @test isless(dual1(1, PARTIALS), dual1(2, PARTIALS2))
    @test !(isless(dual1(1, PARTIALS), dual1(1, PARTIALS2)))
    @test !(isless(dual1(2, PARTIALS), dual1(1, PARTIALS2)))


    @test dual1(1, PARTIALS) < dual1(2, PARTIALS2)
    @test !(dual1(1, PARTIALS) < dual1(1, PARTIALS2))
    @test !(dual1(2, PARTIALS) < dual1(1, PARTIALS2))


    @test dual1(1, PARTIALS) <= dual1(2, PARTIALS2)
    @test dual1(1, PARTIALS) <= dual1(1, PARTIALS2)
    @test !(dual1(2, PARTIALS) <= dual1(1, PARTIALS2))


    @test dual1(2, PARTIALS) > dual1(1, PARTIALS2)
    @test !(dual1(1, PARTIALS) > dual1(1, PARTIALS2))
    @test !(dual1(1, PARTIALS) > dual1(2, PARTIALS2))


    @test dual1(2, PARTIALS) >= dual1(1, PARTIALS2)
    @test dual1(1, PARTIALS) >= dual1(1, PARTIALS2)
    @test !(dual1(1, PARTIALS) >= dual1(2, PARTIALS2))


    @test isnan(dual1(NaN, PARTIALS))
    @test !(isnan(FDNUM))


    @test isfinite(FDNUM)
    @test !(isfinite(dual1(Inf, PARTIALS)))


    @test isinf(dual1(Inf, PARTIALS))
    @test !(isinf(FDNUM))


    @test isreal(FDNUM)

    @test isinteger(dual1(1.0, PARTIALS))
    @test isinteger(FDNUM) == (V == Int)


    @test iseven(dual1(2))
    @test !(iseven(dual1(1)))

    @test iseven(dual1(dual1(2)))
    @test !(iseven(dual1(dual1(1))))

    @test isodd(dual1(1))
    @test !(isodd(dual1(2)))

    @test isodd(dual1(dual1(1)))
    @test !(isodd(dual1(dual1(2))))

    ########################
    # Promotion/Conversion #
    ########################

    WIDE_T = widen(V)

    @test promote_type(Dual{TestTag(),V,N}, V) == Dual{TestTag(),V,N}
    @test promote_type(Dual{TestTag(),V,N}, WIDE_T) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),WIDE_T,N}, V) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),V,N}, Dual{TestTag(),V,N}) == Dual{TestTag(),V,N}
    @test promote_type(Dual{TestTag(),V,N}, Dual{TestTag(),WIDE_T,N}) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),WIDE_T,N}, Dual{TestTag(),Dual{TestTag(),V,M},N}) == Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}

    # issue #322
    @test promote_type(Bool, Dual{TestTag(),V,N}) == Dual{TestTag(),promote_type(Bool, V),N}
    @test promote_type(BigFloat, Dual{TestTag(),V,N}) == Dual{TestTag(),promote_type(BigFloat, V),N}

    WIDE_FDNUM = convert(Dual{TestTag(),WIDE_T,N}, FDNUM)

    @test typeof(WIDE_FDNUM) === Dual{TestTag(),WIDE_T,N}

    @test value(WIDE_FDNUM) == PRIMAL

    @test convert(Dual, FDNUM) === FDNUM
    @test convert(Dual{TestTag(),V,N}, FDNUM) === FDNUM
    @test convert(Dual{TestTag(),WIDE_T,N}, PRIMAL) === dual1(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
    @test convert(Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}, PRIMAL) === dual1(dual1(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,Dual{TestTag(),V,M}}))
    @test convert(Dual{TestTag(),Dual{TestTag(),V,M},N}, FDNUM) === dual1(convert(Dual{TestTag(),V,M}, PRIMAL), convert(Partials{N,Dual{TestTag(),V,M}}, PARTIALS))
    @test convert(Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}, FDNUM) === dual1(convert(Dual{TestTag(),WIDE_T,M}, PRIMAL), convert(Partials{N,Dual{TestTag(),WIDE_T,M}}, PARTIALS))

    ##############
    # Arithmetic #
    ##############

    # Addition/Subtraction #
    #----------------------#

    @dtest FDNUM + FDNUM2 === dual1(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
    @dtest FDNUM + PRIMAL === dual1(value(FDNUM) + PRIMAL, partials(FDNUM))
    @dtest PRIMAL + FDNUM === dual1(value(FDNUM) + PRIMAL, partials(FDNUM))


    @dtest FDNUM - FDNUM2 === dual1(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
    @dtest FDNUM - PRIMAL === dual1(value(FDNUM) - PRIMAL, partials(FDNUM))
    @dtest PRIMAL - FDNUM === dual1(PRIMAL - value(FDNUM), -(partials(FDNUM)))
    @dtest -(FDNUM) === dual1(-(value(FDNUM)), -(partials(FDNUM)))


    # Multiplication #
    #----------------#

    @dtest FDNUM * FDNUM2 === dual1(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
    @dtest FDNUM * PRIMAL === dual1(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
    @dtest PRIMAL * FDNUM === dual1(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)


    # Division #
    #----------#

    if M > 0 && N > 0
        @dtest Dual{1}(FDNUM) / Dual{1}(PRIMAL) === Dual{1}(FDNUM / PRIMAL)
        @dtest Dual{1}(PRIMAL) / Dual{1}(FDNUM) === Dual{1}(PRIMAL / FDNUM)
        @dtest_broken Dual{1}(FDNUM) / FDNUM2 === Dual{1}(FDNUM / FDNUM2)
        @dtest_broken FDNUM / Dual{1}(FDNUM2) === Dual{1}(FDNUM / FDNUM2)
        # following may not be exact, see #264
        @dtest Dual{1}(FDNUM / PRIMAL, FDNUM2 / PRIMAL) ≈ Dual{1}(FDNUM, FDNUM2) / PRIMAL
    end

    @dtest dual_isapprox(FDNUM / FDNUM2, dual1(value(FDNUM) / value(FDNUM2), ForwardDiff._div_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM), value(FDNUM2))))
    @dtest dual_isapprox(FDNUM / PRIMAL, dual1(value(FDNUM) / PRIMAL, partials(FDNUM) / PRIMAL))
    @dtest dual_isapprox(PRIMAL / FDNUM, dual1(PRIMAL / value(FDNUM), (-(PRIMAL) / value(FDNUM)^2) * partials(FDNUM)))


    # Exponentiation #
    #----------------#

    @dtest dual_isapprox(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
    @dtest dual_isapprox(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
    @dtest dual_isapprox(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))


    #@dtest partials(NaNMath.pow(dual1(-2.0, 1.0), dual1(2.0, 0.0)), 1) == -4.0

    ###################################
    # General Mathematical Operations #
    ###################################

    #@dtest conj(FDNUM) === FDNUM

    @dtest transpose(FDNUM) === FDNUM

    @dtest abs(-FDNUM) === FDNUM
    @dtest abs(FDNUM) === FDNUM

    if V != Int
        for (M, f, arity) in DiffRules.diffrules()
            in(f, (:hankelh1, :hankelh1x, :hankelh2, :hankelh2x, :/, :rem2pi)) && continue
            println("       ...auto-testing $(M).$(f) with $arity arguments")
            if arity == 1
                deriv = DiffRules.diffrule(M, f, :x)
                modifier = in(f, (:asec, :acsc, :asecd, :acscd, :acosh, :acoth)) ? one(V) : zero(V)
                @eval begin
                    x = rand() + $modifier
                    dx = dualrun(()->$M.$f(dual1(x, one(x))))
                    @dtest value(dx) == $M.$f(x)
                    @dtest partials(dx, 1) == $deriv
                end
            elseif arity == 2
                derivs = DiffRules.diffrule(M, f, :x, :y)
                @eval begin
                    x, y = rand(1:10), rand()
                    dx = dualrun(()->$M.$f(dual1(x, one(x)), y))
                    dy = dualrun(()->$M.$f(x, dual1(y, one(y))))
                    actualdx = $(derivs[1])
                    @show actualdy = $(derivs[2])
                    @dtest value(dx) == $M.$f(x, y)
                    @dtest value(dy) == value(dx)
                    if isnan(actualdx)
                        @dtest isnan(partials(dx, 1))
                    else
                        @show dualrun(()->partials(dx, 1))
                        @show actualdx
                        @dtest partials(dx, 1) ≈ actualdx
                    end
                    if isnan(actualdy)
                        @dtest isnan(partials(dy, 1))
                    else
                        @show actualdy = $(derivs[2])
                        @show partials(dy, 1)
                        @dtest partials(dy, 1) ≈ actualdy
                    end
                end
            end
        end
    end

    # Special Cases #
    #---------------#

    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM), sqrt(2*(FDNUM^2) + FDNUM2^2))
    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM3), sqrt(FDNUM^2 + FDNUM2^2 + FDNUM3^2))

    @dtest all(map(dual_isapprox, ForwardDiff.sincos(FDNUM), (sin(FDNUM), cos(FDNUM))))

    if V === Float32
        @dtest typeof(sqrt(FDNUM)) === typeof(FDNUM)
    end

    for f in (fma, muladd)
        @dtest dual_isapprox(f(FDNUM, FDNUM2, FDNUM3),   Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS + PARTIALS3))
        @dtest dual_isapprox(f(FDNUM, FDNUM2, PRIMAL3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS))
        @dtest dual_isapprox(f(PRIMAL, FDNUM2, FDNUM3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PARTIALS3))
        @dtest dual_isapprox(f(PRIMAL, FDNUM2, PRIMAL3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2))
        @dtest dual_isapprox(f(FDNUM, PRIMAL2, FDNUM3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS + PARTIALS3))
        @dtest dual_isapprox(f(FDNUM, PRIMAL2, PRIMAL3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS))
        @dtest dual_isapprox(f(PRIMAL, PRIMAL2, FDNUM3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PARTIALS3))
    end
end

@testset "Exponentiation of zero" begin
    x0 = 0.0
    x1 = Dual{:t1}(x0, 1.0)
    x2 = Dual{:t2}(x1, 1.0)
    x3 = Dual{:t3}(x2, 1.0)
    pow = ^  # to call non-literal power
    @dtest pow(x3, 2) === x3^2 === x3 * x3
    @dtest pow(x2, 1) === x2^1 === x2
    @dtest pow(x1, 0) === x1^0 === Dual{:t1}(1.0, 0.0)
end

end # module
