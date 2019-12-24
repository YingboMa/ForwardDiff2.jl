using Test
using Random
using ForwardDiff
using ForwardDiff: Partials, Dual, value, partials
using ForwardDiff2: dualrun

using NaNMath, SpecialFunctions
using DiffRules

import Calculus

samerng() = MersenneTwister(1)

# By lower-bounding the Int range at 2, we avoid cases where differentiating an
# exponentiation of an Int value would cause a DomainError due to reducing the
# exponent by one
intrand(V) = V == Int ? rand(2:10) : rand(V)

dual_isapprox(a, b) = isapprox(a, b)
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T,T3,T4}) where {T,T1,T2,T3,T4} = isapprox(value(a), value(b)) && isapprox(partials(a), partials(b))
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T3,T4,T5}) where {T,T1,T2,T3,T4,T5} = error("Tags don't match")

dual(args...) = dualrun(()->Dual(args...))
dual(T::Type, V::Type, args...) = dualrun(()->Dual{typeof(ForwardDiff2.testtag()), T,V}(args...))
nesteddual(args...) = dualrun(()->dual(args...))

testtag() = dualrun(()->typeof(ForwardDiff.dualtag()))

for N in (0,3), M in (0,4), V in (Int, Float32)
    println("  ...testing Dual{$(testtag()),$V,$N} and Dual{$(testtag()),Dual{$(testtag()),$V,$M},$N}")

    PARTIALS = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL = intrand(V)
    FDNUM = dual(PRIMAL, PARTIALS)

    PARTIALS2 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL2 = intrand(V)
    FDNUM2 = dual(PRIMAL2, PARTIALS2)

    PARTIALS3 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL3 = intrand(V)
    FDNUM3 = dual(PRIMAL3, PARTIALS3)

    M_PARTIALS = Partials{M,V}(ntuple(m -> intrand(V), M))
    NESTED_PARTIALS = convert(Partials{N,Dual{testtag(),V,M}}, PARTIALS)
    NESTED_FDNUM = dual(dual(PRIMAL, M_PARTIALS), NESTED_PARTIALS)

    M_PARTIALS2 = Partials{M,V}(ntuple(m -> intrand(V), M))
    NESTED_PARTIALS2 = convert(Partials{N,Dual{testtag(),V,M}}, PARTIALS2)
    NESTED_FDNUM2 = Dual{testtag()}(Dual{testtag()}(PRIMAL2, M_PARTIALS2), NESTED_PARTIALS2)

    ################
    # Constructors #
    ################

    @test Dual{testtag()}(PRIMAL, PARTIALS...) === FDNUM
    @test Dual(PRIMAL, PARTIALS...) === Dual{Nothing}(PRIMAL, PARTIALS...)
    @test Dual(PRIMAL) === Dual{Nothing}(PRIMAL)

    @test typeof(Dual{testtag()}(widen(V)(PRIMAL), PARTIALS)) === Dual{testtag(),widen(V),N}
    @test typeof(Dual{testtag()}(widen(V)(PRIMAL), PARTIALS.values)) === Dual{testtag(),widen(V),N}
    @test typeof(Dual{testtag()}(widen(V)(PRIMAL), PARTIALS...)) === Dual{testtag(),widen(V),N}
    @test typeof(NESTED_FDNUM) == Dual{testtag(),Dual{testtag(),V,M},N}

    #############
    # Accessors #
    #############

    @test value(PRIMAL) == PRIMAL
    @test value(FDNUM) == PRIMAL
    @test value(NESTED_FDNUM) === Dual{testtag()}(PRIMAL, M_PARTIALS)

    @test partials(PRIMAL) == Partials{0,V}(tuple())
    @test partials(FDNUM) == PARTIALS
    @test partials(NESTED_FDNUM) === NESTED_PARTIALS

    for i in 1:N
        @test partials(FDNUM, i) == PARTIALS[i]
        for j in 1:M
            @test partials(NESTED_FDNUM, i, j) == partials(NESTED_PARTIALS[i], j)
        end
    end

    @test ForwardDiff.npartials(FDNUM) == N
    @test ForwardDiff.npartials(typeof(FDNUM)) == N
    @test ForwardDiff.npartials(NESTED_FDNUM) == N
    @test ForwardDiff.npartials(typeof(NESTED_FDNUM)) == N

    @test ForwardDiff.valtype(FDNUM) == V
    @test ForwardDiff.valtype(typeof(FDNUM)) == V
    @test ForwardDiff.valtype(NESTED_FDNUM) == Dual{testtag(),V,M}
    @test ForwardDiff.valtype(typeof(NESTED_FDNUM)) == Dual{testtag(),V,M}

    #####################
    # Generic Functions #
    #####################

    @test FDNUM === copy(FDNUM)
    @test NESTED_FDNUM === copy(NESTED_FDNUM)

    if V != Int
        @test eps(FDNUM) === eps(PRIMAL)
        @test eps(typeof(FDNUM)) === eps(V)
        @test eps(NESTED_FDNUM) === eps(PRIMAL)
        @test eps(typeof(NESTED_FDNUM)) === eps(V)

        @test floor(Int, FDNUM) === floor(Int, PRIMAL)
        @test floor(Int, FDNUM2) === floor(Int, PRIMAL2)
        @test floor(Int, NESTED_FDNUM) === floor(Int, PRIMAL)

        @test floor(FDNUM) === floor(PRIMAL)
        @test floor(FDNUM2) === floor(PRIMAL2)
        @test floor(NESTED_FDNUM) === floor(PRIMAL)

        @test ceil(Int, FDNUM) === ceil(Int, PRIMAL)
        @test ceil(Int, FDNUM2) === ceil(Int, PRIMAL2)
        @test ceil(Int, NESTED_FDNUM) === ceil(Int, PRIMAL)

        @test ceil(FDNUM) === ceil(PRIMAL)
        @test ceil(FDNUM2) === ceil(PRIMAL2)
        @test ceil(NESTED_FDNUM) === ceil(PRIMAL)

        @test trunc(Int, FDNUM) === trunc(Int, PRIMAL)
        @test trunc(Int, FDNUM2) === trunc(Int, PRIMAL2)
        @test trunc(Int, NESTED_FDNUM) === trunc(Int, PRIMAL)

        @test trunc(FDNUM) === trunc(PRIMAL)
        @test trunc(FDNUM2) === trunc(PRIMAL2)
        @test trunc(NESTED_FDNUM) === trunc(PRIMAL)

        @test round(Int, FDNUM) === round(Int, PRIMAL)
        @test round(Int, FDNUM2) === round(Int, PRIMAL2)
        @test round(Int, NESTED_FDNUM) === round(Int, PRIMAL)

        @test round(FDNUM) === round(PRIMAL)
        @test round(FDNUM2) === round(PRIMAL2)
        @test round(NESTED_FDNUM) === round(PRIMAL)

        @test Base.rtoldefault(typeof(FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
        @test Dual{testtag()}(PRIMAL-eps(V), PARTIALS) ≈ FDNUM
        @test Base.rtoldefault(typeof(NESTED_FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
    end

    @test hash(FDNUM) === hash(PRIMAL)
    @test hash(FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))
    @test hash(NESTED_FDNUM) === hash(PRIMAL)
    @test hash(NESTED_FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))

    TMPIO = IOBuffer()
    write(TMPIO, FDNUM)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM)) === FDNUM
    seekstart(TMPIO)
    write(TMPIO, FDNUM2)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM2)) === FDNUM2
    seekstart(TMPIO)
    write(TMPIO, NESTED_FDNUM)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(NESTED_FDNUM)) === NESTED_FDNUM
    close(TMPIO)

    @test zero(FDNUM) === Dual{testtag()}(zero(PRIMAL), zero(PARTIALS))
    @test zero(typeof(FDNUM)) === Dual{testtag()}(zero(V), zero(Partials{N,V}))
    @test zero(NESTED_FDNUM) === Dual{testtag()}(Dual{testtag()}(zero(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test zero(typeof(NESTED_FDNUM)) === Dual{testtag()}(Dual{testtag()}(zero(V), zero(Partials{M,V})), zero(Partials{N,Dual{testtag(),V,M}}))

    @test one(FDNUM) === Dual{testtag()}(one(PRIMAL), zero(PARTIALS))
    @test one(typeof(FDNUM)) === Dual{testtag()}(one(V), zero(Partials{N,V}))
    @test one(NESTED_FDNUM) === Dual{testtag()}(Dual{testtag()}(one(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test one(typeof(NESTED_FDNUM)) === Dual{testtag()}(Dual{testtag()}(one(V), zero(Partials{M,V})), zero(Partials{N,Dual{testtag(),V,M}}))

    if V <: Integer
        @test rand(samerng(), FDNUM) == rand(samerng(), value(FDNUM))
        @test rand(samerng(), NESTED_FDNUM) == rand(samerng(), value(NESTED_FDNUM))
    elseif V <: AbstractFloat
        @test rand(samerng(), typeof(FDNUM)) === Dual{testtag()}(rand(samerng(), V), zero(Partials{N,V}))
        @test rand(samerng(), typeof(NESTED_FDNUM)) === Dual{testtag()}(Dual{testtag()}(rand(samerng(), V), zero(Partials{M,V})), zero(Partials{N,Dual{testtag(),V,M}}))
        @test randn(samerng(), typeof(FDNUM)) === Dual{testtag()}(randn(samerng(), V), zero(Partials{N,V}))
        @test randn(samerng(), typeof(NESTED_FDNUM)) === Dual{testtag()}(Dual{testtag()}(randn(samerng(), V), zero(Partials{M,V})),
        zero(Partials{N,Dual{testtag(),V,M}}))
        @test randexp(samerng(), typeof(FDNUM)) === Dual{testtag()}(randexp(samerng(), V), zero(Partials{N,V}))
        @test randexp(samerng(), typeof(NESTED_FDNUM)) === Dual{testtag()}(Dual{testtag()}(randexp(samerng(), V), zero(Partials{M,V})),
        zero(Partials{N,Dual{testtag(),V,M}}))
    end

    # Predicates #
    #------------#

    @test ForwardDiff.isconstant(zero(FDNUM))
    @test ForwardDiff.isconstant(one(FDNUM))
    @test ForwardDiff.isconstant(FDNUM) == (N == 0)

    @test ForwardDiff.isconstant(zero(NESTED_FDNUM))
    @test ForwardDiff.isconstant(one(NESTED_FDNUM))
    @test ForwardDiff.isconstant(NESTED_FDNUM) == (N == 0)

    @test isequal(FDNUM, Dual{testtag()}(PRIMAL, PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(FDNUM, FDNUM2)

    @test isequal(NESTED_FDNUM, Dual{testtag()}(Dual{testtag()}(PRIMAL, M_PARTIALS2), NESTED_PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(NESTED_FDNUM, NESTED_FDNUM2)

    @test FDNUM == Dual{testtag()}(PRIMAL, PARTIALS2)
    @test (PRIMAL == PRIMAL2) == (FDNUM == FDNUM2)
    @test (PRIMAL == PRIMAL2) == (NESTED_FDNUM == NESTED_FDNUM2)

    @test isless(Dual{testtag()}(1, PARTIALS), Dual{testtag()}(2, PARTIALS2))
    @test !(isless(Dual{testtag()}(1, PARTIALS), Dual{testtag()}(1, PARTIALS2)))
    @test !(isless(Dual{testtag()}(2, PARTIALS), Dual{testtag()}(1, PARTIALS2)))

    @test isless(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS), Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(isless(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS), Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)))
    @test !(isless(Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS), NESTED_PARTIALS), Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)))

    @test Dual{testtag()}(1, PARTIALS) < Dual{testtag()}(2, PARTIALS2)
    @test !(Dual{testtag()}(1, PARTIALS) < Dual{testtag()}(1, PARTIALS2))
    @test !(Dual{testtag()}(2, PARTIALS) < Dual{testtag()}(1, PARTIALS2))

    @test Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) < Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) < Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS), NESTED_PARTIALS) < Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{testtag()}(1, PARTIALS) <= Dual{testtag()}(2, PARTIALS2)
    @test Dual{testtag()}(1, PARTIALS) <= Dual{testtag()}(1, PARTIALS2)
    @test !(Dual{testtag()}(2, PARTIALS) <= Dual{testtag()}(1, PARTIALS2))

    @test Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) <= Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) <= Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS), NESTED_PARTIALS) <= Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{testtag()}(2, PARTIALS) > Dual{testtag()}(1, PARTIALS2)
    @test !(Dual{testtag()}(1, PARTIALS) > Dual{testtag()}(1, PARTIALS2))
    @test !(Dual{testtag()}(1, PARTIALS) > Dual{testtag()}(2, PARTIALS2))

    @test Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS), NESTED_PARTIALS) > Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) > Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) > Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{testtag()}(2, PARTIALS) >= Dual{testtag()}(1, PARTIALS2)
    @test Dual{testtag()}(1, PARTIALS) >= Dual{testtag()}(1, PARTIALS2)
    @test !(Dual{testtag()}(1, PARTIALS) >= Dual{testtag()}(2, PARTIALS2))

    @test Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS), NESTED_PARTIALS) >= Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) >= Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{testtag()}(Dual{testtag()}(1, M_PARTIALS), NESTED_PARTIALS) >= Dual{testtag()}(Dual{testtag()}(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test isnan(Dual{testtag()}(NaN, PARTIALS))
    @test !(isnan(FDNUM))

    @test isnan(Dual{testtag()}(Dual{testtag()}(NaN, M_PARTIALS), NESTED_PARTIALS))
    @test !(isnan(NESTED_FDNUM))

    @test isfinite(FDNUM)
    @test !(isfinite(Dual{testtag()}(Inf, PARTIALS)))

    @test isfinite(NESTED_FDNUM)
    @test !(isfinite(Dual{testtag()}(Dual{testtag()}(NaN, M_PARTIALS), NESTED_PARTIALS)))

    @test isinf(Dual{testtag()}(Inf, PARTIALS))
    @test !(isinf(FDNUM))

    @test isinf(Dual{testtag()}(Dual{testtag()}(Inf, M_PARTIALS), NESTED_PARTIALS))
    @test !(isinf(NESTED_FDNUM))

    @test isreal(FDNUM)
    @test isreal(NESTED_FDNUM)

    @test isinteger(Dual{testtag()}(1.0, PARTIALS))
    @test isinteger(FDNUM) == (V == Int)

    @test isinteger(Dual{testtag()}(Dual{testtag()}(1.0, M_PARTIALS), NESTED_PARTIALS))
    @test isinteger(NESTED_FDNUM) == (V == Int)

    @test iseven(Dual{testtag()}(2))
    @test !(iseven(Dual{testtag()}(1)))

    @test iseven(Dual{testtag()}(Dual{testtag()}(2)))
    @test !(iseven(Dual{testtag()}(Dual{testtag()}(1))))

    @test isodd(Dual{testtag()}(1))
    @test !(isodd(Dual{testtag()}(2)))

    @test isodd(Dual{testtag()}(Dual{testtag()}(1)))
    @test !(isodd(Dual{testtag()}(Dual{testtag()}(2))))

    ########################
    # Promotion/Conversion #
    ########################

    WIDE_T = widen(V)

    @test promote_type(Dual{testtag(),V,N}, V) == Dual{testtag(),V,N}
    @test promote_type(Dual{testtag(),V,N}, WIDE_T) == Dual{testtag(),WIDE_T,N}
    @test promote_type(Dual{testtag(),WIDE_T,N}, V) == Dual{testtag(),WIDE_T,N}
    @test promote_type(Dual{testtag(),V,N}, Dual{testtag(),V,N}) == Dual{testtag(),V,N}
    @test promote_type(Dual{testtag(),V,N}, Dual{testtag(),WIDE_T,N}) == Dual{testtag(),WIDE_T,N}
    @test promote_type(Dual{testtag(),WIDE_T,N}, Dual{testtag(),Dual{testtag(),V,M},N}) == Dual{testtag(),Dual{testtag(),WIDE_T,M},N}

    # issue #322
    @test promote_type(Bool, Dual{testtag(),V,N}) == Dual{testtag(),promote_type(Bool, V),N}
    @test promote_type(BigFloat, Dual{testtag(),V,N}) == Dual{testtag(),promote_type(BigFloat, V),N}

    WIDE_FDNUM = convert(Dual{testtag(),WIDE_T,N}, FDNUM)
    WIDE_NESTED_FDNUM = convert(Dual{testtag(),Dual{testtag(),WIDE_T,M},N}, NESTED_FDNUM)

    @test typeof(WIDE_FDNUM) === Dual{testtag(),WIDE_T,N}
    @test typeof(WIDE_NESTED_FDNUM) === Dual{testtag(),Dual{testtag(),WIDE_T,M},N}

    @test value(WIDE_FDNUM) == PRIMAL
    @test value(WIDE_NESTED_FDNUM) == PRIMAL

    @test convert(Dual, FDNUM) === FDNUM
    @test convert(Dual, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{testtag(),V,N}, FDNUM) === FDNUM
    @test convert(Dual{testtag(),Dual{testtag(),V,M},N}, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{testtag(),WIDE_T,N}, PRIMAL) === Dual{testtag()}(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
    @test convert(Dual{testtag(),Dual{testtag(),WIDE_T,M},N}, PRIMAL) === Dual{testtag()}(Dual{testtag()}(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,Dual{testtag(),V,M}}))
    @test convert(Dual{testtag(),Dual{testtag(),V,M},N}, FDNUM) === Dual{testtag()}(convert(Dual{testtag(),V,M}, PRIMAL), convert(Partials{N,Dual{testtag(),V,M}}, PARTIALS))
    @test convert(Dual{testtag(),Dual{testtag(),WIDE_T,M},N}, FDNUM) === Dual{testtag()}(convert(Dual{testtag(),WIDE_T,M}, PRIMAL), convert(Partials{N,Dual{testtag(),WIDE_T,M}}, PARTIALS))

    ##############
    # Arithmetic #
    ##############

    # Addition/Subtraction #
    #----------------------#

    dualrun() do
        @test FDNUM + FDNUM2 === Dual(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
        @test FDNUM + PRIMAL === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))
        @test PRIMAL + FDNUM === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))

        @test NESTED_FDNUM + NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) + value(NESTED_FDNUM2), partials(NESTED_FDNUM) + partials(NESTED_FDNUM2))
        @test NESTED_FDNUM + PRIMAL === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))
        @test PRIMAL + NESTED_FDNUM === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))

        @test FDNUM - FDNUM2 === Dual(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
        @test FDNUM - PRIMAL === Dual(value(FDNUM) - PRIMAL, partials(FDNUM))
        @test PRIMAL - FDNUM === Dual(PRIMAL - value(FDNUM), -(partials(FDNUM)))
        @test -(FDNUM) === Dual(-(value(FDNUM)), -(partials(FDNUM)))

        @test NESTED_FDNUM - NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) - value(NESTED_FDNUM2), partials(NESTED_FDNUM) - partials(NESTED_FDNUM2))
        @test NESTED_FDNUM - PRIMAL === Dual(value(NESTED_FDNUM) - PRIMAL, partials(NESTED_FDNUM))
        @test PRIMAL - NESTED_FDNUM === Dual(PRIMAL - value(NESTED_FDNUM), -(partials(NESTED_FDNUM)))
        @test -(NESTED_FDNUM) === Dual(-(value(NESTED_FDNUM)), -(partials(NESTED_FDNUM)))

        # Multiplication #
        #----------------#

        @test FDNUM * FDNUM2 === Dual(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
        @test FDNUM * PRIMAL === Dual(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
        @test PRIMAL * FDNUM === Dual(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)

        @test NESTED_FDNUM * NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) * value(NESTED_FDNUM2), ForwardDiff._mul_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM2), value(NESTED_FDNUM)))
        @test NESTED_FDNUM * PRIMAL === Dual(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)
        @test PRIMAL * NESTED_FDNUM === Dual(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)

        # Division #
        #----------#

        if M > 0 && N > 0
            @test Dual(FDNUM) / Dual(PRIMAL) === Dual(FDNUM / PRIMAL)
            @test Dual(PRIMAL) / Dual(FDNUM) === Dual(PRIMAL / FDNUM)
            @test_broken Dual(FDNUM) / FDNUM2 === Dual(FDNUM / FDNUM2)
            @test_broken FDNUM / Dual(FDNUM2) === Dual(FDNUM / FDNUM2)
            # following may not be exact, see #264
            @test Dual(FDNUM / PRIMAL, FDNUM2 / PRIMAL) ≈ Dual(FDNUM, FDNUM2) / PRIMAL
        end

        @test dual_isapprox(FDNUM / FDNUM2, Dual(value(FDNUM) / value(FDNUM2), ForwardDiff._div_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM), value(FDNUM2))))
        @test dual_isapprox(FDNUM / PRIMAL, Dual(value(FDNUM) / PRIMAL, partials(FDNUM) / PRIMAL))
        @test dual_isapprox(PRIMAL / FDNUM, Dual(PRIMAL / value(FDNUM), (-(PRIMAL) / value(FDNUM)^2) * partials(FDNUM)))

        @test dual_isapprox(NESTED_FDNUM / NESTED_FDNUM2, Dual(value(NESTED_FDNUM) / value(NESTED_FDNUM2), ForwardDiff._div_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM), value(NESTED_FDNUM2))))
        @test dual_isapprox(NESTED_FDNUM / PRIMAL, Dual(value(NESTED_FDNUM) / PRIMAL, partials(NESTED_FDNUM) / PRIMAL))
        @test dual_isapprox(PRIMAL / NESTED_FDNUM, Dual(PRIMAL / value(NESTED_FDNUM), (-(PRIMAL) / value(NESTED_FDNUM)^2) * partials(NESTED_FDNUM)))

        # Exponentiation #
        #----------------#

        @test dual_isapprox(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
        @test dual_isapprox(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
        @test dual_isapprox(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))

        @test dual_isapprox(NESTED_FDNUM^NESTED_FDNUM2, exp(NESTED_FDNUM2 * log(NESTED_FDNUM)))
        @test dual_isapprox(NESTED_FDNUM^PRIMAL, exp(PRIMAL * log(NESTED_FDNUM)))
        @test dual_isapprox(PRIMAL^NESTED_FDNUM, exp(NESTED_FDNUM * log(PRIMAL)))

        #@test partials(NaNMath.pow(Dual(-2.0, 1.0), Dual(2.0, 0.0)), 1) == -4.0

        ###################################
        # General Mathematical Operations #
        ###################################

    end
    @test conj(FDNUM) === FDNUM
    @test conj(NESTED_FDNUM) === NESTED_FDNUM

    @test transpose(FDNUM) === FDNUM
    @test transpose(NESTED_FDNUM) === NESTED_FDNUM

    dualrun() do
        @test abs(-FDNUM) === FDNUM
        @test abs(FDNUM) === FDNUM
        @test abs(-NESTED_FDNUM) === NESTED_FDNUM
        @test abs(NESTED_FDNUM) === NESTED_FDNUM
    end

    if V != Int
        for (M, f, arity) in DiffRules.diffrules()
            in(f, (:hankelh1, :hankelh1x, :hankelh2, :hankelh2x, :/, :rem2pi)) && continue
            println("       ...auto-testing $(M).$(f) with $arity arguments")
            if arity == 1
                deriv = DiffRules.diffrule(M, f, :x)
                modifier = in(f, (:asec, :acsc, :asecd, :acscd, :acosh, :acoth)) ? one(V) : zero(V)
                @eval begin
                    x = rand() + $modifier
                    dx = $M.$f(Dual(x, one(x)))
                    @test value(dx) == $M.$f(x)
                    @test partials(dx, 1) == $deriv
                end
            elseif arity == 2
                derivs = DiffRules.diffrule(M, f, :x, :y)
                @eval begin
                    x, y = rand(1:10), rand()
                    dx = $M.$f(Dual(x, one(x)), y)
                    dy = $M.$f(x, Dual(y, one(y)))
                    actualdx = $(derivs[1])
                    actualdy = $(derivs[2])
                    @test value(dx) == $M.$f(x, y)
                    @test value(dy) == value(dx)
                    if isnan(actualdx)
                        @test isnan(partials(dx, 1))
                    else
                        @test partials(dx, 1) ≈ actualdx
                    end
                    if isnan(actualdy)
                        @test isnan(partials(dy, 1))
                    else
                        @test partials(dy, 1) ≈ actualdy
                    end
                end
            end
        end
    end

    # Special Cases #
    #---------------#

    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM), sqrt(2*(FDNUM^2) + FDNUM2^2))
    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM3), sqrt(FDNUM^2 + FDNUM2^2 + FDNUM3^2))

    #@test all(map(dual_isapprox, ForwardDiff.sincos(FDNUM), (sin(FDNUM), cos(FDNUM))))

    dualrun() do
        if V === Float32
            @test typeof(sqrt(FDNUM)) === typeof(FDNUM)
            @test typeof(sqrt(NESTED_FDNUM)) === typeof(NESTED_FDNUM)
        end

    end
        for f in (fma, muladd)
            dualrun() do
            @test dual_isapprox(f(FDNUM, FDNUM2, FDNUM3),   Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS + PARTIALS3))
        end
            @test dual_isapprox(f(FDNUM, FDNUM2, PRIMAL3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS))
            @test dual_isapprox(f(PRIMAL, FDNUM2, FDNUM3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PARTIALS3))
            @test dual_isapprox(f(PRIMAL, FDNUM2, PRIMAL3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2))
            @test dual_isapprox(f(FDNUM, PRIMAL2, FDNUM3),  Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS + PARTIALS3))
            @test dual_isapprox(f(FDNUM, PRIMAL2, PRIMAL3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS))
            @test dual_isapprox(f(PRIMAL, PRIMAL2, FDNUM3), Dual(f(PRIMAL, PRIMAL2, PRIMAL3), PARTIALS3))
        end
end

@testset "Exponentiation of zero" begin
    x0 = 0.0
    x1 = Dual{:t1}(x0, 1.0)
    x2 = Dual{:t2}(x1, 1.0)
    x3 = Dual{:t3}(x2, 1.0)
    pow = ^  # to call non-literal power
    @test pow(x3, 2) === x3^2 === x3 * x3
    @test pow(x2, 1) === x2^1 === x2
    @test pow(x1, 0) === x1^0 === Dual{:t1}(1.0, 0.0)
end
