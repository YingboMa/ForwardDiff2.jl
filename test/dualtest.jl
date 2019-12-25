#module DualTest

using Test
using Random
using ForwardDiff2
using ForwardDiff2: Dual, Tag, value, partials, tagtype, dualrun

using StaticArrays

using Cassette

using NaNMath, SpecialFunctions
using DiffRules

using ForwardDiff2

macro dtest(expr)
    :(@test dualrun(()->$(esc(expr))))
end

macro dtest2(expr)
    :(@test dualrun(()->dualrun(()->$(esc(expr)))))
end

macro dtest3(expr)
    :(@test dualrun(()->dualrun(()->dualrun(()->$(esc(expr))))))
end

macro dtest_broken(expr)
    :(@test_broken dualrun(()->$(esc(expr))))
end

import Calculus

const Partials{N,V} = SArray{Tuple{N},V,1,N}

const Tag1 = Tag{Nothing}
const Tag2 = Tag{Tag{Nothing}}

samerng() = MersenneTwister(1)

# By lower-bounding the Int range at 2, we avoid cases where differentiating an
# exponentiation of an Int value would cause a DomainError due to reducing the
# exponent by one
intrand(V) = V == Int ? rand(2:10) : rand(V)

dual_isapprox(a, b) = isapprox(a, b)
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T,T3,T4}) where {T,T1,T2,T3,T4} = isapprox(value(a), value(b)) && isapprox(partials(a), partials(b))
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T3,T4,T5}) where {T,T1,T2,T3,T4,T5} = error("Tags don't match")

dual1(primal, partial) = dualrun(()->Dual(primal, partial))
dual2(primal, partial) = dualrun(()->dual1(primal, partial))
dual3(primal, partial) = dualrun(()->dual2(primal, partial))

const Partials{N,V} = SVector{N,V}

for N in (0,3), M in (0,4), V in (Int, Float32)
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
    NESTED_FDNUM = dual2(dual1(PRIMAL, M_PARTIALS), PARTIALS)

    M_PARTIALS2 = Partials{M,V}(ntuple(m -> intrand(V), M))
    NESTED_FDNUM2 = dual2(dual1(PRIMAL2, M_PARTIALS2), PARTIALS2)

    ################
    # Constructors #
    ################

    @test Dual(PRIMAL, PARTIALS) === Dual{Nothing}(PRIMAL, PARTIALS)
    #@test Dual(PRIMAL) === Dual{Nothing}(PRIMAL)
    #@test dual1(PRIMAL, PARTIALS) === FDNUM

    @test typeof(NESTED_FDNUM) == Dual{Tag2,Dual{Tag1,V,Partials{M,V}},Partials{N,V}}
    @test typeof(dual1(widen(V)(PRIMAL), PARTIALS)) === Dual{Tag1,widen(V),Partials{N,widen(V)}}
    #@test typeof(dual1(widen(V)(PRIMAL), PARTIALS.values)) === DT{Tag1,widen(V),N}
    #@test typeof(dual1(widen(V)(PRIMAL), PARTIALS...)) === DT{TestTag(),widen(V),N}

    #############
    # Accessors #
    #############

    @test value(PRIMAL) == PRIMAL
    @test value(FDNUM) == PRIMAL
    @test value(NESTED_FDNUM) === dual1(PRIMAL, M_PARTIALS)

    #@test partials(PRIMAL) == Partials{0,V}(tuple())
    @test partials(FDNUM) == PARTIALS
    @test partials(NESTED_FDNUM) === PARTIALS

            global NESTED_FDNUM
    for i in 1:N
        @test partials(FDNUM, i) == PARTIALS[i]
        for j in 1:M
            # TODO: fix partials(d, i, j)
            #@test partials(NESTED_FDNUM, i, j) == partials(PARTIALS[i], j)
        end
    end

    @test ForwardDiff2.npartials(FDNUM) == N
    #@test ForwardDiff2.npartials(typeof(FDNUM)) == N
    @test ForwardDiff2.npartials(NESTED_FDNUM) == N
    #@test ForwardDiff2.npartials(typeof(NESTED_FDNUM)) == N

    @test ForwardDiff2.valtype(FDNUM) == V
    #@test ForwardDiff2.valtype(typeof(FDNUM)) == V
    @test ForwardDiff2.valtype(NESTED_FDNUM) == Dual{Tag1,V,Partials{M,V}}
    #@test ForwardDiff2.valtype(typeof(NESTED_FDNUM)) == Dual{TestTag(),V,M}

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
        @test Base.rtoldefault(typeof(NESTED_FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
        @dtest Dual(PRIMAL-eps(V), PARTIALS) ≈ FDNUM
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

    @test zero(NESTED_FDNUM) === dual2(dual1(zero(PRIMAL), zero(M_PARTIALS)), zero(PARTIALS))
    @test zero(typeof(NESTED_FDNUM)) === zero(NESTED_FDNUM)
    @test zero(FDNUM) === dual1(zero(PRIMAL), zero(PARTIALS))
    @test zero(typeof(FDNUM)) === zero(FDNUM)

    @test one(NESTED_FDNUM) === dual2(dual1(one(PRIMAL), zero(M_PARTIALS)), zero(PARTIALS))
    @test one(typeof(NESTED_FDNUM)) === one(NESTED_FDNUM)
    @test one(FDNUM) === dual1(one(PRIMAL), zero(PARTIALS))
    @test one(typeof(FDNUM)) === one(FDNUM)

    if V <: Integer
        @test rand(samerng(), FDNUM) == rand(samerng(), value(FDNUM))
        @test rand(samerng(), NESTED_FDNUM) == rand(samerng(), value(NESTED_FDNUM))
    elseif V <: AbstractFloat
        for f in (rand, randn, randexp)
            @test f(samerng(), typeof(NESTED_FDNUM)) === dual2(dual1(f(samerng(), V), zero(Partials{M,V})), zero(Partials{N,V}))
            @test f(samerng(), typeof(FDNUM)) === dual1(f(samerng(), V), zero(Partials{N,V}))
        end
    end

    # Predicates #
    #------------#

    @test ForwardDiff2.isconstant(zero(FDNUM))
    @test ForwardDiff2.isconstant(one(FDNUM))
    @test ForwardDiff2.isconstant(FDNUM) == (N == 0)

    @test ForwardDiff2.isconstant(zero(NESTED_FDNUM))
    @test ForwardDiff2.isconstant(one(NESTED_FDNUM))
    @test ForwardDiff2.isconstant(NESTED_FDNUM) == (N == 0)

    @test isequal(FDNUM, dual1(PRIMAL, PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(FDNUM, FDNUM2)

    @test isequal(NESTED_FDNUM, dual2(dual1(PRIMAL, M_PARTIALS2), PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(NESTED_FDNUM, NESTED_FDNUM2)

    @test FDNUM == dual1(PRIMAL, PARTIALS2)
    @test (PRIMAL == PRIMAL2) == (FDNUM == FDNUM2)
    @test (PRIMAL == PRIMAL2) == (NESTED_FDNUM == NESTED_FDNUM2)

    @test isless(dual1(1, PARTIALS), dual1(2, PARTIALS2))
    @test !(isless(dual1(1, PARTIALS), dual1(1, PARTIALS2)))
    @test !(isless(dual1(2, PARTIALS), dual1(1, PARTIALS2)))

    @test isless(dual2(dual1(1, M_PARTIALS), PARTIALS), dual2(dual1(2, M_PARTIALS2), PARTIALS2))
    @test !(isless(dual2(dual1(1, M_PARTIALS), PARTIALS), dual2(dual1(1, M_PARTIALS2), PARTIALS2)))
    @test !(isless(dual2(dual1(2, M_PARTIALS), PARTIALS), dual2(dual1(1, M_PARTIALS2), PARTIALS2)))

    @test dual1(1, PARTIALS) < dual1(2, PARTIALS2)
    @test !(dual1(1, PARTIALS) < dual1(1, PARTIALS2))
    @test !(dual1(2, PARTIALS) < dual1(1, PARTIALS2))

    @test dual2(dual1(1, M_PARTIALS), PARTIALS) < dual2(dual1(2, M_PARTIALS2), PARTIALS2)
    @test !(dual2(dual1(1, M_PARTIALS), PARTIALS) < dual2(dual1(1, M_PARTIALS2), PARTIALS2))
    @test !(dual1(dual1(2, M_PARTIALS), PARTIALS) < dual2(dual1(1, M_PARTIALS2), PARTIALS2))

    @test dual1(1, PARTIALS) <= dual1(2, PARTIALS2)
    @test dual1(1, PARTIALS) <= dual1(1, PARTIALS2)
    @test !(dual1(2, PARTIALS) <= dual1(1, PARTIALS2))

    @test dual2(dual1(1, M_PARTIALS), PARTIALS) <= dual2(dual1(2, M_PARTIALS2), PARTIALS2)
    @test dual2(dual1(1, M_PARTIALS), PARTIALS) <= dual2(dual1(1, M_PARTIALS2), PARTIALS2)
    @test !(dual2(dual1(2, M_PARTIALS), PARTIALS) <= dual2(dual1(1, M_PARTIALS2), PARTIALS2))

    @test dual1(2, PARTIALS) > dual1(1, PARTIALS2)
    @test !(dual1(1, PARTIALS) > dual1(1, PARTIALS2))
    @test !(dual1(1, PARTIALS) > dual1(2, PARTIALS2))

    @test dual2(dual1(2, M_PARTIALS), PARTIALS) > dual2(dual1(1, M_PARTIALS2), PARTIALS2)
    @test !(dual2(dual1(1, M_PARTIALS), PARTIALS) > dual2(dual1(1, M_PARTIALS2), PARTIALS2))
    @test !(dual2(dual1(1, M_PARTIALS), PARTIALS) > dual2(dual1(2, M_PARTIALS2), PARTIALS2))

    @test dual1(2, PARTIALS) >= dual1(1, PARTIALS2)
    @test dual1(1, PARTIALS) >= dual1(1, PARTIALS2)
    @test !(dual1(1, PARTIALS) >= dual1(2, PARTIALS2))

    @test dual2(dual1(2, M_PARTIALS), PARTIALS) >= dual2(dual1(1, M_PARTIALS2), PARTIALS2)
    @test dual2(dual1(1, M_PARTIALS), PARTIALS) >= dual2(dual1(1, M_PARTIALS2), PARTIALS2)
    @test !(dual2(dual1(1, M_PARTIALS), PARTIALS) >= dual2(dual1(2, M_PARTIALS2), PARTIALS2))

    @test isnan(dual1(NaN, PARTIALS))
    @test !(isnan(FDNUM))

    @test isnan(dual2(dual1(NaN, M_PARTIALS), PARTIALS))
    @test !(isnan(NESTED_FDNUM))

    @test isfinite(FDNUM)
    @test !(isfinite(dual1(Inf, PARTIALS)))

    @test isfinite(NESTED_FDNUM)
    @test !(isfinite(dual2(dual1(NaN, M_PARTIALS), PARTIALS)))

    @test isinf(dual1(Inf, PARTIALS))
    @test !(isinf(FDNUM))

    @test isinf(dual2(dual1(Inf, M_PARTIALS), PARTIALS))

    @test isreal(FDNUM)

    @test isinteger(dual1(1.0, PARTIALS))
    @test isinteger(FDNUM) == (V == Int)

    @test isinteger(dual2(dual1(1.0, M_PARTIALS), PARTIALS))
    @test isinteger(NESTED_FDNUM) == (V == Int)

    @test iseven(dual1(2, 1))
    @test !(iseven(dual1(1, 1)))

    @test iseven(dual2(dual1(2, 1), 1))
    @test !(iseven(dual2(dual1(1, 1), 1)))

    @test isodd(dual1(1, 1))
    @test !(isodd(dual1(2, 1)))

    @test isodd(dual2(dual1(1, 1), 1))
    @test !(isodd(dual2(dual1(2, 1), 1)))

    ########################
    # Promotion/Conversion #
    ########################

    WIDE_T = widen(V)

    @test promote_type(Dual{Tag1,V,N}, V) == Dual{Tag1,V,N}
    @test promote_type(Dual{Tag1,V,N}, WIDE_T) == Dual{Tag1,WIDE_T,N}
    @test promote_type(Dual{Tag1,WIDE_T,N}, V) == Dual{Tag1,WIDE_T,N}
    @test promote_type(Dual{Tag1,V,N}, Dual{Tag1,V,N}) == Dual{Tag1,V,N}
    @test promote_type(Dual{Tag1,V,N}, Dual{Tag1,WIDE_T,N}) == Dual{Tag1,WIDE_T,N}
    @test promote_type(Dual{Tag1,WIDE_T,N}, Dual{Tag1,Dual{Tag1,V,M},N}) == Dual{Tag1,Dual{Tag1,WIDE_T,M},N}

    # issue #322
    @test promote_type(Bool, Dual{Tag1,V,N}) == Dual{Tag1,promote_type(Bool, V),N}
    @test promote_type(BigFloat, Dual{Tag1,V,N}) == Dual{Tag1,promote_type(BigFloat, V),N}

    WIDE_FDNUM = convert(Dual{Tag1,WIDE_T,Partials{N,WIDE_T}}, FDNUM)
    WIDE_NESTED_FDNUM = convert(Dual{Tag2,Dual{Tag1,WIDE_T,Partials{M,WIDE_T}},Partials{N,WIDE_T}}, NESTED_FDNUM)

    @test typeof(WIDE_FDNUM) === Dual{Tag1,WIDE_T,Partials{N,WIDE_T}}
    @test typeof(WIDE_NESTED_FDNUM) === Dual{Tag2,Dual{Tag1,WIDE_T,Partials{M,WIDE_T}},Partials{N,WIDE_T}}

    @test value(WIDE_FDNUM) == PRIMAL
    @test value(WIDE_NESTED_FDNUM) == PRIMAL

    @test convert(Dual, FDNUM) === FDNUM
    @test convert(Dual, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{Tag1,V,Partials{N,V}}, FDNUM) === FDNUM
    @test convert(Dual{Tag2,Dual{Tag1,V,Partials{M,V}},Partials{N,V}}, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{Tag1,WIDE_T,Partials{N,WIDE_T}}, PRIMAL) === dual1(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
    @test convert(Dual{Tag2,Dual{Tag1,WIDE_T,Partials{M,WIDE_T}},Partials{N,WIDE_T}}, PRIMAL) === dual2(dual1(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,V}))
    #@test convert(Dual{Tag2,Dual{Tag1,V,Partials{M,V}},Partials{N,V}}, FDNUM) === dual2(convert(Dual{Tag1,V,Partials{N,V}}, PRIMAL), convert(Partials{N,V}, PARTIALS))
    #@test convert(Dual{Tag2,Dual{Tag1,WIDE_T,Partials{M,WIDE_T}},Partials{N,WIDE_T}}, FDNUM) === dual1(convert(Dual{Tag,WIDE_T,M}, PRIMAL), convert(Partials{N,Dual{Tag1,WIDE_T,Partials{M,WIDE_T}}}, PARTIALS))

    ##############
    # Arithmetic #
    ##############

    # Addition/Subtraction #
    #----------------------#

    @dtest FDNUM + FDNUM2 === Dual(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
    @dtest FDNUM + PRIMAL === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))
    @dtest PRIMAL + FDNUM === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))
    # TODO: fix high order dual numbers
    #=
    @dtest2 NESTED_FDNUM + NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) + value(NESTED_FDNUM2), partials(NESTED_FDNUM) + partials(NESTED_FDNUM2))
    @dtest2 NESTED_FDNUM + PRIMAL === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))
    @dtest2 PRIMAL + NESTED_FDNUM === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))

    @dtest FDNUM - FDNUM2 === Dual(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
    @dtest FDNUM - PRIMAL === Dual(value(FDNUM) - PRIMAL, partials(FDNUM))
    @dtest PRIMAL - FDNUM === Dual(PRIMAL - value(FDNUM), -(partials(FDNUM)))
    @dtest -(FDNUM) === Dual(-(value(FDNUM)), -(partials(FDNUM)))

    @dtest NESTED_FDNUM - NESTED_FDNUM2 === Dual{TestTag()}(value(NESTED_FDNUM) - value(NESTED_FDNUM2), partials(NESTED_FDNUM) - partials(NESTED_FDNUM2))
    @dtest NESTED_FDNUM - PRIMAL === Dual{TestTag()}(value(NESTED_FDNUM) - PRIMAL, partials(NESTED_FDNUM))
    @dtest PRIMAL - NESTED_FDNUM === Dual{TestTag()}(PRIMAL - value(NESTED_FDNUM), -(partials(NESTED_FDNUM)))
    @dtest -(NESTED_FDNUM) === Dual{TestTag()}(-(value(NESTED_FDNUM)), -(partials(NESTED_FDNUM)))

    # Multiplication #
    #----------------#

    @dtest FDNUM * FDNUM2 === dual1(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
    @dtest FDNUM * PRIMAL === dual1(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
    @dtest PRIMAL * FDNUM === dual1(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)

    @dtest NESTED_FDNUM * NESTED_FDNUM2 === Dual{TestTag()}(value(NESTED_FDNUM) * value(NESTED_FDNUM2), ForwardDiff._mul_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM2), value(NESTED_FDNUM)))
    @dtest NESTED_FDNUM * PRIMAL === Dual{TestTag()}(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)
    @dtest PRIMAL * NESTED_FDNUM === Dual{TestTag()}(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)

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

    @dtest dual_isapprox(NESTED_FDNUM / NESTED_FDNUM2, Dual{TestTag()}(value(NESTED_FDNUM) / value(NESTED_FDNUM2), ForwardDiff._div_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM), value(NESTED_FDNUM2))))
    @dtest dual_isapprox(NESTED_FDNUM / PRIMAL, Dual{TestTag()}(value(NESTED_FDNUM) / PRIMAL, partials(NESTED_FDNUM) / PRIMAL))
    @dtest dual_isapprox(PRIMAL / NESTED_FDNUM, Dual{TestTag()}(PRIMAL / value(NESTED_FDNUM), (-(PRIMAL) / value(NESTED_FDNUM)^2) * partials(NESTED_FDNUM)))

    # Exponentiation #
    #----------------#

    @dtest dual_isapprox(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
    @dtest dual_isapprox(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
    @dtest dual_isapprox(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))

    @dtest dual_isapprox(NESTED_FDNUM^NESTED_FDNUM2, exp(NESTED_FDNUM2 * log(NESTED_FDNUM)))
    @dtest dual_isapprox(NESTED_FDNUM^PRIMAL, exp(PRIMAL * log(NESTED_FDNUM)))
    @dtest dual_isapprox(PRIMAL^NESTED_FDNUM, exp(NESTED_FDNUM * log(PRIMAL)))

    #@dtest partials(NaNMath.pow(dual1(-2.0, 1.0), dual1(2.0, 0.0)), 1) == -4.0
    =#

    ###################################
    # General Mathematical Operations #
    ###################################

    #@dtest conj(FDNUM) === FDNUM
    #@dtest conj(NESTED_FDNUM) === NESTED_FDNUM

    @dtest transpose(FDNUM) === FDNUM
    @dtest transpose(NESTED_FDNUM) === NESTED_FDNUM

    @dtest abs(-FDNUM) === FDNUM
    @dtest abs(FDNUM) === FDNUM
    #=
    @dtest2 abs(-NESTED_FDNUM) === NESTED_FDNUM
    @dtest2 abs(NESTED_FDNUM) === NESTED_FDNUM
    =#

    if V != Int
        for (M, f, arity) in DiffRules.diffrules()
            in(f, (:hankelh1, :hankelh1x, :hankelh2, :hankelh2x, :/, :rem2pi)) && continue
            println("       ...auto-testing $(M).$(f) with $arity arguments")
            if arity == 1
                deriv = DiffRules.diffrule(M, f, :x)
                modifier = in(f, (:asec, :acsc, :asecd, :acscd, :acosh, :acoth)) ? one(V) : zero(V)
                @eval begin
                    x = rand() + $modifier
                    dx = dualrun(()->$M.$f(Dual(x, one(x))))
                    @dtest value(dx) == $M.$f(x)
                    @dtest partials(dx, 1) == $deriv
                end
            elseif arity == 2
                derivs = DiffRules.diffrule(M, f, :x, :y)
                @eval begin
                    x, y = rand(1:10), rand()
                    dx = dualrun(()->$M.$f(Dual(x, one(x)), y))
                    dy = dualrun(()->$M.$f(x, Dual(y, one(y))))
                    actualdx = $(derivs[1])
                    actualdy = $(derivs[2])
                    @dtest value(dx) == $M.$f(x, y)
                    @dtest value(dy) == value(dx)
                    if isnan(actualdx)
                        @dtest isnan(partials(dx, 1))
                    else
                        @dtest partials(dx, 1) ≈ actualdx
                    end
                    if isnan(actualdy)
                        @dtest isnan(partials(dy, 1))
                    else
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

    @dtest all(map(dual_isapprox, value(sincos(FDNUM)), (sin(FDNUM), cos(FDNUM))))

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
    x1 = dual1(x0, 1.0)
    x2 = dual2(x1, 1.0)
    x3 = dual3(x2, 1.0)
    pow = ^  # to call non-literal power
    @dtest3 pow(x3, 2) === x3^2 === x3 * x3
    @dtest2 pow(x2, 1) === x2^1 # === x2 # TODO
    @dtest pow(x1, 0) === x1^0 === Dual(1.0, 0.0)
end

#end # module
