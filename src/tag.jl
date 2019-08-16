import ForwardDiff: Dual

struct Tag{Parent} end

_find_dual(ctx::Type{T}, i) where {T} = 0
_find_dual(ctx::Type{T}, i, x::Type{<:Dual{T}}, xs...) where {T} = i
_find_dual(ctx::Type{T}, i, x, xs...) where {T} = _find_dual(ctx, i-1, xs...)

innertagtype(::Type{Tag{T}}) where T = T
@inline @generated function find_dual(::Type{T}, xs...) where {T<:Tag}
    idx = _find_dual(T, length(xs), reverse(xs)...)
    idx === 0 ?
    _find_dual(innertagtype(T), length(xs), reverse(xs)...) : idx
end

# Base case where T is not a Tag
@inline @generated find_dual(T, xs...) = 0

using Test

const Tag1 = Tag{Nothing}
const Tag2 = Tag{Tag1}

@testset "find_dual" begin
    @test find_dual(Tag1, 0, 0) == 0
    @test find_dual(Tag1, Dual{Tag1}(1,1), 1) == 1
    @test find_dual(Tag1, 1, Dual{Tag1}(1,1)) == 2
    @test find_dual(Tag2, 1, Dual{Tag1}(1,1)) == 2
    @test find_dual(Tag2, Dual{Tag1}(1,1), 1) == 1
    @test find_dual(Tag1, Dual{Tag2}(1,1), 1) == 0

    @test find_dual(Tag2, Dual{Tag1}, 1) == 0
    @test find_dual(Tag1, Dual{Tag2}, 1) == 0
    @test find_dual(Tag1, Dual{Tag1}, 1) == 0
end

using Cassette
Cassette.@context TestCtx

const TaggedTestCtx{T} = Cassette.Context{Cassette.nametype(TestCtx), T}

@inline function find_dual_ctx(::TaggedTestCtx{T}, args...) where T
    find_dual(Tag{T}, args...)
end

@testset "find_dual_ctx" begin
    ctx = TaggedTestCtx(metadata=nothing)
    @test find_dual_ctx(ctx, 1, Dual{Tag1}(1,1)) == 2
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), 1) == 1
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), Dual{Tag2}(1,1)) == 1
    @test find_dual_ctx(ctx, Dual{Tag2}(1,1), Dual{Tag1}(1,1)) == 2

    ctx = TaggedTestCtx(metadata=Tag2())
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), Dual{Tag2}(1,1)) == 2
    @test find_dual_ctx(ctx, Dual{Tag2}(1,1), Dual{Tag1}(1,1)) == 1
end
