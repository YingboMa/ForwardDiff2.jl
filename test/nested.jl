using Test
using ForwardDiff2: Dual, partials, dualrun, find_dual, Tag

const Tag1 = Tag{Nothing}
const Tag2 = Tag{Tag{Nothing}}
const tag1 = Tag1()
const tag2 = Tag2()

@testset "find_dual" begin
    @test find_dual(tag1, 0, 0) == 0
    @test find_dual(tag1, Dual{Tag1}(1,1), 1) == 1
    @test find_dual(tag1, 1, Dual{Tag1}(1,1)) == 2
    @test find_dual(tag2, 1, Dual{Tag1}(1,1)) == 0
    @test find_dual(tag2, Dual{Tag1}(1,1), 1) == 0
    @test find_dual(tag1, Dual{Tag2}(1,1), 1) == 0
end

using Cassette
Cassette.@context TestCtx

const TaggedTestCtx{T} = Cassette.Context{Cassette.nametype(TestCtx), T}

@inline function find_dual_ctx(::TaggedTestCtx{T}, args...) where T
    find_dual(Tag{T}(), args...)
end

@testset "find_dual_ctx" begin
    ctx = TaggedTestCtx(metadata=nothing)
    @test find_dual_ctx(ctx, 1, Dual{Tag1}(1,1)) == 2
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), 1) == 1
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), Dual{Tag2}(1,1)) == 1
    @test find_dual_ctx(ctx, Dual{Tag2}(1,1), Dual{Tag1}(1,1)) == 2

    ctx = TaggedTestCtx(metadata=Tag{Nothing}())
    @test find_dual_ctx(ctx, Dual{Tag1}(1,1), Dual{Tag2}(1,1)) == 2
    @test find_dual_ctx(ctx, Dual{Tag2}(1,1), Dual{Tag1}(1,1)) == 1
end

function D(f, x)
    dualrun() do
        xx = Dual(x, one(x))
        partials(f(xx), 1)
    end
end


@testset "nested differentiation" begin
    @test D(x -> x * D(y -> x + y, 1), 1) === 1
end
