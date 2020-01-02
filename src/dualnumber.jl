using Random

#=
length(::WeirdType{T,N}) where {T,N} = N
Dual(1, svec(1,2,3)) # rocks
up = SVector
Dual(1, partials) --> Dual(1, SVector(partials...))
Dual(1, partials::StaticArrays)
Dual(1, [1], y, z) --> Dual(1, [[1], y, z])
Dual(1, [1]) --> Dual(1, 1)
Dual(1, [1]) --> Dual(1, Partials([1,])) # fail Any

Dual(::Number1, [1]) --> Dual(1, Partials([1,])) # fail Any
=#

# Modified from `ForwardDiff.jl`

###########
# Prelude #
###########

const NANSAFE_MODE_ENABLED = false

const UNARY_PREDICATES = Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

const DEFAULT_CHUNK_THRESHOLD = 12

struct Chunk{N} end

const CHUNKS = [Chunk{i}() for i in 1:DEFAULT_CHUNK_THRESHOLD]

function Chunk(input_length::Integer, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    N = pickchunksize(input_length, threshold)
    N <= DEFAULT_CHUNK_THRESHOLD && return CHUNKS[N]
    return Chunk{N}()
end

function Chunk(x::AbstractArray, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    return Chunk(length(x), threshold)
end

# Constrained to `N <= threshold`, minimize (in order of priority):
#   1. the number of chunks that need to be computed
#   2. the number of "left over" perturbations in the final chunk
function pickchunksize(input_length, threshold = DEFAULT_CHUNK_THRESHOLD)
    if input_length <= threshold
        return input_length
    else
        nchunks = round(Int, input_length / DEFAULT_CHUNK_THRESHOLD, RoundUp)
        return round(Int, input_length / nchunks, RoundUp)
    end
end

chunksize(::Chunk{N}) where {N} = N

########
# Dual #
########

struct Dual{T,V,P} <: Real
    value::V
    partials::P
end

################
# Constructors #
################

# we intend for the right cassette context to
# intercept calls to dualtag
dualtag() = nothing

@inline Dual{T}(value::V, partials::P) where {T,V,P} = Dual{T,V,P}(value, partials)

#@inline Dual{T}(value::V, ::Chunk{N}, p::Val{i}) where {T,V,P,i} = Dual{T}(value, single_seed(Partials{N,V}, p))
@inline Dual(value, partials) = Dual{typeof(dualtag())}(value, partials)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x) = x
@inline value(d::Dual) = d.value

@inline valtype(::V) where {V} = V
@inline valtype(::Type{V}) where {V} = V
@inline valtype(::Dual{T,V}) where {T,V} = V
@inline valtype(::Type{Dual{T,V,P}}) where {T,V,P} = V

@inline partials(d::Dual) = d.partials

@inline npartials(d::Dual) = (ps = d.partials) isa Wirtinger ? 1 : length(ps)

#####################
# Generic Functions #
#####################

Base.copy(d::Dual) = d

Base.eps(d::Dual) = eps(value(d))
Base.eps(::Type{D}) where {D<:Dual} = eps(valtype(D))

Base.rtoldefault(::Type{D}) where {D<:Dual} = Base.rtoldefault(valtype(D))

Base.floor(::Type{R}, d::Dual) where {R<:Number} = floor(R, value(d))
Base.floor(d::Dual) = floor(value(d))

Base.ceil(::Type{R}, d::Dual) where {R<:Number} = ceil(R, value(d))
Base.ceil(d::Dual) = ceil(value(d))

Base.trunc(::Type{R}, d::Dual) where {R<:Number} = trunc(R, value(d))
Base.trunc(d::Dual) = trunc(value(d))

Base.round(::Type{R}, d::Dual) where {R<:Number} = round(R, value(d))
Base.round(d::Dual) = round(value(d))

Base.hash(d::Dual) = hash(value(d))
Base.hash(d::Dual, hsh::UInt) = hash(value(d), hsh)

function Base.read(io::IO, ::Type{Dual{T,V,P}}) where {T,V,P}
    value = read(io, V)
    partials = read(io, P)
    return Dual{T,V,P}(value, partials)
end

function Base.write(io::IO, d::Dual)
    write(io, value(d))
    write(io, partials(d))
end

@inline Base.zero(d::Dual) = zero(typeof(d))
@inline Base.zero(::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(zero(V), zero(P))

@inline Base.one(d::Dual) = one(typeof(d))
@inline Base.one(::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(one(V), zero(P))

@inline Random.rand(rng::AbstractRNG, d::Dual) = rand(rng, value(d))
@inline Random.rand(::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(rand(V), zero(P))
@inline Random.rand(rng::AbstractRNG, ::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(rand(rng, V), zero(P))
@inline Random.randn(::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(randn(V), zero(P))
@inline Random.randn(rng::AbstractRNG, ::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(randn(rng, V), zero(P))
@inline Random.randexp(::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(randexp(V), zero(P))
@inline Random.randexp(rng::AbstractRNG, ::Type{Dual{T,V,P}}) where {T,V,P} = Dual{T}(randexp(rng, V), zero(P))

# Predicates #
#------------#

isconstant(d::Dual) = iszero(partials(d))

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::Dual) = $(pred)(value(d))
end

########################
# Promotion/Conversion #
########################

Base.convert(::Type{Dual{T,V,P}}, d::Dual{T}) where {T,V,P} = Dual{T}(convert(V, value(d)), convert(P, partials(d)))
Base.convert(::Type{Dual{T,V,P}}, x) where {T,V,P} = Dual{T}(convert(V, x), zero(P))
Base.convert(::Type{Dual{T,V,P}}, x::Number) where {T,V,P} = Dual{T}(convert(V, x), zero(P))
Base.convert(::Type{D}, d::D) where {D<:Dual} = d
function Base.AbstractFloat(d::Dual{T,V,P}) where {T,V,P}
    Dual{T}(convert(promote_type(Float16, V), value(d)),
            map(p->convert(promote_type(Float16, V), p), partials(d)))
end

###################
# Pretty Printing #
###################

subscript_num(n) = join(reverse(map(d -> '₀' + d, digits(n))))

function tag_show(t, n=0)
    if t isa Nothing
        return subscript_num(n)
    elseif t isa Tag
        tag_show(oldertag(t), n+1)
    else
        return "{" * repr(t) * "}" * subscript_num(n)
    end
end

function Base.show(io::IO, d::Dual{T,V,N}) where {T,V,N}
    print(io, "(")
    print(io, value(d))
    print(io, " + ")
    color = isbits(partials(d)) ? 2 : 3
    if npartials(d) > 0
        printstyled(io, partials(d), color=color)
    else
        printstyled(io, "0", color=color)
    end
    print(io, "ϵ")
    print(io, tag_show(T()))
    print(io, ")")
    return nothing
end
