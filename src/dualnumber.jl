using Random

# Copied from `ForwardDiff.jl`

###########
# Prelude #
###########

const NANSAFE_MODE_ENABLED = false

const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

const UNARY_PREDICATES = Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

const BINARY_PREDICATES = Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=)]

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

replace_match!(f, ismatch, x) = x

function replace_match!(f, ismatch, lines::AbstractArray)
    for i in eachindex(lines)
        line = lines[i]
        if ismatch(line)
            lines[i] = f(line)
        elseif isa(line, Expr)
            replace_match!(f, ismatch, line.args)
        end
    end
    return lines
end

# This is basically a workaround that allows one to use CommonSubexpressions.cse, but with
# qualified bindings. This is not guaranteed to be correct if the input expression contains
# field accesses.
function qualified_cse!(expr)
    placeholders = Dict{Symbol,Expr}()
    replace_match!(x -> isa(x, Expr) && x.head == :(.), expr.args) do x
        placeholder = Symbol("#$(hash(x))")
        placeholders[placeholder] = x
        placeholder
    end
    cse_expr = CommonSubexpressions.cse(expr, false)
    replace_match!(x -> haskey(placeholders, x), cse_expr.args) do x
        placeholders[x]
    end
    return cse_expr
end

############
# Partials #
############

struct Partials{N,V} <: AbstractVector{V}
    values::NTuple{N,V}
end

##############################
# Utility/Accessor Functions #
##############################

@generated function single_seed(::Type{Partials{N,V}}, ::Val{i}) where {N,V,i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(V)), :(zero(V))) for j in 1:N]...)
    return :(Partials($(ex)))
end

@inline valtype(::Partials{N,V}) where {N,V} = V
@inline valtype(::Type{Partials{N,V}}) where {N,V} = V

@inline npartials(::Partials{N}) where {N} = N
@inline npartials(::Type{Partials{N,V}}) where {N,V} = N

@inline Base.length(::Partials{N}) where {N} = N
@inline Base.size(::Partials{N}) where {N} = (N,)

@inline Base.@propagate_inbounds Base.getindex(partials::Partials, I...) = partials.values[I...]

Base.iterate(partials::Partials) = iterate(partials.values)
Base.iterate(partials::Partials, i) = iterate(partials.values, i)

Base.IndexStyle(::Type{<:Partials}) = IndexLinear()

# Can be deleted after https://github.com/JuliaLang/julia/pull/29854 is on a release
Base.mightalias(x::AbstractArray, y::Partials) = false

#####################
# Generic Functions #
#####################

@inline iszero(partials::Partials) = iszero_tuple(partials.values)

@inline Base.zero(partials::Partials) = zero(typeof(partials))
@inline Base.zero(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(zero_tuple(NTuple{N,V}))

@inline Base.one(partials::Partials) = one(typeof(partials))
@inline Base.one(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(one_tuple(NTuple{N,V}))

@inline Random.rand(partials::Partials) = rand(typeof(partials))
@inline Random.rand(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(NTuple{N,V}))
@inline Random.rand(rng::AbstractRNG, partials::Partials) = rand(rng, typeof(partials))
@inline Random.rand(rng::AbstractRNG, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(rng, NTuple{N,V}))

Base.isequal(a::Partials{N}, b::Partials{N}) where {N} = isequal(a.values, b.values)
Base.:(==)(a::Partials{N}, b::Partials{N}) where {N} = a.values == b.values

const PARTIALS_HASH = hash(Partials)

Base.hash(partials::Partials) = hash(partials.values, PARTIALS_HASH)
Base.hash(partials::Partials, hsh::UInt64) = hash(hash(partials), hsh)

@inline Base.copy(partials::Partials) = partials

Base.read(io::IO, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(ntuple(i->read(io, V), N))

function Base.write(io::IO, partials::Partials)
    for p in partials
        write(io, p)
    end
end

########################
# Conversion/Promotion #
########################

Base.promote_rule(::Type{Partials{N,A}}, ::Type{Partials{N,B}}) where {N,A,B} = Partials{N,promote_type(A, B)}

Base.convert(::Type{Partials{N,V}}, partials::Partials) where {N,V} = Partials{N,V}(partials.values)
Base.convert(::Type{Partials{N,V}}, partials::Partials{N,V}) where {N,V} = partials

########################
# Arithmetic Functions #
########################

@inline Base.:+(a::Partials{N}, b::Partials{N}) where {N} = Partials(add_tuples(a.values, b.values))
@inline Base.:-(a::Partials{N}, b::Partials{N}) where {N} = Partials(sub_tuples(a.values, b.values))
@inline Base.:-(partials::Partials) = Partials(minus_tuple(partials.values))
@inline Base.:*(x::Real, partials::Partials) = partials*x

@inline function _div_partials(a::Partials, b::Partials, aval, bval)
    return _mul_partials(a, b, inv(bval), -(aval / (bval*bval)))
end

# NaN/Inf-safe methods #
#----------------------#

if NANSAFE_MODE_ENABLED
    @inline function Base.:*(partials::Partials, x::Real)
        x = ifelse(!isfinite(x) && iszero(partials), one(x), x)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        x = ifelse(x == zero(x) && iszero(partials), one(x), x)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        x_a = ifelse(!isfinite(x_a) && iszero(a), one(x_a), x_a)
        x_b = ifelse(!isfinite(x_b) && iszero(b), one(x_b), x_b)
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
else
    @inline function Base.:*(partials::Partials, x::Real)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
end

# edge cases where N == 0 #
#-------------------------#

@inline Base.:+(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:+(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, b)
@inline Base.:+(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)

@inline Base.:-(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:-(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = -(convert(Partials{N,promote_type(A,B)}, b))
@inline Base.:-(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)
@inline Base.:-(partials::Partials{0,V}) where {V} = partials

@inline Base.:*(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())
@inline Base.:*(x::Real, partials::Partials{0,V}) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline Base.:/(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline _mul_partials(a::Partials{0,A}, b::Partials{0,B}, afactor, bfactor) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline _mul_partials(a::Partials{0,A}, b::Partials{N,B}, afactor, bfactor) where {N,A,B} = bfactor * b
@inline _mul_partials(a::Partials{N,A}, b::Partials{0,B}, afactor, bfactor) where {N,A,B} = afactor * a

##################################
# Generated Functions on NTuples #
##################################
# The below functions are generally
# equivalent to directly mapping over
# tuples using `map`, but run a bit
# faster since they generate inline code
# that doesn't rely on closures.

function tupexpr(f, N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@inline iszero_tuple(::Tuple{}) = true
@inline zero_tuple(::Type{Tuple{}}) = tuple()
@inline one_tuple(::Type{Tuple{}}) = tuple()
@inline rand_tuple(::AbstractRNG, ::Type{Tuple{}}) = tuple()
@inline rand_tuple(::Type{Tuple{}}) = tuple()

@generated function iszero_tuple(tup::NTuple{N,V}) where {N,V}
    ex = Expr(:&&, [:(z == tup[$i]) for i=1:N]...)
    return quote
        z = zero(V)
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@generated function zero_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = zero(V)
        return $ex
    end
end

@generated function one_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = one(V)
        return $ex
    end
end

@generated function rand_tuple(rng::AbstractRNG, ::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(rng, V)), N)
end

@generated function rand_tuple(::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(V)), N)
end

@generated function scale_tuple(tup::NTuple{N}, x) where N
    return tupexpr(i -> :(tup[$i] * x), N)
end

@generated function div_tuple_by_scalar(tup::NTuple{N}, x) where N
    return tupexpr(i -> :(tup[$i] / x), N)
end

@generated function add_tuples(a::NTuple{N}, b::NTuple{N})  where N
    return tupexpr(i -> :(a[$i] + b[$i]), N)
end

@generated function sub_tuples(a::NTuple{N}, b::NTuple{N})  where N
    return tupexpr(i -> :(a[$i] - b[$i]), N)
end

@generated function minus_tuple(tup::NTuple{N}) where N
    return tupexpr(i -> :(-tup[$i]), N)
end

@generated function mul_tuples(a::NTuple{N}, b::NTuple{N}, afactor, bfactor) where N
    return tupexpr(i -> :((afactor * a[$i]) + (bfactor * b[$i])), N)
end

###################
# Pretty Printing #
###################

Base.show(io::IO, p::Partials{N}) where {N} = print(io, "Partials", p.values)

########
# Dual #
########

"""
    ForwardDiff2.can_dual(V::Type)

Determines whether the type V is allowed as the scalar type in a
Dual. By default, only `<:Real` types are allowed.
"""
can_dual(::Type{<:Real}) = true
can_dual(::Type) = false

struct Dual{T,V,N} <: Real
    value::V
    partials::Partials{N,V}
    function Dual{T, V, N}(value::V, partials::Partials{N, V}) where {T, V, N}
        can_dual(V) || throw_cannot_dual(V)
        new{T, V, N}(value, partials)
    end
end

##############
# Exceptions #
##############

struct DualMismatchError{A,B} <: Exception
    a::A
    b::B
end

Base.showerror(io::IO, e::DualMismatchError{A,B}) where {A,B} =
    print(io, "Cannot determine ordering of Dual tags $(e.a) and $(e.b)")

@noinline function throw_cannot_dual(V::Type)
    throw(ArgumentError("Cannot create a dual over scalar type $V." *
        " If the type behaves as a scalar, define FowardDiff.can_dual."))
end

"""
    ForwardDiff2.≺(a, b)::Bool

Determines the order in which tagged `Dual` objects are composed. If true, then `Dual{b}`
objects will appear outside `Dual{a}` objects.

This is important when working with nested differentiation: currently, only the outermost
tag can be extracted, so it should be used in the _innermost_ function.
"""
@noinline ≺(a,b) = throw(DualMismatchError(a,b))

################
# Constructors #
################

@inline Dual{T}(value::V, partials::Partials{N,V}) where {T,N,V} = Dual{T,V,N}(value, partials)

@inline function Dual{T}(value::A, partials::Partials{N,B}) where {T,N,A,B}
    C = promote_type(A, B)
    return Dual{T}(convert(C, value), convert(Partials{N,C}, partials))
end

# we intend for the right cassette context to
# intercept calls to dualtag
dualtag() = nothing

@inline Dual{T}(value, partials::Tuple) where {T} = Dual{T}(value, Partials(partials))
@inline Dual{T}(value, partials::Tuple{}) where {T} = Dual{T}(value, Partials{0,typeof(value)}(partials))
@inline Dual{T}(value) where {T} = Dual{T}(value, ())
@inline Dual{T}(x::Dual{T}) where {T} = Dual{T}(x, ())
@inline Dual{T}(value, partial1, partials...) where {T} = Dual{T}(value, tuple(partial1, partials...))
@inline Dual{T}(value::V, ::Chunk{N}, p::Val{i}) where {T,V,N,i} = Dual{T}(value, single_seed(Partials{N,V}, p))
@inline Dual(args...) = Dual{typeof(dualtag())}(args...)

# we define these special cases so that the "constructor <--> convert" pun holds for `Dual`
@inline Dual{T,V,N}(x::Dual{T,V,N}) where {T,V,N} = x
@inline Dual{T,V,N}(x) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V,N}(x::Number) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V}(x) where {T,V} = convert(Dual{T,V}, x)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x) = x
@inline value(d::Dual) = d.value

@inline value(::Type{T}, x) where T = x
@inline value(::Type{T}, d::Dual{T}) where T = value(d)
@inline function value(::Type{T}, d::Dual{S}) where {T,S}
    if S ≺ T
        d
    else
        throw(DualMismatchError(T,S))        
    end
end

@inline partials(x) = Partials{0,typeof(x)}(tuple())
@inline partials(d::Dual) = d.partials
@inline partials(x, i...) = zero(x)
@inline Base.@propagate_inbounds partials(d::Dual, i) = d.partials[i]
@inline Base.@propagate_inbounds partials(d::Dual, i, j) = partials(d, i).partials[j]
@inline Base.@propagate_inbounds partials(d::Dual, i, j, k...) = partials(partials(d, i, j), k...)

@inline Base.@propagate_inbounds partials(::Type{T}, x, i...) where T = partials(x, i...)
@inline Base.@propagate_inbounds partials(::Type{T}, d::Dual{T}, i...) where T = partials(d, i...)
@inline function partials(::Type{T}, d::Dual{S}, i...) where {T,S}
    if S ≺ T
        zero(d)
    else
        throw(DualMismatchError(T,S))
    end
end


@inline npartials(::Dual{T,V,N}) where {T,V,N} = N
@inline npartials(::Type{Dual{T,V,N}}) where {T,V,N} = N

@inline order(::Type{V}) where {V} = 0
@inline order(::Type{Dual{T,V,N}}) where {T,V,N} = 1 + order(V)

@inline valtype(::V) where {V} = V
@inline valtype(::Type{V}) where {V} = V
@inline valtype(::Dual{T,V,N}) where {T,V,N} = V
@inline valtype(::Type{Dual{T,V,N}}) where {T,V,N} = V

@inline tagtype(::V) where {V} = Nothing
@inline tagtype(::Type{V}) where {V} = Nothing
@inline tagtype(::Dual{T,V,N}) where {T,V,N} = T
@inline tagtype(::Type{Dual{T,V,N}}) where {T,V,N} = T

####################################
# N-ary Operation Definition Tools #
####################################

macro define_binary_dual_op(f, xy_body, x_body, y_body)
    FD = ForwardDiff2
    defs = quote
        @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}) where {Txy} = $xy_body
        @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}) where {Tx,Ty} = Ty ≺ Tx ? $x_body : $y_body
    end
    for R in AMBIGUOUS_TYPES
        expr = quote
            @inline $(f)(x::$FD.Dual{Tx}, y::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}) where {Ty} = $y_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

macro define_ternary_dual_op(f, xyz_body, xy_body, xz_body, yz_body, x_body, y_body, z_body)
    FD = ForwardDiff2
    defs = quote
        @inline $(f)(x::$FD.Dual{Txyz}, y::$FD.Dual{Txyz}, z::$FD.Dual{Txyz}) where {Txyz} = $xyz_body
        @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}, z::$FD.Dual{Tz}) where {Txy,Tz} = Tz ≺ Txy ? $xy_body : $z_body
        @inline $(f)(x::$FD.Dual{Txz}, y::$FD.Dual{Ty}, z::$FD.Dual{Txz}) where {Txz,Ty} = Ty ≺ Txz ? $xz_body : $y_body
        @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Tyz}, z::$FD.Dual{Tyz}) where {Tx,Tyz} = Tyz ≺ Tx ? $x_body  : $yz_body
        @inline function $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}, z::$FD.Dual{Tz}) where {Tx,Ty,Tz}
            if Tz ≺ Tx && Ty ≺ Tx
                $x_body
            elseif Tz ≺ Ty
                $y_body
            else
                $z_body
            end
        end
    end
    for R in AMBIGUOUS_TYPES
        expr = quote
            @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}, z::$R) where {Txy} = $xy_body
            @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}, z::$R)  where {Tx, Ty} = Ty ≺ Tx ? $x_body : $y_body
            @inline $(f)(x::$FD.Dual{Txz}, y::$R, z::$FD.Dual{Txz}) where {Txz} = $xz_body
            @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$FD.Dual{Tz}) where {Tx,Tz} = Tz ≺ Tx ? $x_body : $z_body
            @inline $(f)(x::$R, y::$FD.Dual{Tyz}, z::$FD.Dual{Tyz}) where {Tyz} = $yz_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$FD.Dual{Tz}) where {Ty,Tz} = Tz ≺ Ty ? $y_body : $z_body
        end
        append!(defs.args, expr.args)
        for Q in AMBIGUOUS_TYPES
            Q === R && continue
            expr = quote
                @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$Q) where {Tx} = $x_body
                @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$Q) where {Ty} = $y_body
                @inline $(f)(x::$R, y::$Q, z::$FD.Dual{Tz}) where {Tz} = $z_body
            end
            append!(defs.args, expr.args)
        end
        expr = quote
            @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$R) where {Ty} = $y_body
            @inline $(f)(x::$R, y::$R, z::$FD.Dual{Tz}) where {Tz} = $z_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

function unary_dual_definition(M, f)
    FD = ForwardDiff2
    Mf = M == :Base ? f : :($M.$f)
    work = qualified_cse!(quote
        val = $Mf(x)
        deriv = $(DiffRules.diffrule(M, f, :x))
    end)
    return quote
        @inline function $M.$f(d::$FD.Dual{T}) where T
            x = $FD.value(d)
            $work
            return $FD.Dual{T}(val, deriv * $FD.partials(d))
        end
    end
end

function binary_dual_definition(M, f)
    FD = ForwardDiff2
    dvx, dvy = DiffRules.diffrule(M, f, :vx, :vy)
    Mf = M == :Base ? f : :($M.$f)
    xy_work = qualified_cse!(quote
        val = $Mf(vx, vy)
        dvx = $dvx
        dvy = $dvy
    end)
    dvx, _ = DiffRules.diffrule(M, f, :vx, :y)
    x_work = qualified_cse!(quote
        val = $Mf(vx, y)
        dvx = $dvx
    end)
    _, dvy = DiffRules.diffrule(M, f, :x, :vy)
    y_work = qualified_cse!(quote
        val = $Mf(x, vy)
        dvy = $dvy
    end)
    expr = quote
        @define_binary_dual_op(
            $M.$f,
            begin
                vx, vy = $FD.value(x), $FD.value(y)
                $xy_work
                return $FD.Dual{Txy}(val, $FD._mul_partials($FD.partials(x), $FD.partials(y), dvx, dvy))
            end,
            begin
                vx = $FD.value(x)
                $x_work
                return $FD.Dual{Tx}(val, dvx * $FD.partials(x))
            end,
            begin
                vy = $FD.value(y)
                $y_work
                return $FD.Dual{Ty}(val, dvy * $FD.partials(y))
            end
        )
    end
    return expr
end

#####################
# Generic Functions #
#####################

Base.copy(d::Dual) = d

Base.eps(d::Dual) = eps(value(d))
Base.eps(::Type{D}) where {D<:Dual} = eps(valtype(D))

Base.rtoldefault(::Type{D}) where {D<:Dual} = Base.rtoldefault(valtype(D))

Base.floor(::Type{R}, d::Dual) where {R<:Real} = floor(R, value(d))
Base.floor(d::Dual) = floor(value(d))

Base.ceil(::Type{R}, d::Dual) where {R<:Real} = ceil(R, value(d))
Base.ceil(d::Dual) = ceil(value(d))

Base.trunc(::Type{R}, d::Dual) where {R<:Real} = trunc(R, value(d))
Base.trunc(d::Dual) = trunc(value(d))

Base.round(::Type{R}, d::Dual) where {R<:Real} = round(R, value(d))
Base.round(d::Dual) = round(value(d))

Base.hash(d::Dual) = hash(value(d))
Base.hash(d::Dual, hsh::UInt) = hash(value(d), hsh)

function Base.read(io::IO, ::Type{Dual{T,V,N}}) where {T,V,N}
    value = read(io, V)
    partials = read(io, Partials{N,V})
    return Dual{T,V,N}(value, partials)
end

function Base.write(io::IO, d::Dual)
    write(io, value(d))
    write(io, partials(d))
end

@inline Base.zero(d::Dual) = zero(typeof(d))
@inline Base.zero(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(zero(V), zero(Partials{N,V}))

@inline Base.one(d::Dual) = one(typeof(d))
@inline Base.one(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(one(V), zero(Partials{N,V}))

@inline Random.rand(rng::AbstractRNG, d::Dual) = rand(rng, value(d))
@inline Random.rand(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(V), zero(Partials{N,V}))
@inline Random.rand(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(rng, V), zero(Partials{N,V}))
@inline Random.randn(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(V), zero(Partials{N,V}))
@inline Random.randn(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(rng, V), zero(Partials{N,V}))
@inline Random.randexp(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(V), zero(Partials{N,V}))
@inline Random.randexp(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(rng, V), zero(Partials{N,V}))

# Predicates #
#------------#

isconstant(d::Dual) = iszero(partials(d))

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::Dual) = $(pred)(value(d))
end

for pred in BINARY_PREDICATES
    @eval begin
        @define_binary_dual_op(
            Base.$(pred),
            $(pred)(value(x), value(y)),
            $(pred)(value(x), y),
            $(pred)(x, value(y))
        )
    end
end

########################
# Promotion/Conversion #
########################

Base.@pure function Base.promote_rule(::Type{Dual{T1,V1,N1}},
                                      ::Type{Dual{T2,V2,N2}}) where {T1,V1,N1,T2,V2,N2}
    # V1 and V2 might themselves be Dual types
    if T2 ≺ T1
        Dual{T1,promote_type(V1,Dual{T2,V2,N2}),N1}
    else
        Dual{T2,promote_type(V2,Dual{T1,V1,N1}),N2}
    end
end

function Base.promote_rule(::Type{Dual{T,A,N}},
                           ::Type{Dual{T,B,N}}) where {T,A,B,N}
    return Dual{T,promote_type(A, B),N}
end

for R in (Irrational, Real, BigFloat, Bool)
    if isconcretetype(R) # issue #322
        @eval begin
            Base.promote_rule(::Type{$R}, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T,promote_type($R, V),N}
            Base.promote_rule(::Type{Dual{T,V,N}}, ::Type{$R}) where {T,V,N} = Dual{T,promote_type(V, $R),N}
        end
    else
        @eval begin
            Base.promote_rule(::Type{R}, ::Type{Dual{T,V,N}}) where {R<:$R,T,V,N} = Dual{T,promote_type(R, V),N}
            Base.promote_rule(::Type{Dual{T,V,N}}, ::Type{R}) where {T,V,N,R<:$R} = Dual{T,promote_type(V, R),N}
        end
    end
end

Base.convert(::Type{Dual{T,V,N}}, d::Dual{T}) where {T,V,N} = Dual{T}(convert(V, value(d)), convert(Partials{N,V}, partials(d)))
Base.convert(::Type{Dual{T,V,N}}, x) where {T,V,N} = Dual{T}(convert(V, x), zero(Partials{N,V}))
Base.convert(::Type{Dual{T,V,N}}, x::Number) where {T,V,N} = Dual{T}(convert(V, x), zero(Partials{N,V}))
Base.convert(::Type{D}, d::D) where {D<:Dual} = d

Base.float(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,promote_type(V, Float16),N}, d)
Base.AbstractFloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,promote_type(V, Float16),N}, d)

###################################
# General Mathematical Operations #
###################################

@inline Base.conj(d::Dual) = d

###################
# Pretty Printing #
###################

function Base.show(io::IO, d::Dual{T,V,N}) where {T,V,N}
    print(io, "Dual{$(repr(T))}(", value(d))
    for i in 1:N
        print(io, ",", partials(d, i))
    end
    print(io, ")")
end
