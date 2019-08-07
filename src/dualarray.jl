using ForwardDiff
import ForwardDiff: Dual, value, partials, npartials, tagtype, valtype
using MacroTools: @forward

# TODO: integrate Dual with ChainRules?

# TODO: Tagging?

# TODO: better printing
struct DualArray{T,E,M,D<:AbstractArray,I} <: AbstractArray{E,M}
    data::D
    DualArray{T,I}(a::AbstractArray{E,N}) where {T,I,E,N} = new{T,E,N-1,typeof(a),I}(a)
end
DualArray(a::AbstractArray) = DualArray{Nothing,1}(a)
ForwardDiff.npartials(d::DualArray{T,E,M,D,I}) where {T,E,M,D,I} = I
ForwardDiff.tagtype(::Type{<:DualArray{T}}) where {T} = T
ForwardDiff.tagtype(::T) where {T<:DualArray} = tagtype(T)
data(d::DualArray) = d.data

###
### Array interface
###

Base.eltype(d::DualArray{T,E}) where {T,E} = Dual{T,E,npartials(d)}
droplast(d::Tuple) = d |> reverse |> Base.tail |> reverse
Base.size(d::DualArray) = size(data(d)) |> droplast
Base.size(d::DualArray, i) = i <= ndims(d) ? size(data(d), i) : 1
Base.IndexStyle(d::DualArray) = Base.IndexStyle(data(d))
Base.strides(d::DualArray) = strides(data(d)) |> droplast
Base.similar(d::DualArray{T}, ::Type{S}, dims::Dims) where {T,S} = DualArray{T,npartials(d)}(similar(data(d), S, (dims..., npartials(d)+1)))

###
### Broadcast interface
###

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle

struct DualStyle{M,T,I,D} <: AbstractArrayStyle{M}
end

Base.Broadcast.BroadcastStyle(::Type{<:DualArray{T,E,M,D,I}}) where {T,E,M,D,I} = DualStyle{M,T,I,typeof(Base.Broadcast.BroadcastStyle(D))}()
function Base.similar(bc::Broadcasted{<:DualStyle{M,T,I,D}}, ::Type{E}) where {M,T,I,D,E}
    if E <: Dual
        V = valtype(E)
        arr = DualArray{T,I}(similar(Array{V}, (axes(bc)..., Base.OneTo(I+1)))) # TODO: work with arbitrary array types. Maybe use `ArrayInterface.jl`?
    else
        bc′ = convert(Broadcasted{D}, bc)
        arr = Base.similar(bc′, E)
    end
    return arr
end
Base.BroadcastStyle(::DualStyle{M,T,I,D}, ::DualStyle{M,T,I,V}) where {M,T,I,D,V} = DualStyle{M,T,I,typeof(Base.BroadcastStyle(D(), V()))}()
Base.BroadcastStyle(::DualStyle{M,T,I,D}, B::BroadcastStyle) where {M,T,I,D} = DualStyle{M,T,I,typeof(Base.BroadcastStyle(D(), B))}()
Base.BroadcastStyle(::DualStyle{M,T,I,D}, B::DefaultArrayStyle) where {M,T,I,D} = DualStyle{M,T,I,typeof(Base.BroadcastStyle(D(), B))}()

function ForwardDiff.value(d::DualArray)
    n = ndims(d)
    return @view d[ntuple(_ -> Colon(), Val(n))..., 1]
end

function ForwardDiff.partials(d::DualArray)
    n = ndims(d)
    return @view d[ntuple(_ -> Colon(), Val(n))..., 2:end]
end

Base.eachindex(d::DualArray) = eachindex(@view data(d)[:, 1])

Base.@propagate_inbounds function Base.getindex(d::DualArray, i...)
    dd = data(d)
    val = dd[i..., 1]
    parts = ntuple(j->dd[i..., j+1], Val(npartials(d)))
    return Dual(val, parts)
end

# TODO: linear indexing
Base.@propagate_inbounds function Base.setindex!(d::DualArray, dual::Dual, i...)
    dd = data(d)
    dd[i..., 1] = value(dual)
    dd[i..., 2:end] .= partials(dual)
    return dual
end

Base.@propagate_inbounds function Base.setindex!(d::DualArray, x::Number, i...)
    dd = data(d)
    dd[i..., 1] = x
    dd[i..., 2:end] .= zero(eltype(dd))
    return x
end
