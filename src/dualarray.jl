using ForwardDiff: Dual, value, partials
using MacroTools: @forward

struct DualArray{T,E,N,I,D<:AbstractArray{E,N}} <: AbstractArray{E,N}
    data::D
end
DualArray{T,I}(a::AbstractArray{E,N}) where {T,I,E,N} = DualArray{T,E,N,I,typeof(a)}(a)
DualArray(a::AbstractArray) = DualArray{Nothing,2}(a)

# TODO: broadcast
@forward DualArray.data Base.length, Base.ndims, Base.size, Base.axes, Base.stride, Base.strides
Base.eachindex(d::DualArray) = eachindex(@view d.data[:, 1])
Base.similar(d::T) where {T<:DualArray} = T(similar(d.data))
Base.eltype(d::DualArray{T,E,N,I,D}) where {T,E,N,I,D} = Dual{T,E,N}

tagname(d::DualArray{T}) where {T} = T
ncols(d::DualArray{T,E,N,I}) where {T,E,N,I} = I

Base.@propagate_inbounds function Base.getindex(d::DualArray, i...)
    val = d.data[i..., 1]
    parts = ntuple(j->d.data[i..., j+1], Val(ncols(d)-1))
    return Dual(val, parts)
end

Base.@propagate_inbounds function Base.setindex!(d::DualArray, dual::Dual, i...)
    d.data[i..., 1] = value(dual)
    d.data[i..., 2:end] .= partials(dual)
    return dual
end
