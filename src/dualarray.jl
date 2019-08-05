struct DualArray{T,E,N,D<:AbstractArray{E,N}} <: AbstractArray{E,N}
    data::D
    DualArray{T}(a::AbstractArray{E,N}) where {T,E,N} = new{T,E,N,typeof(a)}(a)
end

DualArray(a::AbstractArray) = DualArray{Nothing}(a)

tagname(d::DualArray{T}) where T = T
