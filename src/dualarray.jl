using StaticArrays: SVector, StaticArray

Base.:+(x::StaticArray) = x

struct DualArray{T,N,V<:AbstractArray,D<:AbstractArray,E,M} <: AbstractArray{E,M}
    data::V
    partials::D
    # ndims(data) + 1 == ndims(partials)
    function DualArray{T}(v::AbstractArray, p) where {T}
        N = size(p, ndims(p))
        # only use SVector for now
        V, D = typeof(v), typeof(p)
        X = SVector{N,eltype(p)}
        E = Dual{T,eltype(v),X}
        M = ndims(v)
        return new{T,N,V,D,E,M}(v, p)
    end
end

DualArray(a::AbstractArray, b::AbstractArray) = DualArray{typeof(dualtag())}(a, b)
data(d::DualArray) = d.data
allpartials(d::DualArray) = d.partials

npartials(d::DualArray) = npartials(typeof(d))
npartials(::Type{<:DualArray{T,N}}) where {T,N} = N

###
### Printing
###

function Base.print_array(io::IO, da::DualArray)
    _dispsize((w, h),) = (w, div(max(h-npartials(da)-1,0), (npartials(da) + 1)))
    sz = :displaysize => _dispsize(get(io, :displaysize, displaysize(io)))

    Base.printstyled(io, "Primals:\n", bold=false, color=2)
    prev_params = io isa IOContext ? io.dict : ()
    ioc = IOContext(io, prev_params..., sz)
    Base.print_array(ioc, data(da))
    Base.println(io)
    Base.printstyled(io,"Partials:\n", bold=false, color=3)
    Base.print_array(ioc, allpartials(da))
    return nothing
end

###
### Array interface
###

#droplast(d::Tuple) = d |> reverse |> Base.tail |> reverse
Base.size(d::DualArray) = size(data(d))
Base.IndexStyle(d::DualArray) = Base.IndexStyle(data(d))
Base.similar(d::DualArray{T}, ::Type{S}, dims::Dims) where {T, S} = DualArray{T}(similar(data(d)), similar(allpartials(d)))
Base.eachindex(d::DualArray) = eachindex(data(d))

#Base.@propagate_inbounds _slice(A, i...) = @view A[i..., :]
#Base.@propagate_inbounds _slice(A::StaticArray, i...) = A[i..., :]

partial_type(::Type{Dual{T,V,P}}) where {T,V,P} = P
Base.@propagate_inbounds function Base.getindex(d::DualArray{T}, i::Int...) where {T}
    ps = allpartials(d)
    P = partial_type(eltype(d))
    partials_tuple = ntuple(j->ps[i..., j], Val(npartials(d)))
    return Dual{T}(data(d)[i...], P(partials_tuple))
end

Base.@propagate_inbounds function Base.setindex!(d::DualArray{T}, dual::Dual{T}, i::Int...) where {T}
    data(d)[i...] = value(dual)
    allpartials(d)[i..., :] .= partials(dual)
    return dual
end
