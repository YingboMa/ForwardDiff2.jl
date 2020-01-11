using StaticArrays: SVector

partial_type(::Dual{T,V,P}) where {T,V,P} = P

struct DualArray{T,E,M,V<:AbstractArray,D<:AbstractArray} <: AbstractArray{E,M}
    data::V
    partials::D
    function DualArray{T}(v::AbstractArray{E,N}, p::P) where {T,E,N,P<:AbstractArray}
        # TODO: non-allocating X?
        X = typeof(similar(p, Base.tail(ntuple(_->0, Val(ndims(P))))))
        # we need the eltype of `DualArray` to be `Dual{T,E,X}` as opposed to
        # some kind of `view`, because we can convert `SubArray` to `Array` but
        # not vise a versa.
        #
        # We need that to differentiate through the following code
        # `(foo(x::AbstractArray{T})::T) where {T} = x[1]`
        return new{T,Dual{T,E,X},N,typeof(v),typeof(p)}(v, p)
    end
end

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

DualArray(a::AbstractArray, b::AbstractArray) = DualArray{typeof(dualtag())}(a, b)
npartials(d::DualArray) = size(d.partials, ndims(d.partials))
data(d::DualArray) = d.data
allpartials(d::DualArray) = d.partials

###
### Array interface
###

#droplast(d::Tuple) = d |> reverse |> Base.tail |> reverse
Base.size(d::DualArray) = size(data(d))
Base.IndexStyle(d::DualArray) = Base.IndexStyle(data(d))
Base.similar(d::DualArray{T}, ::Type{S}, dims::Dims) where {T, S} = DualArray{T}(similar(data(d)), similar(allpartials(d)))
Base.eachindex(d::DualArray) = eachindex(data(d))

using StaticArrays

Base.@propagate_inbounds _slice(A, i...) = @view A[i..., :]
Base.@propagate_inbounds _slice(A::StaticArray, i...) = A[i..., :]

Base.@propagate_inbounds function Base.getindex(d::DualArray{T}, i::Int...) where {T}
    return Dual{T}(data(d)[i...], _slice(allpartials(d), i...))
end

Base.@propagate_inbounds function Base.setindex!(d::DualArray{T}, dual::Dual{T}, i::Int...) where {T}
    data(d)[i...] = value(dual)
    allpartials(d)[i..., :] .= partials(dual)
    return dual
end
