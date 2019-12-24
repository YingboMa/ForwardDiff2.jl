using StaticArrays: SVector
# TODO: Tagging?
# TODO: Integrate better with SVector. Maybe even use SIMD.jl?

struct DualArray{T,E,M,D<:AbstractArray,I} <: AbstractArray{E,M}
    data::D
    DualArray{T,I}(a::AbstractArray{E,N}) where {T,I,E,N} = new{T,E,N-1,typeof(a),I}(a)
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
    Base.print_array(ioc, value(da))
    Base.println(io)
    for i=1:npartials(da)
        Base.printstyled(io,"Partials($i):\n", bold=false, color=3)
        Base.print_array(ioc, partials.(da, i))
        i !== npartials(da) && Base.println(io)
    end
    return nothing
end

DualArray(a::AbstractArray) = DualArray{Nothing,size(a, ndims(a))-1}(a)
npartials(d::DualArray{T,E,M,D,I}) where {T,E,M,D,I} = I
tagtype(::Type{<:DualArray{T}}) where {T} = T
tagtype(::T) where {T<:DualArray} = tagtype(T)
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

function value(d::DualArray)
    n = ndims(d)
    dd = data(d)
    return @view dd[ntuple(_ -> Colon(), Val(n))..., 1]
end

function partials(d::DualArray)
    n = ndims(d)
    dd = data(d)
    return @view dd[ntuple(_ -> Colon(), Val(n))..., 2:end]
end

Base.eachindex(d::DualArray) = eachindex(@view data(d)[:, 1])

Base.@propagate_inbounds function Base.getindex(d::DualArray, i::Int...)
    dd = data(d)
    # TODO: do something different if dd is not Linear index style
    ii = LinearIndices(size(d))[i...]
    val   = dd[ii]
    slice_len = length(d)
    parts = ntuple(j->dd[j * slice_len + ii], Val(npartials(d)))
    return Dual(val, SVector(parts))
end

Base.@propagate_inbounds function Base.setindex!(d::DualArray, dual, i::Int...)
    dd = data(d)
    ii = LinearIndices(size(d))[i...]
    dd[ii] = value(dual)

    slice_len = length(d)
    for j = 1:npartials(d)
        dd[j * slice_len + ii] = partials(dual, j)
    end
    return dual
end
