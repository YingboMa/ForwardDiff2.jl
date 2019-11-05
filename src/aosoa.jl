using SIMD
import Base.Cartesian: @ntuple
const intlog2 = Core.Intrinsics.cttz_int

const Bucket{NFields, NElems} = NTuple{NFields, Vec{NElems}}

Base.@pure @inline function Base.eltype(T::Type{<:Bucket})
    Tuple{map(eltype, T.parameters)...}
end

struct AoSoA{T, Storage <: Bucket} <: AbstractVector{T}
    storage::Vector{Storage}
    function AoSoA(storage)
        S = eltype(storage)
        new{eltype(S), S}(storage)
    end
end

bucketwidth(::AoSoA{<:Any, <:Bucket{<:Any, NElems}}) where {NElems} = NElems

@generated function makebuckets!(arr, ::Val{bucketwidth}, columns::Vararg{<:Any, N}) where {bucketwidth, N}
    quote
        for k in 0:length(arr)-1
            arr[k+1] = @ntuple $N j ->Vec((@ntuple $bucketwidth i->columns[j][k * $bucketwidth + i]))
        end
        return arr
    end
end

function makeAoSoA(::Val{bucketwidth}, columns...) where {bucketwidth}
    len = length(first(columns))
    exp = intlog2(UInt(bucketwidth)) 
    @assert 2^exp == bucketwidth
    @assert all(x->length(x) == len, columns)
    @assert len % bucketwidth == 0
    BucketT = Tuple{map(T->Vec{bucketwidth, T}, eltype.(columns))...}
    AoSoA(makebuckets!(Vector{BucketT}(undef, div(len, bucketwidth)), Val{bucketwidth}(), columns...))
end

Base.size(a::AoSoA) = (length(a.storage) * bucketwidth(a),)
@inline function idx2bucketloc(a, i)
    w = UInt(bucketwidth(a))
    nshift = intlog2(w)
    ci = i - 1
    bucketidx = ci >> nshift
    j = ci & (w - 1)
    return bucketidx+1, j+1
end

Base.@propagate_inbounds function Base.getindex(a::AoSoA, i::Integer)
    bucketidx, j = idx2bucketloc(a, i)
    map(arr->arr[j], a.storage[bucketidx])
end

Base.@propagate_inbounds function Base.setindex!(a::AoSoA, vs, i::Integer)
    bucketidx, j = idx2bucketloc(a, i)
    newbucket = map((arr, v)->Base.setindex(arr, v, j), a.storage[bucketidx], vs)
    a.storage[bucketidx] = newbucket
    vs
end

#=

using SIMD

function sumfields(arr)
    T = Vec{4, Float64}
    T2 = Vec{bucketwidth(arr), Float64}
    init = zero(T)
    @inbounds for i in eachindex(arr.storage)
        v1 = T2(arr.storage[i][1])
        v2 = T2(arr.storage[i][2])
        v3 = T2(arr.storage[i][3])
        v4 = T2(arr.storage[i][4])
        init += T((sum(v1), sum(v2), sum(v3), sum(v4)))
    end
    #mapreduce(sumbucket, (x,y)->map(+, x, y), arr.storage, init=map(zero, first(arr)))
    return init
end
=#
