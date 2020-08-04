mutable struct NGramCounts{N,T}
    counts::Dict{T,NGramCounts}
    total::Int
end
NGramCounts{N,T}() where {N,T} = NGramCounts{N,T}(Dict{T,NGramCounts{N-1,T}}(), 0)

Base.count(c::NGramCounts, x) = total(submodel(c, x))
Base.keys(c::NGramCounts) = keys(c.counts)
Base.length(c::NGramCounts) = length(c.counts)

function Base.merge(a::NGramCounts{N,T}, b::NGramCounts{N,T}) where {N,T}
    c = deepcopy(a)
    for k in keys(b)
        c.counts[k] = merge(submodel(c, k), submodel(b, k))
    end
    c.total += b.total
    return c
end

total(c::NGramCounts) = c.total

gram_size(c::NGramCounts{N,T}) where {N,T} = N

function inc!(c::NGramCounts, gram)
    for x in gram
        c.total += 1
        c = submodel!(c, x)
    end
    c.total += 1
end

order(c::NGramCounts{N,T}) where {N,T} = N-1

submodel(c::NGramCounts{N,T}, x) where {N,T} =
    get(() -> NGramCounts{N-1,T}(), c.counts, x)

function submodel(c::NGramCounts, g::NGram)
    for x in g
        c = submodel(c, x)
    end
    return c
end

submodel!(c::NGramCounts{N,T}, x) where {N,T} =
    get!(() -> NGramCounts{N-1,T}(), c.counts, x)

function submodel!(c::NGramCounts, g::NGram)
    for x in g
        c = submodel!(c, x)
    end
    return c
end
