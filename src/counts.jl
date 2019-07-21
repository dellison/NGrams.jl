mutable struct TokenCounter{T}
    counts::Dict{T,Int}
    total::Int
end

TokenCounter() = TokenCounter{Any}(Dict{Any,Int}(), 0)

count(counts::TokenCounter, token) = get(counts.counts, token, 0)
total(counts::TokenCounter) = counts.total

observed_ratio(counts::TokenCounter, token) = (count(counts, token), total(counts))

function add_tokens!(counts::TokenCounter, tokens)
    for token in tokens
        counts.counts[token] = get(counts.counts, token, 0) + 1
    end
    counts.total += length(tokens)
    return counts
end

mutable struct NGramCounter{T}
    n::Int
    counts::Dict{T,NGramCounter{T}}
    total::Int
end

NGramCounter(n) = NGramCounter{Any}(n)

function NGramCounter{T}(n::Int) where T
    @assert 1 <= n
    counts = Dict{T,NGramCounter{T}}()
    NGramCounter(n, counts, 0)
end

total(grams::NGramCounter) = grams.total
total(x::Number) = x

function add_ngrams!(counts::NGramCounter{T}, grams) where T
    for gram in grams
        c = counts
        for token in gram
            c.total += 1
            c = get!(c.counts, token) do
                n, gcounts = c.n - 1, Dict{T,NGramCounter{T}}()
                NGramCounter{T}(n, gcounts, 0)
            end
        end
        c.total += 1
    end
    return grams
end

count(counts::NGramCounter{T}, token::T) where T = total(get(counts.counts, token, 0))
function count(grams::NGramCounter, tokens::AbstractArray)
    c = grams
    for token in tokens
        c = get(c.counts, token, nothing)
        c === nothing && (return 0)
    end
    return total(c)
end

function observed_ratio(grams, history, token)
    c = grams
    for h in history
        c = get(c.counts, h, nothing)
        c === nothing && (return (0, 0))
    end
    ct, t = (count(c, token), total(c))
    return (count(c, token), total(c))
end
