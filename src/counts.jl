mutable struct NGramCounter{T}
    n::Int
    bos::String
    eos::String
    counts::Dict{T,NGramCounter{T}}
    total::Int
end

NGramCounter(n; bos=BOS, eos=EOS) = NGramCounter{Any}(n, bos=BOS, eos=EOS)

function NGramCounter{T}(n::Int; bos=BOS, eos=EOS) where T
    @assert 1 <= n
    counts = Dict{T,NGramCounter{T}}()
    NGramCounter(n, bos, eos, counts, 0)
end

total(grams::NGramCounter) = grams.total
total(x::Number) = x

function add_ngrams!(grams::NGramCounter{T}, tokens) where T
    for gram in ngrams(grams.n, tokens)
        c = grams
        for token in gram
            c.total += 1
            c = get!(c.counts, token) do
                n, bos, eos = grams.n - 1, grams.bos, grams.eos
                counts = Dict{T,NGramCounter{T}}()
                NGramCounter{T}(n, bos, eos, counts, 0)
            end
        end
        c.total += 1
    end
    return grams
end

count(counts::NGramCounter{T}, token::T) where T = total(get(counts.counts, token, 0))
function count(counts::NGramCounter, tokens)
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
