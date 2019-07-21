const NGram{N, T} = NTuple{N, T}
const Unigram{T}  = NGram{1, T}
const Bigram{T}   = NGram{2, T}
const Trigram{T}  = NGram{3, T}

"""
    ngrams(n, tokens; add_bos=true, bos=BOS, add_eos=true, eos=EOS) 

Return a sequence of n-grams (of length `n`) from `tokens`.
"""
function ngrams(n::Int, tokens; add_bos=true, bos=BOS, add_eos=true, eos=EOS)
    ts = add_tags(tokens, n, add_bos=add_bos, bos=bos, add_eos=add_eos, eos=eos)
    return NGramIterator{n,typeof(ts)}(n, ts)
end

ngrams(tokens, n::Int; kwargs...) = ngrams(n, tokens; kwargs...)

"""
    unigrams(tokens)

Return a sequence of unigrams from `tokens`.
"""
unigrams(tokens; kwargs...) = ngrams(1, tokens; kwargs...)

"""
    bigrams(tokens)

Return a sequence of bigrams from `tokens`.
"""
bigrams(tokens; kwargs...)  = ngrams(2, tokens; kwargs...)

"""
    trigrams(tokens)

Return a sequence of trigrams from `tokens`.
"""
trigrams(tokens; kwargs...) = ngrams(3, tokens; kwargs...)

_add_bos(tokens, n, bos=BOS) = vcat(fill(bos, n), tokens)
_add_eos(tokens, n, eos=EOS) = vcat(tokens, fill(eos, n))

add_tags(tokens, n; add_bos=true, bos=BOS, add_eos=true, eos=EOS) =
    tokens |>
    (ts -> add_bos ? _add_bos(ts, isone(n) ? n : n-1, bos) : ts) |>
    (ts -> add_eos ? _add_eos(ts, 1, eos) : ts)

struct NGramIterator{N,T}
    n::Int
    tokens::T
end

collect(ngrams::NGramIterator) = [gram for gram in ngrams]

eltype(::NGramIterator{N,T}) where {N,T} = NGram{N,T}

function iterate(ngrams::NGramIterator, state=1)
    state > length(ngrams.tokens) - ngrams.n + 1 && return nothing
    return tuple(view(ngrams.tokens, state:state+ngrams.n-1)...), state + 1
end

length(ngrams::NGramIterator) = length(ngrams.tokens) - ngrams.n + 1
