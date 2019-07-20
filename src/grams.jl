const NGram{N, T} = NTuple{N, T}
const Unigram{T}  = NGram{1, T}
const Bigram{T}   = NGram{2, T}
const Trigram{T}  = NGram{3, T}

function ngrams(n::Int, tokens; add_bos=true, bos=BOS, add_eos=true, eos=EOS)
    tokens = tokens |>
        (ts -> add_bos ? _add_bos(ts, isone(n) ? n : n-1, bos) : ts) |>
        (ts -> add_eos ? _add_eos(ts, 1, eos) : ts)
    return NGramIterator{n,typeof(tokens)}(n, tokens)
end

ngrams(tokens, n::Int; kwargs...) = ngrams(n, tokens; kwargs...)

unigrams(tokens; kwargs...) = ngrams(1, tokens; kwargs...)
bigrams(tokens; kwargs...)  = ngrams(2, tokens; kwargs...)
trigrams(tokens; kwargs...) = ngrams(3, tokens; kwargs...)

_add_bos(tokens, n, bos=BOS) = vcat(fill(bos, n), tokens)
_add_eos(tokens, n, eos=EOS) = vcat(tokens, fill(eos, n))


struct NGramIterator{N,T}
    n::Int
    tokens::T
end

collect(ngrams::NGramIterator) = [gram for gram in ngrams]

eltype(::Type{NGramIterator{N,T}}) where {N,T} = NGram{N,T}
eltype(::NGramIterator{N,T}) where {N,T} = NGram{N,T}

function iterate(ngrams::NGramIterator, state=1)
    state > length(ngrams.tokens) - ngrams.n + 1 && return nothing
    return tuple(view(ngrams.tokens, state:state+ngrams.n-1)...), state + 1
end

length(ngrams::NGramIterator) = length(ngrams.tokens) - ngrams.n + 1
