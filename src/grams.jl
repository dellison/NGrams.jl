const NGram{N, T} = NTuple{N,T}
const Unigram{T}  = NGram{1,T}
const Bigram{T}   = NGram{2,T}
const Trigram{T}  = NGram{3,T}

gram(x)                 = (x,)
gram(x::AbstractVector) = Tuple(x)
gram(x::Tuple)          = x

pad(sequence, n, bos, eos) = vcat(bos === nothing ? [] : fill(bos, n),
                                  sequence,
                                  eos === nothing ? [] : fill(eos, n))

"""
    ngrams(tokens, n; add_bos=true, bos=BOS, add_eos=true, eos=EOS) 

Iterate over a sequence of n-grams (of length `n`) from `tokens`.
"""
ngrams(xs::AbstractVector, n::Int; bos=BOS, eos=EOS) =
    NGramIterator(n, pad(xs, n-1, bos, eos))
ngrams(xs, n::Int; kw...) = ngrams(collect(xs), n; kw...)
ngrams(n::Int, xs; kw...) = ngrams(xs, n; kw...)

"""
    unigrams(tokens)

Return a sequence of unigrams from `tokens`.
"""
unigrams(xs) = map(tuple, xs)

"""
    bigrams(xs; bos="*BOS*", eos="*EOS*")

Iterate over a sequence of bigrams from `xs`.
"""
bigrams(xs; kw...) = ngrams(xs, 2; kw...)

"""
    trigrams(xs; bos="*BOS*", eos="*EOS*")

Iterate over a sequence of trigrams from `xs`.
"""
trigrams(xs; ks...) = ngrams(xs, 3; ks...)

struct NGramIterator{N,T}
    n::Int
    tokens::T
end
NGramIterator(N::Int, xs) = NGramIterator{N,typeof(xs)}(N,xs)

Base.collect(ngrams::NGramIterator) = [gram for gram in ngrams]
Base.eltype(::NGramIterator{N,T}) where {N,T} = NGram{N,eltype(T)}

function Base.iterate(ngrams::NGramIterator, state=1)
    state > length(ngrams.tokens) - ngrams.n + 1 && return nothing
    return tuple(view(ngrams.tokens, state:state+ngrams.n-1)...), state + 1
end

Base.length(ngrams::NGramIterator) = length(ngrams.tokens) - ngrams.n + 1

Base.IteratorSize(::Type{<:NGramIterator{N,T}}) where {N,T} = Base.HasLength()
Base.IteratorEltype(::Type{<:NGramIterator{N,T}}) where {N,T} = NGram{N,eltype(T)}
