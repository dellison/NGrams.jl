mutable struct LanguageModel{N,T,P}
    bos::T
    eos::T
    seq::NGramCounts{N,T}
    estimator::P
end

"""
    LanguageModel(N; bos, eos, estimator=NGrams.MLE())

Create an `N`-gram language model, estimating probabilities with `estimator`.
"""
LanguageModel(N::Int; bos=BOS, eos=EOS, estimator=MLE()) =
    LanguageModel(N, bos, eos, estimator)
LanguageModel(N::Int, bos::T, eos::T, estimator=MLE()) where T =
    LanguageModel{N,T,typeof(estimator)}(bos, eos, NGramCounts{N,T}(), estimator)
LanguageModel(N::Int, p::ProbabilityEstimator; bos=BOS, eos=EOS) =
    LanguageModel{N,Any,typeof(p)}(bos, eos, NGramCounts{N,Any}(), p)

Base.count(m::LanguageModel, v::AbstractVector) = count(m, tuple(v...))
Base.count(m::LanguageModel{N,T,P}, x::T) where {N,T,P} = total(submodel(m.seq, x))
function Base.count(m::LanguageModel, gram::Tuple)
    for x in gram
        m = submodel(m, x)
    end
    return total(m)
end

"""
    NGrams.observe!(lm::LanguageModel, tokens)

Train the language model by observing a sequence of tokens.
"""
function observe!(m::LanguageModel{N,T,P}, tokens) where {N,T,P}
    for gram in ngrams(tokens, N; bos=m.bos, eos=m.eos)
        inc!(m.seq, gram)
    end
end
    
prob(m::LanguageModel, a...) = prob(m.estimator, m.seq, gram.(a)...)

submodel(m::LanguageModel, x) =
    LanguageModel(m.bos, m.eos, submodel(m.seq, x), m.estimator)

for f in (:gram_size, :order, :total)
    @eval $f(m::LanguageModel, a...; kw...) = $f(m.seq, a...; kw...)
end

