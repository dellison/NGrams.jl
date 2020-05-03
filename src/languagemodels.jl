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
    NGrams.fit!(lm::LanguageModel, tokens)

Train the language model by observing a sequence of tokens.
"""
function fit!(m::LanguageModel{N,T,P}, tokens) where {N,T,P}
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


function generate(m::LanguageModel, num_words=1, text_seed=String[])
    # todo random seed for repro
    @assert num_words >= 1
    output = copy(text_seed)
    while length(output) < num_words - length(text_seed)
        push!(output, sample(submodel(m, _hist(m, output)), keys(m.seq)))
    end
    return output
end

sample(lm::LanguageModel, a...; k...) = sample(Random.GLOBAL_RNG, lm, a...; k...)
function sample(rng::AbstractRNG, lm::LanguageModel, vocabulary=keys(lm.seq))
    t = rand(rng)# * total(lm)
    words = keys(lm.seq)
    n = length(words)
    i = 1
    cw = 0
    for word in words
        cw += prob(lm, word)
        if cw >= t
            return word
        end
    end
    return lm.bos # last(words) # ??
end

# relevant history
function _hist(m::LanguageModel, xs)
    n = gram_size(m) - 2
    if n >= length(xs) # need to pad
        return gram(vcat(repeat([m.bos], n - length(xs) + 1), xs))
    else
        return gram(xs[end-n:end])
    end
end
