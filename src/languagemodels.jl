struct LanguageModel{T}
    bos::T
    eos::T
    counts::NGramCounter
    tokens::TokenCounter
    model
end

"""
    LanguageModel(n::Int, model; bos="*BOS*", eos="*EOS*")

Create a language model with n-gram length `n` and probability model `model`.
"""
LanguageModel(n::Int, model; bos=BOS, eos=EOS) =
    LanguageModel(bos, eos, NGramCounter(n), TokenCounter(), model)

# count(lm::LanguageModel, token) = count(lm.counts, [token])
count(lm::LanguageModel, token) = count(lm.tokens, token)
count(lm::LanguageModel, tokens::AbstractArray) = count(lm.counts, tokens)
p(lm, history, token) = p(lm.model, lm.counts, history, token)
p(lm, token) = p(lm.model, lm.tokens, token)

function train!(lm::LanguageModel, sentence)
    n, bos, eos = lm.counts.n, lm.bos, lm.eos
    tokens = add_tags(sentence, n, bos=bos, eos=eos)
    add_tokens!(lm.tokens, tokens)
    add_ngrams!(lm.counts, ngrams(n, tokens, add_bos=false, add_eos=false))
    return lm
end

"""
    train_lm(corpus, n, model; bos=BOS, eos=EOS)

Train a language model.
"""
function train_lm(corpus, n, model; bos=BOS, eos=EOS)
    lm = LanguageModel(n, model; bos=bos, eos=eos)
    for sentence in corpus
        train!(lm, sentence)
    end
    return lm
end

include("models/mle.jl")
include("models/laplace.jl")
include("models/addk.jl")
