struct LanguageModel{T}
    counts::NGramCounter
    model::T
end

"""
    LanguageModel(n::Int, model; bos="*BOS*", eos="*EOS*")

Create a language model with n-gram length `n` and probability model `model`.
"""
LanguageModel(n::Int, model; bos=BOS, eos=EOS) =
    LanguageModel(NGramCounter(n; bos=bos, eos=eos), model)

count(lm::LanguageModel, token) = count(lm.counts, [token])
count(lm::LanguageModel, tokens::AbstractArray) = count(lm.counts, tokens)
p(lm, history, token) = p(lm.model, lm.counts, history, token)

train!(lm::LanguageModel, sentence) = add_ngrams!(lm.counts, sentence)

function train_lm(corpus, n, model; bos=BOS, eos=EOS)
    lm = LanguageModel(n, model; bos=bos, eos=eos)
    for sentence in corpus
        train!(lm, sentence)
    end
    return lm
end

include("models/mle.jl")
