module NGrams

const BOS = "*BOS*"
const EOS = "*EOS*"

import Base: collect, count, eltype, iterate, length

export NGram, Unigram, Bigram, Trigram
export ngrams, unigrams, bigrams, trigrams

export LanguageModel, train_lm
export MLE, Laplace, Add1, AddK, LinearInterpolation, AbsoluteDiscounting

include("grams.jl")
include("counts.jl")
include("languagemodels.jl")

end # module
