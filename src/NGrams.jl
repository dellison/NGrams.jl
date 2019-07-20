module NGrams

const BOS = "*BOS*"
const EOS = "*EOS*"

import Base: iterate, length

export NGram, Unigram, Bigram, Trigram
export ngrams, unigrams, bigrams, trigrams

include("grams.jl")

end # module
