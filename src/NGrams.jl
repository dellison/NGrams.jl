module NGrams

using Random

const BOS = "*BOS*"
const EOS = "*EOS*"

export ngrams, unigrams, bigrams, trigrams

include("grams.jl")
include("counts.jl")
include("probability.jl")
include("languagemodels.jl")

end # module
