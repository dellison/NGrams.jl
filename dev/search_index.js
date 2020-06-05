var documenterSearchIndex = {"docs":
[{"location":"#NGrams.jl-1","page":"Home","title":"NGrams.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"NGrams.jl is a Julia package for working with n-gram models of natural language.","category":"page"},{"location":"languagemodeling/#Language-Models-1","page":"Language Modeling","title":"Language Models","text":"","category":"section"},{"location":"languagemodeling/#","page":"Language Modeling","title":"Language Modeling","text":"NGrams.LanguageModel","category":"page"},{"location":"languagemodeling/#NGrams.LanguageModel","page":"Language Modeling","title":"NGrams.LanguageModel","text":"LanguageModel(N; bos, eos, estimator=NGrams.MLE())\n\nCreate an N-gram language model, estimating probabilities with estimator.\n\n\n\n\n\n","category":"type"},{"location":"languagemodeling/#Training-1","page":"Language Modeling","title":"Training","text":"","category":"section"},{"location":"languagemodeling/#","page":"Language Modeling","title":"Language Modeling","text":"NGrams.fit!","category":"page"},{"location":"languagemodeling/#NGrams.fit!","page":"Language Modeling","title":"NGrams.fit!","text":"NGrams.fit!(lm::LanguageModel, tokens)\n\nTrain the language model by observing a sequence of tokens.\n\n\n\n\n\n","category":"function"},{"location":"languagemodeling/#Probability-and-Smoothing-1","page":"Language Modeling","title":"Probability and Smoothing","text":"","category":"section"},{"location":"languagemodeling/#","page":"Language Modeling","title":"Language Modeling","text":"NGrams.MLE\nNGrams.AddK\nNGrams.Laplace\nNGrams.LinearInterpolation\nNGrams.AbsoluteDiscounting","category":"page"},{"location":"languagemodeling/#NGrams.MLE","page":"Language Modeling","title":"NGrams.MLE","text":"NGrams.MLE()\n\nMaximum Likelihood Estimation for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"languagemodeling/#NGrams.AddK","page":"Language Modeling","title":"NGrams.AddK","text":"NGrams.AddK(k::Number)\n\nAdd-k probability smoothing for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"languagemodeling/#NGrams.Laplace","page":"Language Modeling","title":"NGrams.Laplace","text":"NGrams.Laplace()\n\nLaplace (add-1) smoothing for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"languagemodeling/#NGrams.LinearInterpolation","page":"Language Modeling","title":"NGrams.LinearInterpolation","text":"LinearInterpolation(λ)\n\nLinear interpolation for probability smoothing in n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"languagemodeling/#NGrams.AbsoluteDiscounting","page":"Language Modeling","title":"NGrams.AbsoluteDiscounting","text":"NGrams.AbsoluteDiscounting(d::Number)\n\nAbsolute discounting for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"ngrams/#N-Grams-1","page":"N-Grams","title":"N-Grams","text":"","category":"section"},{"location":"ngrams/#","page":"N-Grams","title":"N-Grams","text":"ngrams\nunigrams\nbigrams\ntrigrams","category":"page"},{"location":"ngrams/#NGrams.ngrams","page":"N-Grams","title":"NGrams.ngrams","text":"ngrams(tokens, n; add_bos=true, bos=BOS, add_eos=true, eos=EOS)\n\nIterate over a sequence of n-grams (of length n) from tokens.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.unigrams","page":"N-Grams","title":"NGrams.unigrams","text":"unigrams(tokens)\n\nReturn a sequence of unigrams from tokens.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.bigrams","page":"N-Grams","title":"NGrams.bigrams","text":"bigrams(xs; bos=\"*BOS*\", eos=\"*EOS*\")\n\nIterate over a sequence of bigrams from xs.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.trigrams","page":"N-Grams","title":"NGrams.trigrams","text":"trigrams(xs; bos=\"*BOS*\", eos=\"*EOS*\")\n\nIterate over a sequence of trigrams from xs.\n\n\n\n\n\n","category":"function"}]
}
