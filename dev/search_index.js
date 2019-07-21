var documenterSearchIndex = {"docs":
[{"location":"languagemodels/#Language-Models-1","page":"Language Models","title":"Language Models","text":"","category":"section"},{"location":"languagemodels/#","page":"Language Models","title":"Language Models","text":"LanguageModel","category":"page"},{"location":"languagemodels/#NGrams.LanguageModel","page":"Language Models","title":"NGrams.LanguageModel","text":"LanguageModel(n::Int, model; bos=\"*BOS*\", eos=\"*EOS*\")\n\nCreate a language model with n-gram length n and probability model model.\n\n\n\n\n\n","category":"type"},{"location":"languagemodels/#Training-1","page":"Language Models","title":"Training","text":"","category":"section"},{"location":"languagemodels/#","page":"Language Models","title":"Language Models","text":"train_lm","category":"page"},{"location":"languagemodels/#NGrams.train_lm","page":"Language Models","title":"NGrams.train_lm","text":"train_lm(corpus, n, model; bos=BOS, eos=EOS)\n\nTrain a language model.\n\n\n\n\n\n","category":"function"},{"location":"languagemodels/#","page":"Language Models","title":"Language Models","text":"<!– ## Evaluation –>","category":"page"},{"location":"languagemodels/#","page":"Language Models","title":"Language Models","text":"<!– ## Generation –>","category":"page"},{"location":"languagemodels/#Probability-and-Smoothing-1","page":"Language Models","title":"Probability and Smoothing","text":"","category":"section"},{"location":"languagemodels/#","page":"Language Models","title":"Language Models","text":"MLE\nAddK\nLaplace","category":"page"},{"location":"languagemodels/#NGrams.MLE","page":"Language Models","title":"NGrams.MLE","text":"MLE\n\nMaximum-likelihood estimation for ngrams.\n\n\n\n\n\n","category":"type"},{"location":"languagemodels/#NGrams.AddK","page":"Language Models","title":"NGrams.AddK","text":"AddKSmoothing\n\nAdd-k probability smoothing for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"languagemodels/#NGrams.Laplace","page":"Language Models","title":"NGrams.Laplace","text":"Laplace\n\nLaplace (add-1) probability smoothing for n-gram language modeling.\n\n\n\n\n\n","category":"type"},{"location":"#NGrams.jl-1","page":"Home","title":"NGrams.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"NGrams.jl is a Julia package for working with n-gram models of natural language.","category":"page"},{"location":"ngrams/#N-Grams-1","page":"N-Grams","title":"N-Grams","text":"","category":"section"},{"location":"ngrams/#","page":"N-Grams","title":"N-Grams","text":"ngrams\nunigrams\nbigrams\ntrigrams","category":"page"},{"location":"ngrams/#NGrams.ngrams","page":"N-Grams","title":"NGrams.ngrams","text":"ngrams(n, tokens; add_bos=true, bos=BOS, add_eos=true, eos=EOS)\n\nReturn a sequence of n-grams (of length n) from tokens.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.unigrams","page":"N-Grams","title":"NGrams.unigrams","text":"unigrams(tokens)\n\nReturn a sequence of unigrams from tokens.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.bigrams","page":"N-Grams","title":"NGrams.bigrams","text":"bigrams(tokens)\n\nReturn a sequence of bigrams from tokens.\n\n\n\n\n\n","category":"function"},{"location":"ngrams/#NGrams.trigrams","page":"N-Grams","title":"NGrams.trigrams","text":"trigrams(tokens)\n\nReturn a sequence of trigrams from tokens.\n\n\n\n\n\n","category":"function"}]
}
