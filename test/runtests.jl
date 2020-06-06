using NGrams, Test

@testset "NGrams.jl" begin

    @testset "Grams" begin
        corpus = ["hello", "there"]

        grams = unigrams(corpus)
        @test eltype(grams) <: NGrams.Unigram
        @test collect(grams) == [("hello",),
                                 ("there",)]

        grams = bigrams(corpus)
        @test eltype(grams) <: NGrams.Bigram
        @test collect(grams) == [("*BOS*","hello"),
                                 ("hello","there"),
                                 ("there","*EOS*")]

        grams = trigrams(corpus)
        @test eltype(grams) <: NGrams.Trigram
        @test collect(grams) == [("*BOS*","*BOS*","hello"),
                                 ("*BOS*","hello","there"),
                                 ("hello","there","*EOS*"),
                                 ("there","*EOS*","*EOS*")]

        for n in 1:3
            @test collect(ngrams(n, corpus)) == collect(ngrams(corpus, n))
            @test collect(ngrams(n, 1:5)) == collect(ngrams(1:5, n))
        end

        @test collect(ngrams(3, 1:5, bos=nothing, eos=nothing)) ==
            [(1,2,3), (2,3,4), (3,4,5)] # padless
    end

    @testset "Language Models" begin

        corpus = """
                 sometimes i don't know where
                 this dirty road is taking me
                 sometimes i don't even know the reason why
                 but i guess i keep a-gamblin'
                 lots of booze and lots of ramblin'
                 well it's easier than just a-waitin' around to die
                 """
        corpus = [split(line) for line in split(strip(corpus), "\n")]

        @test length(corpus) == 6
        @test length.(corpus) == [5, 6, 8, 6, 7, 9] # 41 words total

        corpus_size = length(collect(Base.Iterators.flatten(corpus))) + length(corpus)

        # V is the size of the vocabulary (plus the OOV token <unk>)
        V = length(unique(Base.Iterators.flatten(corpus))) + 1
        @test V == 34

        function train_lm(corpus, N, estimator)
            lm = NGrams.LanguageModel(N, estimator)
            for sentence in corpus
                NGrams.fit!(lm, sentence)
            end
            return lm
        end

        lm = NGrams.LanguageModel(3)
        @test NGrams.gram_size(lm) == 3 == NGrams.order(lm) + 1

        @testset "MLE" begin
            lm = train_lm(corpus, 2, NGrams.MLE())
            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.prob(lm, "<unk>") == 0.
            @test NGrams.prob(lm, "*BOS*") == (6 / (41 + 6))
            @test NGrams.prob(lm, "don't") == 2 / (41 + 6)
            @test NGrams.prob(lm, ["i"], "<unk>") == 0.
            @test NGrams.prob(lm, ["i"], "don't") == 0.5
            @test NGrams.prob(lm, ["i"], "keep") == 0.25
            @test NGrams.prob(lm, ["i"], "guess") == 0.25
        end

        @testset "Laplace Smoothing" begin
            lm = train_lm(corpus, 2, NGrams.Laplace())
            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.prob(lm, "<unk>") == 1 / (corpus_size + V + 1)
            @test NGrams.prob(lm, "*BOS*") == 7 / (corpus_size + V + 1)
            @test NGrams.prob(lm, "don't") == 3 / (corpus_size + V + 1)
            @test NGrams.prob(lm, ["i"], "<unk>") == 1 / 8
            @test NGrams.prob(lm, ["i"], "don't") == 3 / 8
            @test NGrams.prob(lm, ["i"], "keep")  == 2 / 8
            @test NGrams.prob(lm, ["i"], "guess") ==  2 / 8
        end

        @testset "Add-k Smoothing" begin
            lm = train_lm(corpus, 2, NGrams.AddK(0.1))
            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.prob(lm, "<unk>") == 0.1 / (corpus_size + 0.1V + 0.1)
            @test NGrams.prob(lm, "*BOS*") == 6.1 / (corpus_size + 0.1V + 0.1)
            @test NGrams.prob(lm, "don't") == 2.1 / (corpus_size + 0.1V + 0.1)
            @test NGrams.prob(lm, ["i"], "<unk>") ≈ 0.1 / 4.4
            @test NGrams.prob(lm, ["i"], "don't") ≈ 2.1 / 4.4
            @test NGrams.prob(lm, ["i"], "keep")  ≈ 1.1 / 4.4
            @test NGrams.prob(lm, ["i"], "guess") ≈ 1.1 / 4.4
        end

        @testset "Linear Interpolation" begin
            lm = train_lm(corpus, 2, NGrams.LinearInterpolation([0.8, 0.2]))
            @test NGrams.prob(lm, ["i"], "<unk>") == 0.0
            @test NGrams.prob(lm, ["i"], "don't") == .2*2/(41+6) + 0.8*0.5
        end

        @testset "Absolute Discounting" begin
            lm = train_lm(corpus, 2, NGrams.AbsoluteDiscounting(0.5))
            @test NGrams.prob(lm, "<unk>") == 0.
            @test NGrams.prob(lm, "road") == 0.5 / corpus_size
            @test NGrams.prob(lm, "sometimes") == 1.5 / corpus_size
        end

        @testset "Generation" begin
            lm = NGrams.LanguageModel(3)
            NGrams.fit!(lm, ["this", "is", "good"])
            NGrams.fit!(lm, ["this", "is", "bad"])

            for i in 1:10
                x = last(NGrams.generate(lm, 3, ["this", "is"]))
                @test x in ("good", "bad")
            end

            NGrams.fit!(lm, ["this", "is", "neutral"])

            counts, total = Dict(), 0
            for _ in 1:10000
                token = NGrams.generate(lm, 3, ["this", "is"]) |> last
                counts[token] = get(counts, token, 0) + 1
                total += 1
            end
            for token in ("good", "bad", "neutral")
                @test counts[token] / total ≈ 0.33 atol=0.01
            end
        end
    end

end
