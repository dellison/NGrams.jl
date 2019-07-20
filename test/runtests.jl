using NGrams, Test

@testset "NGrams.jl" begin

    @testset "Grams" begin
        corpus = ["hello", "there"]

        grams = unigrams(corpus)
        @test eltype(grams) <: Unigram
        @test collect(grams) == [("*BOS*",),
                                 ("hello",),
                                 ("there",),
                                 ("*EOS*",)]

        grams = bigrams(corpus)
        @test eltype(grams) <: Bigram
        @test collect(grams) == [("*BOS*","hello"),
                                 ("hello","there"),
                                 ("there","*EOS*")]

        grams = trigrams(corpus)
        @test eltype(grams) <: Trigram
        @test collect(grams) == [("*BOS*","*BOS*","hello"),
                                 ("*BOS*","hello","there"),
                                 ("hello","there","*EOS*")]

        for n in 1:3
            @test collect(ngrams(n, corpus)) == collect(ngrams(corpus, n))
        end
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
        @test length.(corpus) == [5, 6, 8, 6, 7, 9]

        @testset "MLE" begin
            lm = train_lm(corpus, 2, MLE())

            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.p(lm, ["i"], "don't") == 0.5
            @test NGrams.p(lm, ["i"], "keep") == 0.25
            @test NGrams.p(lm, ["i"], "guess") == 0.25
        end
    end

end
