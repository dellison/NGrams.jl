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
        @test length.(corpus) == [5, 6, 8, 6, 7, 9] # 41 words total

        @testset "MLE" begin
            lm = train_lm(corpus, 2, MLE())

            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.p(lm, "<unk>") == 0.
            @test NGrams.p(lm, "*BOS*") == NGrams.p(lm, "*EOS*") == (6 / (41 + 12))
            @test NGrams.p(lm, "don't") == 2 / (41 + 12)
            @test NGrams.p(lm, ["i"], "<unk>") == 0.
            @test NGrams.p(lm, ["i"], "don't") == 0.5
            @test NGrams.p(lm, ["i"], "keep") == 0.25
            @test NGrams.p(lm, ["i"], "guess") == 0.25
        end

        @testset "Laplace Smoothing" begin
            lm = train_lm(corpus, 2, Laplace())
            # V is the size of the vocabulary (plus the OOV token <unk>)
            V = length(unique(Base.Iterators.flatten(corpus))) + 1
            @test V == 34

            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.p(lm, "<unk>") == 1 / (53 + 35)
            @test NGrams.p(lm, "*BOS*") == 7 / (53 + 35)
            @test NGrams.p(lm, "don't") == 3 / (53 + 35)
            @test NGrams.p(lm, ["i"], "<unk>") == 1 / (4 + 34)
            @test NGrams.p(lm, ["i"], "don't") == 3 / (4 + 34)
            @test NGrams.p(lm, ["i"], "keep") == 2 / (4 + 34)
            @test NGrams.p(lm, ["i"], "guess") ==  2 / (4 + 34)
        end

        @testset "Add-k Smoothing" begin
            lm = train_lm(corpus, 2, AddK(0.1))
            # V is the size of the vocabulary (plus the OOV token <unk>)
            V = length(unique(Base.Iterators.flatten(corpus))) + 0.1
            @test V == 33.1

            @test NGrams.count(lm, "<unk>") == 0
            @test NGrams.count(lm, ["<unk>", "<unk>"]) == 0
            @test NGrams.count(lm, "i") == 4
            @test NGrams.count(lm, ["i", "don't"]) == 2
            @test NGrams.p(lm, "<unk>") == 0.1 / (53 + 3.5)
            @test NGrams.p(lm, "*BOS*") == 6.1 / (53 + 3.5)
            @test NGrams.p(lm, "don't") == 2.1 / (53 + 3.5)
            @test NGrams.p(lm, ["i"], "<unk>") == 0.1 / (4 + 3.4)
            @test NGrams.p(lm, ["i"], "don't") == 2.1 / (4 + 3.4)
            @test NGrams.p(lm, ["i"], "keep") == 1.1 / (4 + 3.4)
            @test NGrams.p(lm, ["i"], "guess") ==  1.1 / (4 + 3.4)
        end

        @testset "Linear Interpolation" begin
            lm = train_lm(corpus, 2, LinearInterpolation([0.8, 0.2]))

            @test NGrams.p(lm, ["i"], "<unk>") == 0.0
            @test NGrams.p(lm, ["i"], "don't") == .2 * 2/47 + 0.8 * 2/4
        end

        @testset "Absolute Discounting" begin
            lm = train_lm(corpus, 2, AbsoluteDiscounting(0.5))
            @test NGrams.p(lm, "<unk>") == 0.0
            @test NGrams.p(lm, "road") == 0.5 / 53
            @test NGrams.p(lm, "sometimes") == 1.5 / 53
        end
    end

end
