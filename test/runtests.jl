using NGrams, Test

@testset "NGrams.jl" begin

    @testset "Grams" begin
        corpus = ["hello", "there"]

        grams = unigrams(corpus)
        @show grams
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
    end

end
