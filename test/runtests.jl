using NGrams, Test

@testset "NGrams.jl" begin

    @testset "Grams" begin
        corpus = ["hello", "there"]

        @test collect(unigrams(corpus)) == [("*BOS*",),
                                            ("hello",),
                                            ("there",),
                                            ("*EOS*",)]

        @test collect(bigrams(corpus)) == [("*BOS*","hello"),
                                           ("hello","there"),
                                           ("there","*EOS*")]

        @test collect(trigrams(corpus)) == [("*BOS*","*BOS*","hello"),
                                            ("*BOS*","hello","there"),
                                            ("hello","there","*EOS*")]
    end

end
