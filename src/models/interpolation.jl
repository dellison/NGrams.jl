"""
    LinearInterpolation

Estimation of n-gram probabilities using linear interpolation.
"""
struct LinearInterpolation
    λ::Vector{Float64}

    function LinearInterpolation(λ)
        @assert sum(λ) == 1.
        new(λ)
    end
end

function p(model::LinearInterpolation, counts, history, token)
    prob, λ = 0.0, model.λ
    for i in 1:length(history)
        c, N = observed_ratio(counts, history[end-i+1:end], token)
        prob += c / N * λ[i]
    end
    prob += count(counts, token) / total(counts) * λ[length(history)+1]
    return prob
end

p(::LinearInterpolation, counts, token) = count(counts, token) / total(counts)

function train_lm(corpus, n, model::LinearInterpolation; bos=BOS, eos=EOS)
    @assert length(model.λ) == n
    lm = LanguageModel(n, model; bos=bos, eos=eos)
    for sentence in corpus
        train!(lm, sentence)
    end
    return lm
end
