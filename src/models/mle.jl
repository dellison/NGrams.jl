"""
    MLE

Maximum-likelihood estimation for ngram language modeling.
"""
struct MLE end

function p(::MLE, counts, history, token)
    count, total = observed_ratio(counts, history, token)
    return count / total
end

p(::MLE, counts, token) = count(counts, token) / total(counts)
