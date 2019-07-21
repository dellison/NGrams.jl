"""
    MLE

Maximum-likelihood estimation for ngrams.
"""
struct MLE end

function p(::MLE, counts, history, token)
    count, total = observed_ratio(counts, history, token)
    return count / total
end
