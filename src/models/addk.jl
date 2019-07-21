"""
    AddKSmoothing

Add-k probability smoothing for n-gram language modeling.
"""
struct AddK{K<:Number}
    k::K
end

function p(model::AddK, counts, history, token)
    c, N = observed_ratio(counts, history, token)
    V, k = length(counts.counts), model.k
    return (c + k) / (N + k * V)
end

function p(model::AddK, counts, token)
    c, N = observed_ratio(counts, token)
    V, k = length(counts.counts), model.k
    return (c + k) / (N + k * V)
end
