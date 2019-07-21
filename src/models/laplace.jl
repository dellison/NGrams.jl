"""
    Laplace

Laplace (add-1) probability smoothing for n-gram language modeling.
"""
struct Laplace end

const Add1Smoothing = Laplace

function p(model::Laplace, counts, history, token)
    c, N = observed_ratio(counts, history, token)
    V = length(counts.counts)
    return (c + 1) / (N + V)
end
