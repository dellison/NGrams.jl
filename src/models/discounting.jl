"""
    AbsoluteDiscounting

Absolute discounting for probability estimation in n-gram language modeling.
"""
struct AbsoluteDiscounting{D<:Number}
    d::D
end

function p(model::AbsoluteDiscounting, counts, history, token)
    count, total = observed_ratio(counts, history, token)
    return max(count - model.d, 0) / total
end

p(model::AbsoluteDiscounting, counts, token) =
    max(count(counts, token) - model.d, 0) / total(counts)
