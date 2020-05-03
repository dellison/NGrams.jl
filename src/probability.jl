abstract type ProbabilityEstimator end

# API:
# prob(::ProbabilityEstimator, counts, x)    => p(x)
# prob(::ProbabilityEstimator, counts, h, x) => p(x | h)

prob(p::ProbabilityEstimator, counts, h, x) = prob(p, submodel(counts, h), x)
prob(p::ProbabilityEstimator, counts, x)    = error("not implemented!")

"""
    NGrams.MLE()

Maximum Likelihood Estimation for n-gram language modeling.
"""
struct MLE <: ProbabilityEstimator end
prob(::MLE, counts, x) = count(counts, x) / total(counts)

"""
    NGrams.AddK(k::Number)

Add-k probability smoothing for n-gram language modeling.
"""
struct AddK{K<:Number} <: ProbabilityEstimator
    k::K
end
function prob(k::AddK, counts, x)
    c, tot, n = count(counts, x), total(counts), length(counts)
    (c + k.k) / (tot + (n * k.k) + k.k)
end

"""
    NGrams.Laplace()

Laplace (add-1) smoothing for n-gram language modeling.
"""
struct Laplace <: ProbabilityEstimator end
const Add1 = Laplace
function prob(::Laplace, counts, x)
    c, tot = count(counts, x), total(counts)
    return (c + one(c)) / (tot + length(counts) + one(c))
end

"""
    NGrams.AbsoluteDiscounting(d::Number)

Absolute discounting for n-gram language modeling.
"""
struct AbsoluteDiscounting{D<:Number} <: ProbabilityEstimator
    d::D
end
prob(d::AbsoluteDiscounting, counts, x) = max(count(counts, x) - d.d, 0) / total(counts)

struct Interpolated{N,T<:Number}
    k::NTuple{N,T}
end

Base.getindex(l::Interpolated, i) = l.k[i]
Base.lastindex(l::Interpolated) = length(l.k)


"""
    LinearInterpolation(λ)

Linear interpolation for probability smoothing in n-gram language modeling.
"""
struct LinearInterpolation{N,T<:Number} <: ProbabilityEstimator
    lambda::NTuple{N,T}

    function LinearInterpolation(λ::Tuple)
        @assert sum(λ) == 1.
        N, T = length(λ), eltype(λ)
        new{N,T}(λ)
    end
end
LinearInterpolation(v::AbstractVector) = LinearInterpolation(Tuple(v))

Base.getindex(l::LinearInterpolation, i) = l.lambda[i]
Base.lastindex(l::LinearInterpolation) = length(l.lambda)

prob(λ::LinearInterpolation, counts, x) = prob(MLE(), counts, x)
function prob(λ::LinearInterpolation, counts, h, x)
    H = length(h)
    p = prob(λ, counts, x) * λ[H+1]
    for i in 1:H
        h_ = h[end-i+1:end]
        p += prob(λ, submodel(counts, h_), x) * λ[i]
    end
    return p
end
