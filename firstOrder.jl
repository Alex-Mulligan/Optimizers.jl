include("./local.jl")
using LinearAlgebra

abstract type DescentMethod end

struct GradientDescent <: DescentMethod
    α
end
init!(M::GradientDescent, f, ∇f, x) = M
function step!(M::GradientDescent, f, ∇f, x)
    α,g = M.α,∇f(x)
    return x - α*g
end

mutable struct ConjGradientDescent <: DescentMethod
    d
    g
end
function init!(M::ConjGradientDescent, f, ∇f, x)
    M.g = ∇f(x)
    M.d = -M.g
    return M
end
function step!(M::ConjGradientDescent, f, ∇f, x)
    d, g = M.d, M.g
    g′ = ∇f(x)
    β = max(0, dot(g′, g′-g)/(g⋅g))
    d′ = -g′ + β*d
    x′ = line_search(f, x, d′)
    M.d, M.g = d′, g′
    return x′
end

mutable struct Momentum <: DescentMethod
    α
    β
    v
end
function init!(M::Momentum, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function step!(M::Momentum, f, ∇f, x)
    α, β, v, g = M.α, M.β, M.v, ∇f(x)
    v[:] = β*v - α*g
    return x + v
end

mutable struct NesterovMomentum <: DescentMethod
    α
    β
    v
end
function init!(M::NesterovMomentum, f, ∇f, x)
    M.v = zeros(length(x))
    return M
end
function step!(M::NesterovMomentum, f, ∇f, x)
    α, β, v = M.α, M.β, M.v
    v[:] = β*v - α*∇f(x + β*v)
    return x + v
end

mutable struct Adagrad <: DescentMethod
    α
    ϵ
    s
end
function init!(M::Adagrad, f, ∇f, x)
    M.s = zeros(length(x))
    return M
end
function step!(M::Adagrad, f, ∇f, x)
    α,ϵ,s,g = M.α,M.ϵ,M.s,∇f(x)
    s[:] += g.*g
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct RMSProp <: DescentMethod
    α
    γ
    ϵ
    s
end
function init!(M::RMSProp, f, ∇f, x)
    M.s = zeros(length(x))
    return M
end
function step!(M::RMSProp, f, ∇f, x)
    α, γ, ϵ, s, g = M.α, M.γ, M.ϵ, M.s, ∇f(x)
    s[:] = γ*s + (1-γ)*(g.*g)
    return x - α*g ./ (sqrt.(s) .+ ϵ)
end

mutable struct Adadelta <: DescentMethod
    γg
    γx
    ϵ
    sg
    sx
end
function init!(M::Adadelta, f, ∇f, x)
    M.sg = zeros(length(x))
    M.sx = zeros(length(x))
end
function step!(M::Adadelta, f, ∇f, x)
    γg,γx,ϵ,sg,sx,g = M.γg,M.γx,M.ϵ,M.sg,M.sx,∇f(x)
    sg[:] = γg*sg + (1-γg)*(g.*g)
    x′ = - (sqrt.(sx) .+ ϵ) ./ (sqrt.(sg) .+ ϵ) .* g
    sx[:] = γx*sx + (1-γx)*(x′.*x′)
    return x + x′
end

mutable struct Adam <: DescentMethod
    α
    γm1
    γm2
    ϵ
    steps
    em1
    em2
end
function init!(M::Adam, f, ∇f, x)
    M.steps = 0
    M.em1 = zeros(length(x))
    M.em2 = zeros(length(x))
end
function step!(M::Adam, f, ∇f, x)
    α,γm1,γm2,ϵ,steps,em1,em2,g = M.α,M.γm1,M.γm2,M.ϵ,M.steps,M.em1,M.em2,∇f(x)
    em1[:] = γm1*em1 + (1-γm1)*g
    em2[:] = γm2*em2 + (1-γm2)*(g.*g)
    M.steps = steps + 1
    em1′ = em1 ./ (1-γm1)
    em2′ = em2 ./ (1-γm2)
    return x - α*em1′ ./ (sqrt.(em2′) .+ ϵ)
end

mutable struct HypergradientDescent <: DescentMethod
    α
    Startα
    μ
    Oldg
end
function init!(M::HypergradientDescent, f, ∇f, x)
    M.α = M.Startα
    M.Oldg = zeros(length(x))
    return M
end
function step!(M::HypergradientDescent, f, ∇f, x)
    α,μ,g,Oldg = M.α,M.μ,∇f(x),M.Oldg
    α = α + μ*(g⋅Oldg)
    M.α,M.Oldg = α,g
    return x - α*g
end

mutable struct HyperNesterov <: DescentMethod
    α
    Startα
    β
    μ
    v
    Oldg
end
function init!(M::HyperNesterov, f, ∇f, x)
    M.α = M.Startα
    M.Oldg = zeros(length(x))
    M.v = zeros(length(x))
    return M
end
function step!(M::HyperNesterov, f, ∇f, x)
    α,μ,g,Oldg,β,v = M.α,M.μ,∇f(x),M.Oldg,M.β,M.v
    α = α + μ*(g⋅Oldg)
    v[:] = β*v - α*∇f(x + β*v)
    M.α,M.Oldg = α,g
    return x + v
end

function run!(M::DescentMethod, f, ∇f, x; n=1e3, ϵ=1e-3)
    init!(M, f, ∇f, x)
    nrm = 0
    n_iter = 0
    while n_iter < n
        x′ = step!(M, f, ∇f, x)
        nrm = norm(x′-x)
        n_iter += 1
        if nrm < ϵ
            return x′
        end
        x = x′
    end
    return x
end