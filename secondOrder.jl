include("./firstOrder.jl")
using LinearAlgebra

function Newton(∇f, H, x, tol, max_iter)
    k,update = 1,fill(Inf, length(x))
    while norm(update) > tol && k <= max_iter
        update = H(x) \ ∇f(x)
        x -= update
        k += 1
    end
    return x
end

#an option for univariate function optimization
#uses an approximate of the Hessian; requires two starting points
function Secant_approximation(f′, a, b, tol)
    a′ = f′(a)
    update = Inf
    while abs(update) > tol
        b′ = f′(b)
        update = (b-a)/(b′-a′)*b′
        a, b, a′ = b, b - update, b′
    end
    return b
end

mutable struct DFP <: DescentMethod
    Q
end
function init!(M::DFP, f, ∇f, x)
    M.Q = Matrix(1.0I, length(x), length(x))
    return M
end
function step!(M::DFP, f, ∇f, x)
    Q,g = M.Q,∇f(x)
    x′ = line_search(f, x, -Q*g)
    g′ = ∇f(x′)
    dx = x′ - x
    dg = g′ - g
    Q[:] = Q - (Q*dg*dg'*Q)/(dg'*Q*dg)+(dx*dx')/(dx'*dg)
    return x′
end

mutable struct BFGS <: DescentMethod
    Q
end
function init!(M::BFGS, f, ∇f, x)
    M.Q = Matrix(1.0I, length(x), length(x))
    return M
end
function step!(M::BFGS, f, ∇f, x)
    Q,g = M.Q,∇f(x)
    x′ = line_search(f, x, -Q*g)
    g′ = ∇f(x′)
    dx = x′ - x
    dg = g′ - g
    Q[:] = Q - (dx*dg'Q+Q*dg*dx')/(dx'*dg)+(1+(dg'*Q*dg)/(dx'*dg))[1]*(dx*dx')/(dx'*dg)
    return x′
end

mutable struct LBFGS <: DescentMethod
    m
    dx
    dg
    qs
end
function init!(M::LBFGS, f, ∇f, x)
    M.dx = []
    M.dg = []
    M.qs = []
    return M
end
function step!(M::LBFGS, f, ∇f, x)
    dx,dg,qs,g = M.dx,M.dg,M.qs,∇f(x)
    m = length(dx)
    if m > 0
        q = g
        for i in m:-1:1
            qs[i] = copy(q)
            q -= (dx[i]⋅q)/(dg[i]⋅dx[i])*dg[i]
        end
        z = (dg[m].*dx[m].*q)/(dg[m]⋅dg[m])
        for i in 1 : m
            z += dx[i]*(dx[i]⋅qs[i]-dg[i]⋅z)/(dg[i]⋅dx[i])
        end
        x′ = line_search(f, x, -z)
    else
        x′ = line_search(f, x, -g)
    end
    g′ = ∇f(x′)
    push!(dx, x′-x)
    push!(dg, g′-g)
    push!(qs, zeros(length(x)))
    while length(dx) > M.m
        popfirst!(dx)
        popfirst!(dg)
        popfirst!(qs)
    end
    return x′
end