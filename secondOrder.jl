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