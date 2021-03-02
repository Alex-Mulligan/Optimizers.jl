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