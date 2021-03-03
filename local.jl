include("./bracketing.jl")

function line_search(f, x, d)
    o = α -> f(x+α*d)
    a,b = bracket_min(o)
    a,b = golden_section_search(o, a, b, 1e2)
    α = (a+b)/2
    return x+α*d
end

function backtrack_line_search(f, ∇f, x, d, α; p=0.5, β=1e-4)
    y,g = f(x),∇f(x)
    while f(x+α*d) > y + β*α*(g ⋅ d)
        α *= p
    end
    return α
end