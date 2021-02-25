function bracket_min(f, x=0, s=1e-2, k=2)
    a,fa = x,f(x)
    b,fb = a+s,f(a+s)
    if fb > fa
        a,b = b,a
        fa,fb = fb,fa
        s = -s
    end
    while true
        c,fc = b+s,f(b+s)
        if fc>fb
            return a<c ? (a,c) : (c,a)
        end
        a,fa,b,fb = b,fb,c,fc
        s *= k
    end
end

function fibonacci_search(f, a, b, n; ϵ=0.01)
    s = (1-√5)/(1+√5)
    ρ = 1/(Base.MathConstants.golden*(1-s^(n+1))/(1-s^n))
    d = ρ*b+(1-ρ)*a
    fd = f(d)
    for i in 1 : n-1
        if i == n-1
            c = ϵ*a+(1-ϵ)*d
        else
            c = ρ*a+(1-ρ)*b
        end
        fc = f(c)
        if fc < fd
            b,d,fd = d,c,fc
        else
            a,b = b,c
        end
        ρ = 1/(Base.MathConstants.golden*(1-s^(n-i+1))/(1-s^(n-i)))
    end
    return a<b ? (a,b) : (b,a)
end

function golden_section_search(f, a, b, n)
    ρ = Base.MathConstants.golden-1
    d = ρ*b+(1-ρ)*a
    fd = f(d)
    for i = 1 : n-1
        c = ρ*a+(1-ρ)*b
        fc = f(c)
        if fc < fd
            b,d,fd = d,c,fc
        else
            a,b = b,c
        end
    end
    return a<b ? (a,b) : (b,a)
end

function quadratic_fit_search(f, a, b, c, n)
    fa,fb,fc = f(a),f(b),f(c)
    for i in 1:n-3
        x = 0.5*(fa*(b^2-c^2)+fb*(c^2-a^2)+fc*(a^2-b^2))/(fa*(b-c)+fb*(c-a)+fc*(a-b))
        fx = f(x)
        if x > b
            if fx > fb
                c,fc = x,fx
            else
                a,fa,b,fb = b,fb,x,fx
            end
        elseif x < b
            if fx > fb
                a,fa = x,fx
            else
                c,fc,b,fb = b,fb,x,fx
            end
        end
    end
    return (a,b,c)
end

function bracket_sign_change(f′, a, b; k=2)
    if a > b
        a,b = b,a
    end
    center,half_width = (b+a)/2,(b-a)/2
    while f′(a)*f′(b) > 0
        half_width *= k
        a = center-half_width
        b = center+half_width
    end
    return (a,b)
end

function bisection(f′, a, b, ϵ=1e-3)
    if sign(f′(a)) == sign(f′(b))
        a,b = bracket_sign_change(f′, a, b)
    end
    if a > b
        a,b = b,a
    end
    fa, fb = f′(a), f′(b)
    if fa == 0
        b = a
    end
    if fb == 0
        a = b
    end
    while b-a > ϵ
        x = (a+b)/2
        fx = f′(x)
        if fx == 0
            a,b = x,x
        elseif sign(fx) == sign(fa)
            a = x
        else
            b = x
        end
    end
    return (a,b)
end

function brent_dekker(f, a, b, ϵ=1e-3)
    if f(a)<f(b)
        a,b = b,a
    end
    c = a
    flag = true
    while abs(b-a)>ϵ && f(b) != 0
        if f(a) != f(c) && f(b) != f(c)
            fa,fb,fc = f(a),f(b),f(c)
            s = (a*fb*fc)/((fa-fb)*(fa-fc))+(b*fa*fc)/((fb-fc)*(fb-fa))+(c*fa*fb)/((fc-fa)*(fc-fb))
        else
            x,y = f(b),f(a)
            s = b-((x*(b-a))/(x-y))
        end
        if !((3*a+b)/4 <= s <= b) || (flag && abs(s-b)>=abs(b-c)/2) || (!flag && abs(s-b)>=abs(c-d)/2) || (flag && abs(b-c)<ϵ) || (!flag && abs(c-d)<ϵ)
            s = (a+b)/2
            flag = true
        else
            flag = false
        end
        d,c = c,b
        if f(a)*f(s)<0
            b = s
        else
            a = s
        end
        if abs(f(a)) < abs(f(b))
            a,b = b,a
        end
    end
    return b
end