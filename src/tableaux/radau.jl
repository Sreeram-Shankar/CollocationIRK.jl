using QuadGK
setprecision(BigFloat, 200)


#jacobi polynomial function using hypergeometric function
function jacobi_P(n::Int, alpha::BigFloat, beta::BigFloat, x::BigFloat)
    if n == 0
        return one(x)
    elseif n == 1
        return (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
    else
        P_prev = one(x)
        P_curr = (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
        for k in 2:n
            a1 = (2*k + alpha + beta - 1) * ((2*k + alpha + beta) * (2*k + alpha + beta - 2) * x + (alpha^2 - beta^2))
            a2 = 2 * (k + alpha - 1) * (k + beta - 1) * (2*k + alpha + beta)
            P_next = (a1 * P_curr - a2 * P_prev) / (2 * k * (k + alpha + beta) * (2*k + alpha + beta - 2))
            P_prev = P_curr
            P_curr = P_next
        end
        return P_curr
    end
end

#finds the roots of the jacobi polynomial using the bisection method
function jacobi_roots_radau_interior(n::Int, dps_scan::Int=0)
    alpha = BigFloat(1)
    beta = BigFloat(0)
    f = x -> jacobi_P(n, alpha, beta, x)
    
    if dps_scan == 0
        dps_scan = max(2000, 400 * n)
    end
    xs = [BigFloat(-1) + 2*i/(dps_scan-1) for i in 0:(dps_scan-1)]
    fs = [f(x) for x in xs]
    
    #implements the bisection method to find the roots
    roots = BigFloat[]
    for i in 1:(dps_scan-1)
        a, b = xs[i], xs[i+1]
        fa, fb = fs[i], fs[i+1]
        if fa == 0
            push!(roots, a)
            continue
        end
        if fa * fb < 0
            for _ in 1:200
                m = (a + b) / 2
                fm = f(m)
                if abs(fm) < BigFloat("1e-70") || abs(b - a) < BigFloat("1e-50")
                    push!(roots, m)
                    break
                end
                if fa * fm < 0
                    b, fb = m, fm
                else
                    a, fa = m, fm
                end
            end
        end
    end
    
    #filters out the roots that are not on the interior
    roots = [r for r in roots if -1 < r < 1]
    roots = sort(unique([nstr_fixed(r, 60) for r in roots]))
    roots = [parse(BigFloat, r) for r in roots]
    
    #if the number of roots isnt the number of interior nodes, recurses with higher precision
    if length(roots) != n
        if dps_scan < 20000
            return jacobi_roots_radau_interior(n, 20000)
        end
        error("Expected $n interior roots, got $(length(roots))")
    end
    return roots
end

#generates radau nodes on the interval [0, 1]
function radau_right_nodes_on_01(s::Int)
    if s < 1
        error("Radau quadrature requires s >= 1.")
    end
    if s == 1
        return [BigFloat(1)]
    end
    interior = jacobi_roots_radau_interior(s - 1)
    x_all = [interior; BigFloat(1)]
    c = [(x + 1) / 2 for x in x_all]
    return c
end

#builds the butcher tableau matrices by doing integration on the Lagrange basis polynomial
function build_A_b(c::Vector{BigFloat})
    s = length(c)
    A = zeros(BigFloat, s, s)
    b = zeros(BigFloat, s)
    for j in 1:s
        Lj = lagrange_basis(c, j)
        b[j], _ = quadgk(Lj, BigFloat(0), BigFloat(1), rtol=eps(BigFloat))
        for i in 1:s
            A[i, j], _ = quadgk(Lj, BigFloat(0), c[i], rtol=eps(BigFloat))
        end
    end
    return A, b
end

#main Radau IIA builder function
function build_radau_irk(s::Int, verbose::Bool=true)
    if verbose
        println("  Computing Radau–IIA nodes for s=$s …")
    end
    c = radau_right_nodes_on_01(s)
    if verbose
        println("  Building A and b by integrating Lagrange basis …")
    end
    A, b = build_A_b(c)
    if verbose
        println("  Verifying tableau …")
    end
    check_tableau(A, b, c, verbose)
    return A, b, c
end