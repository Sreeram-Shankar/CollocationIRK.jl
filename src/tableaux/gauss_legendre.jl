using QuadGK
setprecision(BigFloat, 200)

#legendre polynomial function using recurrence relation
function legendre_poly(n::Int, x::BigFloat)
    if n == 0
        return one(x)
    elseif n == 1
        return x
    else
        P_prev = one(x)
        P_curr = x
        for k in 2:n
            P_next = ((2*k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev = P_curr
            P_curr = P_next
        end
        return P_curr
    end
end


#finds the derivative of the legendre polynomials
function legendre_poly_prime(n::Int, x::BigFloat)
    den = x^2 - 1
    if abs(den) > BigFloat("1e-30")
        return n * (x * legendre_poly(n, x) - legendre_poly(n-1, x)) / den
    else
        h = BigFloat("1e-30")
        return (legendre_poly(n, x + h) - legendre_poly(n, x - h)) / (2*h)
    end
end



#finds the roots and weights of the legendre polynomials
function legendre_roots_and_weights(n::Int)
    xs = [cos(BigFloat(π) * BigFloat(4*k - 1) / BigFloat(4*n + 2)) for k in 1:n]
    
    for k in 1:n
        x = xs[k]
        for _ in 1:80
            fx = legendre_poly(n, x)
            dfx = legendre_poly_prime(n, x)
            
            if dfx == 0
                break
            end
            x_new = x - fx / dfx
            if abs(x_new - x) < eps(BigFloat) * 100
                break
            end
            x = x_new
        end
        xs[k] = x
    end
    
    #computes the weights
    ws = BigFloat[]
    for x in xs
        dPn = legendre_poly_prime(n, x)
        w = BigFloat(2) / ((1 - x^2) * dPn^2)
        push!(ws, w)
    end
    
    #transforms from [-1, 1] to [0, 1]
    c = [(x + 1) / 2 for x in xs]
    b = [w / 2 for w in ws]
    
    #sorts the nodes by their c values to make the tableau diagonally dominant
    perm = sortperm(c)
    c_sorted = [c[i] for i in perm]
    b_sorted = [b[i] for i in perm]
    
    return c_sorted, b_sorted, perm
end


#builds the gauss legendre butcher tableau
function build_gauss_legendre_irk(s::Int, verbose::Bool=true)
    if verbose
        println("Computing Gauss–Legendre IRK (s=$s) …")
    end
    c, b, perm = legendre_roots_and_weights(s)

    A = zeros(BigFloat, s, s)
    for j in 1:s
        Lj = lagrange_basis(c, j)
        for i in 1:s
            A[i, j], _ = quadgk(Lj, BigFloat(0), c[i], rtol=eps(BigFloat))
        end
    end

    check_tableau(A, b, c, verbose)
    return A, b, c
end