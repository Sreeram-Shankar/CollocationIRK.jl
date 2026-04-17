#defines the finite difference Jacobian
function finite_diff_jac(fun, x)
    n = length(x)
    T = eltype(x)
    f0 = fun(x)
    J = zeros(T, n, n)
    for j in 1:n
        step = sqrt(eps(real(T))) * max(one(real(T)), abs(x[j]))
        dx = zeros(T, n)
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    end
    return J
end

#Newton solver for the nonlinear system of equations
function newton_solve(residual, y0, jac=nothing, tol=BigFloat("1e-10"), max_iter=50, linsolve=nothing)
    y = copy(y0)

    #reuses the Jacobian if it is provided otherwise uses finite differences
    J = jac !== nothing ? jac(y) : finite_diff_jac(residual, y)

    norm_r_prev = norm(residual(y))
    for i in 1:max_iter
        r = residual(y)
        norm_r = norm(r)

        #checks if the norm of the residual is less than the tolerance for divergence or convergence
        if norm_r < tol
            return y, :converged
        end
        if i > 5 && norm_r > 10 * norm_r_prev
            return y, :diverged
        end

        #uses LU if nothing is provided otherwise uses the linear solver
        if linsolve === nothing
            dy = lu(J) \ (-r)
        else
            prob = LinearProblem(J, -r)
            sol = solve(prob, linsolve)
            dy = sol.u
        end

        y += dy
        if norm(dy) < tol
            return y, :converged
        end
        norm_r_prev = norm_r
    end
    return y, :max
end

#Newton solver using the Schur decomposition of A
function newton_solve_schur(residual, y0, Jf, h, A, Z, T_schur, n, s, tol=BigFloat("1e-10"), max_iter=100, linsolve=nothing)
    y = copy(y0)
    h_f64 = Float64(h)
    tol_f64 = Float64(tol)
    J = Matrix{Float64}(Jf[1])

    #builds the Schur matrix
    W_schur = kron(I(s), I(n)) - h_f64 * kron(T_schur, J)
    W_schur_lu = lu(W_schur)
    KZ = kron(Z, I(n))
    KZt = kron(Z', I(n))

    #solves the nonlinear system of equations directly exploting the structure of the Schur matrix
    norm_r_prev = norm(residual(y))
    for iter in 1:max_iter
        r = residual(y)
        norm_r = norm(r)

        #checks if the norm of the residual is less than the tolerance
        if norm_r < tol
            return y, :converged
        end
        if iter > 5 && norm_r > 10 * norm_r_prev
            return y, :diverged
        end

        #transforms the residual rw = (Z' ⊗ I) * r
        rw = KZt * Float64.(r)
        dw = W_schur_lu \ rw
        dz = -(KZ * dw)

        #updates the solution
        y .+= dz
    
        #checks if the norm of the update is less than the tolerance again after the update
        if norm(dz) < tol_f64
            return y, :converged
        end
        norm_r_prev = norm_r
    end
    return y, :maxiter
end