#defines a single step of the IRK solver
function step_irk(f, t, y, h, A, b, c, jac=nothing, tol=BigFloat("1e-10"), max_iter=50, linsolve=nothing, diagonalize=false, Z=nothing, T_schur=nothing)
    s = length(b)
    n = length(y)
    T = eltype(y)
    Y = repeat(reshape(y, 1, length(y)), s, 1)
    t_nodes = [t + c[i] * h for i in 1:s]

    #builds the residual
    function residual(z_flat)
        Z = reshape(z_flat, s, n)
        R = zeros(T, size(Z))
        for i in 1:s
            acc = zeros(T, n)
            for j in 1:s
                acc += A[i, j] * f(t_nodes[j], Z[j, :])
            end
            R[i, :] = Z[i, :] - y - h * acc
        end
        return vec(R)
    end

    #builds the Jacobian
    Jf = [jac !== nothing ? jac(t_nodes[j], Y[j, :]) : finite_diff_jac(z -> f(t_nodes[j], z), Y[j, :]) for j in 1:s]
    function jacobian(z_flat)
        Z = reshape(z_flat, s, n)
        J_full = zeros(T, s * n, s * n)
        I_n = Matrix{T}(I, n, n)
        for j in 1:s
            for i in 1:s
                block = -h * A[i, j] * Jf[j]
                if i == j
                    block = block + I_n
                end
                J_full[(i-1)*n+1:i*n, (j-1)*n+1:j*n] = block
            end
        end
        return J_full
    end

    #solves the nonlinear system of equations
    z0 = vec(Y)

    #uses the Schur decomposition if it is provided otherwise is normal
    if diagonalize && Z !== nothing && T_schur !== nothing
        z_star, status = newton_solve_schur(residual, z0, Jf, h, A, Z, T_schur, n, s, tol, max_iter, linsolve)
    else
        z_star, status = newton_solve(residual, z0, jacobian, tol, max_iter, linsolve)
    end

    if status != :converged
        error("Newton solver failed to converge")
    end
    Y = reshape(z_star, s, n)
    K = zeros(T, s, n)
    for i in 1:s
        K[i, :] = f(t_nodes[i], Y[i, :])
    end
    return y + h * sum([b[i] * K[i, :] for i in 1:s]), K[end, :]
end


#solver for the IVP using the IRK solver
function solve_irk(f, t_span, y0, h, family=:gl, s=3, jac=nothing, tol=BigFloat("1e-10"), max_iter=50, verbose=true, linsolve=nothing, adaptive=false, diagonalize=false)
    #gets the type of the initial condition and changes to a float if it is an integer
    T = eltype(y0)
    if T <: Integer
        T = float(T)
    end

    #gets the tableau and unpacks it
    tab = get_tableau(family, s, verbose)
    A, b, c = tab.A, tab.b, tab.c
    Z_schur = family == :radau ? tab.Z : nothing
    T_schur = family == :radau ? tab.T : nothing

    #sets up the grid and initial conditions
    t0, tf = BigFloat(t_span[1]), BigFloat(t_span[2])
    h = BigFloat(h)
    y0 = BigFloat.(y0)

    #gets the order of the method
    p = family == :gl ? 2*s : 2*s - 1

    #if the solver is not adaptive it just loops with a fixed dt
    if !adaptive
        N = Int(ceil((tf - t0) / h))
        t_grid = [t0 + BigFloat(n - 1) * h for n in 1:(N + 1)]
        Y = zeros(BigFloat, N+1, length(y0))
        Y[1, :] = y0
        dY = zeros(BigFloat, N+1, length(y0))
        dY[1, :] = f(t0, y0)
        for n in 1:N
            y_next, f_next = step_irk(f, t_grid[n], Y[n, :], h, A, b, c, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)
            Y[n+1, :] = y_next
            dY[n+1, :] = f_next
            if verbose
                println("Step $n: t = $(Float64(t_grid[n+1])), y = $(Float64.(Y[n+1, :]))")
            end
        end
        if verbose
            println("Completed IVP solve using $family IRK of order $s")
        end
        return T.(t_grid), Matrix{T}(Y), Matrix{T}(dY)

    #if the solver is adaptive it loops with a variable dt
    else
        t = t0
        y = copy(y0)
        t_out = [t]
        y_out = [copy(y)]
        EEst_prev = BigFloat(1)
        step_count = 0
        dy_out = [f(t0, y0)]

        while t < tf
            #avoids overshoot the end of the interval
            h = min(h, tf - t)

            #computes the error estimate
            EEst, y_new, f_next = richardson_error(f, t, y, h, A, b, c, s, p, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)

            #accepts the step if the error estimate is less than the tolerance
            if EEst <= 1
                t += h
                y = y_new
                push!(t_out, t)
                push!(y_out, copy(y))
                push!(dy_out, f_next)
                step_count += 1
                if verbose
                    println("Step $step_count: t = $(Float64(t)), h = $(Float64(h)), EEst = $(Float64(EEst))")
                end
                EEst_prev = EEst
            end

            #updates the step size regardless of result
            h = pi_controller(EEst, EEst_prev, h, p)
        end

        #returns the grid and solution
        if verbose
            println("Completed IVP solve using $family IRK of order $s")
        end
        t_grid = T.(t_out)
        Y = Matrix{T}(reduce(hcat, y_out)')
        dY = Matrix{T}(reduce(hcat, dy_out)')
        return t_grid, Y, dY
    end
end