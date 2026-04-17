#computes a Richardson error estimate for the given step based on the order
function richardson_error(f, t, y, h, A, b, c, s, p, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)
    #takes a full step
    y_full, _ = step_irk(f, t, y, h, A, b, c, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)
    h2 = h / 2

    #takes two half steps
    y_half, _ = step_irk(f, t, y, h2, A, b, c, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)
    y_half, f_next = step_irk(f, t + h2, y_half, h2, A, b, c, jac, tol, max_iter, linsolve, diagonalize, Z_schur, T_schur)

    #computes the error based on the order and the tolerance betweeen the full and half steps
    diff = (y_full - y_half) / (2^p - 1)
    EEst = norm(diff) / (norm(y) * tol + tol)
    return EEst, y_half, f_next
end

#adjusts the step size with a PI controller based on the order and the error estimate
function pi_controller(EEst, EEst_prev, h, p, safety=0.9, qmin=0.1, qmax=1.5)
    alpha = 0.7 / (p + 1)
    beta  = 0.4 / (p + 1)

    #very tight cap when already accurate
    if EEst < 1e-3
        q = safety * (1/EEst)^alpha * min((EEst_prev/EEst)^beta, 1.0)
        q = clamp(q, qmin, 1.1)
    else
        q = safety * (1/EEst)^alpha * (EEst_prev/EEst)^beta
        q = clamp(q, qmin, qmax)
    end
    return h * q
end