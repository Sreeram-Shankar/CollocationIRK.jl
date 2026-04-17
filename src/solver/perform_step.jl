#adds compatible functions for the collocation rhs with SciML for the ODEProblem
function _collocation_rhs(prob::ODEProblem)
    if SciMLBase.isinplace(prob)
        (t, y) -> begin
            du = similar(y)
            prob.f(du, y, prob.p, t)
            return du
        end
    else
        (t, y) -> prob.f(y, prob.p, t)
    end
end

#adds compatible functions for the collocation Jacobian with SciML for the ODEProblem
function _collocation_jac(prob::ODEProblem)
    prob.f.jac === nothing && return nothing
    if SciMLBase.isinplace(prob)
        (t, y) -> begin
            jp = prob.f.jac_prototype
            J = if jp === nothing
                n = length(y)
                zeros(eltype(y), n, n)
            else
                zero(jp)
            end
            prob.f.jac(J, y, prob.p, t)
            return J
        end
    else
        (t, y) -> prob.f.jac(y, prob.p, t)
    end
end

#solves the ODEProblem using the Gauss-Legendre algorithm
function OrdinaryDiffEq.solve(prob::ODEProblem, alg::FIRK_GL; dt, kwargs...)
    f = _collocation_rhs(prob)
    jac = _collocation_jac(prob)
    t_span = prob.tspan
    y0 = prob.u0
    s = alg.stages
    tol = alg.tol
    max_iter = alg.max_iter
    verbose = alg.verbose
    linsolve = alg.linsolve
    t_grid, Y, dY = solve_irk(f, t_span, y0, dt, :gl, s, jac, tol, max_iter, verbose, linsolve, alg.adaptive)
    u = [Y[i, :] for i in 1:length(t_grid)]
    interp = HermiteInterpolation(t_grid, Y, dY)
    return SciMLBase.build_solution(prob, alg, t_grid, u, interp=interp, retcode=ReturnCode.Success)
end

#solves the ODEProblem using the RadauIIA algorithm
function OrdinaryDiffEq.solve(prob::ODEProblem, alg::FIRK_RadauIIA; dt, kwargs...)
    f = _collocation_rhs(prob)
    jac = _collocation_jac(prob)
    t_span = prob.tspan
    y0 = prob.u0
    s = alg.stages
    tol = alg.tol
    max_iter = alg.max_iter
    verbose = alg.verbose
    linsolve = alg.linsolve
    diagonalize = alg.diagonalize
    t_grid, Y, dY = solve_irk(f, t_span, y0, dt, :radau, s, jac, tol, max_iter, verbose, linsolve, alg.adaptive, diagonalize)
    u = [Y[i, :] for i in 1:length(t_grid)]
    interp = HermiteInterpolation(t_grid, Y, dY)
    return SciMLBase.build_solution(prob, alg, t_grid, u, interp=interp, retcode=ReturnCode.Success)
end