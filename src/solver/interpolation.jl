#defines a struct and constructor for the hermite interpolation
struct HermiteInterpolation{T}
    t_grid::Vector{T}
    Y::Matrix{T}
    dY::Matrix{T}
end

#handles SciML compatibility for the interpolation
function (interp::HermiteInterpolation)(t_eval, idxs, deriv, p, continuity)
    if t_eval isa AbstractVector
        vals = [interp_solution(t, interp.t_grid, interp.Y, interp.dY) for t in t_eval]
        return RecursiveArrayTools.DiffEqArray(vals, t_eval)
    else
        return interp_solution(t_eval, interp.t_grid, interp.Y, interp.dY)
    end
end
SciMLBase.interp_summary(::HermiteInterpolation) = "Cubic Hermite interpolation"


#evaluates the cubic hermite interpolant at the given time using the values at the endpoints
function hermite_interp(t_eval, t_n, t_n1, y_n, y_n1, dy_n, dy_n1)
    h = t_n1 - t_n
    θ = (t_eval - t_n) / h

    #builds the basis functions for the cubic hermite interpolant
    h00 = (1 + 2θ) * (1 - θ)^2
    h10 = θ * (1 - θ)^2
    h01 = θ^2 * (3 - 2θ)
    h11 = θ^2 * (θ - 1)

    return h00 * y_n + h10 * h * dy_n + h01 * y_n1 + h11 * h * dy_n1
end

#evaluates the solution at any time t_eval using cubic hermite interpolation to interpolate the solution and derivative
function interp_solution(t_eval, t_grid, Y, dY)
    #finds the interval containing the evaluation time
    n = length(t_grid)
    if t_eval <= t_grid[1]
        return Y[1, :]
    end
    if t_eval >= t_grid[end]
        return Y[end, :]
    end

    #performs a binary search for the interval
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if t_grid[mid] <= t_eval
            lo = mid
        else
            hi = mid
        end
    end
    return hermite_interp(t_eval, t_grid[lo], t_grid[hi], Y[lo, :], Y[hi, :], dY[lo, :], dY[hi, :])
end