#creates the Gauss-Legendre algorithm
struct FIRK_GL <: SciMLBase.AbstractODEAlgorithm
    stages::Int
    tol::BigFloat
    max_iter::Int
    linsolve::Any
    adaptive::Bool
    verbose::Bool
end

#creates the Radau IIA algorithm
struct FIRK_RadauIIA <: SciMLBase.AbstractODEAlgorithm
    stages::Int
    tol::BigFloat
    max_iter::Int
    linsolve::Any
    adaptive::Bool
    verbose::Bool
    diagonalize::Bool
end

FIRK_GL(; stages=3, tol=BigFloat("1e-10"), max_iter=50, linsolve=nothing, adaptive=false, verbose=false) = FIRK_GL(stages, tol, max_iter, linsolve, adaptive, verbose)
FIRK_RadauIIA(; stages=3, tol=BigFloat("1e-10"), max_iter=50, linsolve=nothing, adaptive=false, verbose=false, diagonalize=false) = FIRK_RadauIIA(stages, tol, max_iter, linsolve, adaptive, verbose, diagonalize)