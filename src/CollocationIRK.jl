module CollocationIRK

using QuadGK
using OrdinaryDiffEq
using SciMLBase
using LinearSolve
using RecursiveArrayTools
import LinearAlgebra: norm, lu, I, schur

include("tableaux/tableau_utils.jl")
include("tableaux/gauss_legendre.jl")
include("tableaux/radau.jl")
include("tableaux/tableau_cache.jl")

include("solver/newton.jl")
include("solver/controller.jl")
include("solver/interpolation.jl")
include("solver/integrator.jl")
include("solver/algorithms.jl")
include("solver/perform_step.jl")

export solve_irk, FIRK_GL, FIRK_RadauIIA, hermite_interp, interp_solution, HermiteInterpolation

end