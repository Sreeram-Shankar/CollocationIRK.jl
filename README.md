# CollocationIRK

Julia package: **Gauss–Legendre** and **Radau IIA** fully implicit Runge–Kutta (FIRK) methods as collocation integrators, exposed as `FIRK_GL` and `FIRK_RadauIIA` algorithm types for `OrdinaryDiffEq.solve` on `SciMLBase.ODEProblem`.

**Methods.** With `s` stages: Gauss–Legendre has classical order `2s`; Radau IIA (right Radau, endpoint `c_s = 1`) has order `2s−1`. Coefficients `(A,b,c)` are built in **`BigFloat`** (QuadGK on Lagrange basis; see `src/tableaux/`). In `solve_irk`, `u0`, the time span, and `dt` are promoted to **`BigFloat`** for the internal loop; outputs are cast back to `eltype(u0)` (integer `u0` is converted to a floating type first).

**Which method when.** **Radau IIA** is an **L-stable** FIRK: the linear stability function satisfies **R(∞) = 0**, so stiff decay modes and unresolved temporal scales are damped rather than ringing. That makes it the natural choice for **stiff** problems where you want a **dissipative** (contracting) discrete flow in the stiff limit. **Gauss–Legendre** collocation yields a **symplectic** implicit Runge–Kutta method: for Hamiltonian or other structure-preserving problems it behaves like a **conserving** (reversible, long-time geometric) integrator rather than one that artificially damps fast modes—so it is usually preferred for **nonstiff** (or stiffness-secondary) flows where symplecticity and symmetry matter more than L-stability. In short: Radau biases toward **damping**; Gauss–Legendre toward **conservation** of the discrete spectral structure.

**High order and reference use.** The package is aimed at **high stage counts** and **high classical order** with coefficients computed at extended precision, so you can generate **reference solutions** (convergence studies, benchmarks against other solvers, manufactured solutions) where a cheap low-order method would not be accurate enough.

**Stepper.** Each step solves the stage equations by **Newton iteration** (tolerance `tol`, at most `max_iter` iterations). The condensed stage system can use a user-supplied **`LinearSolve.jl`** algorithm (`linsolve`); otherwise a dense direct linear solve is used. For Radau, **`diagonalize=true`** applies a Schur-style decoupling of the stage blocks (Hairer–Wanner style) to reduce the cost of those solves when it applies. **Adaptive** mode uses a Richardson-type error estimate and a PI-type step-size update on the accepted error ratio.

**SciML surface.** `solve(prob, alg; dt=...)` returns a standard **`ODESolution`** (via `SciMLBase.build_solution`): `sol.t`, `sol.u`, `sol.retcode`, and dense evaluation **`sol(t)`** using **cubic Hermite** interpolation from stored states and time derivatives. **`ODEFunction`** may supply **`jac`**; if absent, Jacobians are finite-differenced. **`SciMLBase.isinplace(prob)`** is honored: in-place `f!(du,u,p,t)` is wrapped by allocating a temporary `du` and returning it to the `(t,y) -> dy` form the collocation residual uses.

**Dependencies (see `Project.toml`).** `OrdinaryDiffEq`, `SciMLBase`, `LinearSolve`, `QuadGK`, `LinearAlgebra`, `RecursiveArrayTools`. Julia **≥ 1.9**.

## Install

From a clone of this repo (path = your machine):

```julia
using Pkg
Pkg.develop(path)
```

Session:

```julia
using CollocationIRK, OrdinaryDiffEq
```

## Examples

**Out-of-place** `f(u,p,t) -> du`:

```julia
f(u, p, t) = [u[2], -u[1]]
prob = ODEProblem(f, [1.0, 0.0], (0.0, 10.0))
sol = solve(prob, FIRK_GL(stages=3), dt=0.1)           # order 6, fixed dt
sol = solve(prob, FIRK_RadauIIA(stages=3), dt=0.1)     # order 5, fixed dt
```

**In-place** `f!(du,u,p,t)` with analytical Jacobian:

```julia
function f!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
function jac!(J, u, p, t)
    J[1,1] = 0.0; J[1,2] = 1.0
    J[2,1] = -1.0; J[2,2] = 0.0
end
ff = ODEFunction(f!; jac=jac!)
prob = ODEProblem(ff, [1.0, 0.0], (0.0, 10.0))
sol = solve(prob, FIRK_RadauIIA(stages=3, diagonalize=true), dt=0.1)
```

**Adaptive** (still pass an initial `dt`):

```julia
sol = solve(prob, FIRK_RadauIIA(stages=3, adaptive=true), dt=0.1)
```

**Iterative linear solver** (example: GMRES):

```julia
using LinearSolve
sol = solve(prob, FIRK_GL(stages=3, linsolve=KrylovJL_GMRES()), dt=0.1)
```

**Dense output:**

```julia
y_mid = sol(0.15)
```

**Comparing vectors in the REPL** (not re-exported by this package):

```julia
using LinearAlgebra
norm(sol.u[end] .- u_reference)
```

**Plots:** load `Plots` (or another backend) yourself; then `plot(sol)` follows the usual SciML conventions.

## Algorithm struct keywords

| Keyword | Default | Meaning |
|---------|---------|---------|
| `stages` | `3` | Number of stages `s` (orders above). |
| `tol` | `BigFloat("1e-10")` | Newton residual tolerance. |
| `max_iter` | `50` | Maximum Newton iterations per step. |
| `linsolve` | `nothing` | `LinearSolve.jl` algorithm for stage Newton systems; `nothing` uses a direct dense solve. |
| `adaptive` | `false` | Use error estimate + step-size controller. |
| `verbose` | `false` | Print per-step diagnostics. |
| `diagonalize` | `false` | **Radau only:** Schur diagonalization of the Radau stage coupling. |

## References

- E. Hairer, G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd Revised ed., Springer Series in Computational Mathematics 8 (Springer, 2008).
- E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd Revised ed., Springer Series in Computational Mathematics 14 (Springer, 2008).
- E. Hairer, G. Wanner, [Stiff differential equations solved by Radau methods](https://doi.org/10.1016/S0377-0427(99)00134-X), *Journal of Computational and Applied Mathematics* **111** (1–2) (1999), 93–111 (special issue: Numerical Methods for Differential Equations, Coimbra, 1998; received 28 April 1998, revised 26 December 1998).