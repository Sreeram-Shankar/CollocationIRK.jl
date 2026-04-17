using CollocationIRK
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase
using Test

#u' = [u₂, -u₁]  ⟹  u₁(t) = cos(t), u₂(t) = -sin(t) for u₀ = [1, 0]
u_analytic(t) = [cos(t), -sin(t)]

@testset "CollocationIRK" begin
    f(u, p, t) = [u[2], -u[1]]
    u0 = [1.0, 0.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f, u0, tspan)
    ref = u_analytic(10.0)

    @testset "FIRK_GL fixed dt" begin
        sol = solve(prob, FIRK_GL(stages=3), dt=0.1)
        @test sol.retcode == ReturnCode.Success
        @test length(sol.t) == length(sol.u) >= 2
        @test norm(sol.u[end] - ref) < 1e-6
    end

    @testset "FIRK_RadauIIA fixed dt" begin
        sol = solve(prob, FIRK_RadauIIA(stages=3), dt=0.1)
        @test sol.retcode == ReturnCode.Success
        @test norm(sol.u[end] - ref) < 1e-6
    end

    @testset "FIRK_RadauIIA diagonalize" begin
        sol = solve(prob, FIRK_RadauIIA(stages=3, diagonalize=true), dt=0.1)
        @test norm(sol.u[end] - ref) < 1e-6
    end

    @testset "FIRK_RadauIIA adaptive" begin
        sol = solve(prob, FIRK_RadauIIA(stages=3, adaptive=true), dt=0.1)
        @test sol.retcode == ReturnCode.Success
        @test norm(sol.u[end] - ref) < 1e-4
    end

    @testset "FIRK_GL adaptive" begin
        sol = solve(prob, FIRK_GL(stages=3, adaptive=true), dt=0.1)
        @test sol.retcode == ReturnCode.Success
        @test norm(sol.u[end] - ref) < 1e-4
    end

    @testset "FIRK_GL stages=2" begin
        sol = solve(prob, FIRK_GL(stages=2), dt=0.05)
        @test sol.retcode == ReturnCode.Success
        @test norm(sol.u[end] - ref) < 1e-4
    end

    @testset "FIRK_RadauIIA stages=4" begin
        sol = solve(prob, FIRK_RadauIIA(stages=4), dt=0.1)
        @test sol.retcode == ReturnCode.Success
        @test norm(sol.u[end] - ref) < 1e-8
    end

    @testset "dense output sol(t)" begin
        sol = solve(prob, FIRK_GL(stages=4), dt=0.1)
        tq = π / 4
        @test norm(sol(tq) - u_analytic(tq)) < 1e-4
    end

    @testset "in-place f! with analytical jac!" begin
        function f!(du, u, p, t)
            du[1] = u[2]
            du[2] = -u[1]
        end
        function jac!(J, u, p, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
        end
        ff = ODEFunction(f!; jac=jac!)
        prob_iip = ODEProblem(ff, u0, tspan)

        sol_gl = solve(prob_iip, FIRK_GL(stages=3), dt=0.1)
        @test sol_gl.retcode == ReturnCode.Success
        @test norm(sol_gl.u[end] - ref) < 1e-6

        sol_rd = solve(prob_iip, FIRK_RadauIIA(stages=3, diagonalize=true), dt=0.1)
        @test sol_rd.retcode == ReturnCode.Success
        @test norm(sol_rd.u[end] - ref) < 1e-6
    end
end
