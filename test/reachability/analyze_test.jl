using F1Tenth
using ReachabilityAnalysis
using LinearAlgebra
using Test

@testset "reachability_analyze: :st initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
    @test length(sol.F[1]) > 0
end

@testset "reachability_analyze: :ks initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], 0.001 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.001 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
end

@testset "reachability_analyze: crosses the v=0.5 KS/ST mode boundary" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    # v0 ≈ 0.45, fully within the :ks regime (v_hi <= 0.5). a is state-augmented
    # with a zero derivative, so it stays ≈ 1.0 for the whole horizon (du[4] = a):
    # v(t) ≈ 0.45 + 1.0*t crosses 0.5 at t ≈ 0.05s, well inside a 0.15s tspan,
    # so the flowpipe should visit both the :ks and :st locations.
    initial_set = Zonotope([0.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0], 0.001 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([1.0, 0.0], 0.001 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.15);
        alg=TMJets(abstol=1e-12, orderT=6, orderQ=1))

    @test length(sol.F) >= 2
end

@testset "reachability_analyze: initial_set straddling v=0.5 throws" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 0.3 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    @test_throws ArgumentError F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1))
end
