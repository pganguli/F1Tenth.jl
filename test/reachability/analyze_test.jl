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

@testset "reachability_analyze: initial_set straddling v=0.5 throws" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 0.3 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    @test_throws ArgumentError F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1))
end
