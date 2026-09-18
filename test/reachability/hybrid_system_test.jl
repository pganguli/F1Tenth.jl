using F1Tenth
using ReachabilityAnalysis
using StaticArrays
using Test

@testset "build_hybrid_system: automaton structure" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    @test nmodes(hs) == 2
    @test ntransitions(hs) == 2
end

@testset "build_hybrid_system: :ks mode matches KS_RHS at a sample point" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    ks_mode = mode(hs, 1)
    f! = mapping(ks_mode)

    u = [0.0, 0.0, 0.1, 0.3, 0.2, 0.05, 0.01, 1.5, 0.2]  # v = 0.3 -> :ks regime
    du = zeros(9)
    f!(du, u, nothing, 0.0)

    p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
        params.h, params.m, params.Iᶻ, u[9], u[8])
    state = F1Tenth.State(u[1], u[2], u[3], u[4], u[5], u[6], u[7])
    ks_expected = F1Tenth.KS_RHS(state, p)

    @test du[1:5] ≈ collect(ks_expected[1:5]) atol=1e-10
    lf, lr = params.lᶠ, params.lʳ
    expected_psidd = u[8] / (lf + lr) * tan(u[3]) + u[4] / ((lf + lr) * cos(u[3])^2) * u[9]
    @test du[6] ≈ expected_psidd atol=1e-10
    @test du[7] == 0.0
    @test du[8] == 0.0
    @test du[9] == 0.0
end

@testset "build_hybrid_system: :st mode matches ST_RHS at a sample point" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    st_mode = mode(hs, 2)
    f! = mapping(st_mode)

    u = [0.0, 0.0, 0.1, 2.0, 0.2, 0.05, 0.01, 1.5, 0.2]  # v = 2.0 -> :st regime
    du = zeros(9)
    f!(du, u, nothing, 0.0)

    p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
        params.h, params.m, params.Iᶻ, u[9], u[8])
    state = F1Tenth.State(u[1], u[2], u[3], u[4], u[5], u[6], u[7])
    st_expected = F1Tenth.ST_RHS(state, p)

    @test du[1:7] ≈ collect(st_expected) atol=1e-10
    @test du[8] == 0.0
    @test du[9] == 0.0
end
