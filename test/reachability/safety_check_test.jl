using F1Tenth
using ReachabilityAnalysis
using StaticArrays
using LinearAlgebra
using Test

function _straight_test_track()
    n = 20
    centerline = [SVector(Float64(i), 0.0) for i in 0:(n - 1)]
    return F1Tenth.build_track(centerline, fill(1.5, n), fill(1.5, n))
end

@testset "check_flowpipe_safety: tight uncertainty inside a wide corridor is safe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = _straight_test_track()

    initial_set = Zonotope([5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test safe
    @test isempty(violating)
end

@testset "check_flowpipe_safety: huge uncertainty exceeding the corridor is unsafe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = _straight_test_track()

    initial_set = Zonotope([5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        Diagonal([5.0, 5.0, 0.01, 0.01, 0.01, 0.01, 0.01]))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test !safe
    @test !isempty(violating)
end

@testset "check_flowpipe_safety: empty flowpipe throws ArgumentError" begin
    track = _straight_test_track()
    fake_sol = (F=[],)
    @test_throws ArgumentError F1Tenth.check_flowpipe_safety(fake_sol, track)
end

@testset "check_flowpipe_safety: empty track corridor throws ArgumentError" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    empty_track = F1Tenth.build_track(SVector{2,Float64}[], Float64[], Float64[])
    @test_throws ArgumentError F1Tenth.check_flowpipe_safety(sol, empty_track)
end
