using F1Tenth
using StaticArrays
using Random: MersenneTwister
using Test

function _square_room_bvh(half_side::Float64)
    h = half_side
    segments = [
        F1Tenth.Segment(SVector(-h, -h), SVector(h, -h)),
        F1Tenth.Segment(SVector(h, -h), SVector(h, h)),
        F1Tenth.Segment(SVector(h, h), SVector(-h, h)),
        F1Tenth.Segment(SVector(-h, h), SVector(-h, -h)),
    ]
    return F1Tenth.build_bvh(segments)
end

@testset "cast_ray: hits a known wall at the expected distance" begin
    bvh = _square_room_bvh(10.0)
    t = F1Tenth.cast_ray(SVector(0.0, 0.0), 0.0, bvh, 30.0)
    @test isapprox(t, 10.0; atol=1e-6)

    t_up = F1Tenth.cast_ray(SVector(0.0, 0.0), π / 2, bvh, 30.0)
    @test isapprox(t_up, 10.0; atol=1e-6)
end

@testset "cast_ray: returns max_range when the wall is out of range" begin
    bvh = _square_room_bvh(10.0)
    t = F1Tenth.cast_ray(SVector(0.0, 0.0), 0.0, bvh, 5.0)
    @test t == 5.0
end

@testset "car_vertices: axis-aligned car at origin" begin
    car = F1Tenth.CarFootprint(; length=2.0, width=1.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    verts = F1Tenth.car_vertices(state, car)
    @test length(verts) == 4
    @test SVector(1.0, 0.5) in verts
    @test SVector(-1.0, -0.5) in verts
end

@testset "scan: sees the wall and a blocking agent" begin
    bvh = _square_room_bvh(10.0)
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.0, max_range=30.0, noise_std=0.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # fov=0.0 with num_beams=3 sends all 3 beams straight along +x.
    ranges_no_agents = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg)
    @test all(isapprox.(ranges_no_agents, 10.0; atol=1e-6))

    blocking_agent = F1Tenth.State(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ranges_with_agent = F1Tenth.scan(state, bvh, [blocking_agent], car, cfg)
    @test all(r -> r < 10.0 - 1e-6, ranges_with_agent)
end

@testset "scan: deterministic noise via explicit rng" begin
    bvh = _square_room_bvh(10.0)
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=30.0, noise_std=0.05)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    r1 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r2 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r3 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(2))
    @test r1 == r2
    @test r1 != r3
end
