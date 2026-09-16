using F1Tenth
using StaticArrays
using Test

@testset "collides: SAT on axis-aligned rectangles" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    overlapping = [SVector(1.0, 1.0), SVector(3.0, 1.0), SVector(3.0, 3.0), SVector(1.0, 3.0)]
    separated = [SVector(5.0, 5.0), SVector(7.0, 5.0), SVector(7.0, 7.0), SVector(5.0, 7.0)]
    touching = [SVector(2.0, 0.0), SVector(4.0, 0.0), SVector(4.0, 2.0), SVector(2.0, 2.0)]

    @test F1Tenth.collides(a, overlapping)
    @test !F1Tenth.collides(a, separated)
    @test F1Tenth.collides(a, touching)
end

@testset "collides: SAT on a rotated rectangle" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    # A 45-degree-rotated square centered on a's center (1,1), sized to
    # clearly overlap a's interior (not just touch at a boundary point,
    # which would make the test fragile to floating-point rounding).
    rotated = [SVector(1.0, -0.2), SVector(2.2, 1.0), SVector(1.0, 2.2), SVector(-0.2, 1.0)]
    @test F1Tenth.collides(a, rotated)
end

@testset "check_car_collisions: all-pairs over an agent list" begin
    car = F1Tenth.CarFootprint(; length=0.5, width=0.3)
    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # overlaps agent 1
        F1Tenth.State(20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0), # isolated
    ]
    flags = F1Tenth.check_car_collisions(states, car)
    @test flags == [true, true, false]
end

@testset "check_track_collision: penetrating vs clear pose" begin
    half_side = 5.0
    h = half_side
    segments = [
        F1Tenth.Segment(SVector(-h, -h), SVector(h, -h)),
        F1Tenth.Segment(SVector(h, -h), SVector(h, h)),
        F1Tenth.Segment(SVector(h, h), SVector(-h, h)),
        F1Tenth.Segment(SVector(-h, h), SVector(-h, -h)),
    ]
    bvh = F1Tenth.build_bvh(segments)
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)

    clear = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test !F1Tenth.check_track_collision(clear, bvh, car)

    penetrating = F1Tenth.State(h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test F1Tenth.check_track_collision(penetrating, bvh, car)
end
