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

@testset "step_multi_agent!: Phase 1 state evolution is unaffected" begin
    bvh = _square_room_bvh(50.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.0)

    state = F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.1, 1.0)
    dt = 0.01

    expected = F1Tenth.step_dynamics(state, action, params, dt)
    new_states, _, _ = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, dt, cfg)

    @test new_states[1] == expected
end

@testset "step_multi_agent!: flags a car driven into the wall" begin
    bvh = _square_room_bvh(1.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)

    # Close to the wall boundary and driving forward into it. At v=5.0,
    # dt=0.1 the car's body straddles the wall (x=1.0) after the step,
    # so a body edge crosses the boundary segment (rather than the whole
    # car tunneling past it, which check_track_collision cannot detect).
    state = F1Tenth.State(0.6, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.0, 0.0)

    _, _, collisions = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, 0.1, cfg)
    @test collisions[1]
end

@testset "step_multi_agent!: flags two cars driven into each other" begin
    bvh = _square_room_bvh(50.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)

    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    actions = [F1Tenth.Action(0.0, 0.0), F1Tenth.Action(0.0, 0.0)]

    _, scans, collisions = F1Tenth.step_multi_agent!(states, actions, bvh, params, car, 0.01, cfg)
    @test collisions == [true, true]
    @test length(scans) == 2
    @test length(scans[1]) == cfg.num_beams
end

@testset "step_multi_agent!: LiDAR noise is actually applied to scans" begin
    bvh = _square_room_bvh(50.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.05)

    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.0, 0.0)

    _, scans1, _ = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, 0.0, cfg;
        rng=MersenneTwister(1))
    _, scans2, _ = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, 0.0, cfg;
        rng=MersenneTwister(2))
    _, scans_clean, _ = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, 0.0,
        F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.0))

    @test scans1[1] != scans2[1]
    @test scans1[1] != scans_clean[1]
end
