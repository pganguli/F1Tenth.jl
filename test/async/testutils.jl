using StaticArrays

function _small_track()
    n = 8
    R = 5.0
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
end

function _async_test_setup(; n_agents::Int=1)
    track = _small_track()
    bvh = F1Tenth.build_bvh(track.segments)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=1.0, max_range=10.0, noise_std=0.0)
    states = [F1Tenth.State(Float64(i - 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in 1:n_agents]
    actions = [F1Tenth.Action(0.0, 0.0) for _ in 1:n_agents]
    return (; track, bvh, params, car, cfg, states, actions)
end

mutable struct FakeRenderer <: F1Tenth.AbstractRenderer
    frames::Vector{Any}
    fail_on_call::Int
    calls::Int
end

FakeRenderer(; fail_on_call::Int=0) = FakeRenderer(Any[], fail_on_call, 0)

F1Tenth.init!(r::FakeRenderer, track::F1Tenth.Track) = r

function F1Tenth.render!(r::FakeRenderer, states::Vector{F1Tenth.State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    r.calls += 1
    r.fail_on_call != 0 && r.calls == r.fail_on_call && error("FakeRenderer: forced failure")
    push!(r.frames, (copy(states), copy(scans), copy(collisions)))
    return nothing
end

F1Tenth.close!(r::FakeRenderer) = nothing
F1Tenth.reset!(r::FakeRenderer, track::F1Tenth.Track) = (empty!(r.frames); r.calls = 0; r)
