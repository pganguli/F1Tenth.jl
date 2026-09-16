function step_multi_agent!(states::Vector{State}, actions::Vector{Action},
    bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64, cfg::LiDARConfig)
    n = length(states)
    new_states = [step_dynamics(states[i], actions[i], params, dt) for i in 1:n]

    scans = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        others = [new_states[j] for j in 1:n if j != i]
        scans[i] = scan(new_states[i], bvh, others, car, cfg)
    end

    car_flags = check_car_collisions(new_states, car)
    collisions = [car_flags[i] || check_track_collision(new_states[i], bvh, car) for i in 1:n]

    return new_states, scans, collisions
end
