# The waypoint_follow example, run closed loop with its own planner, should
# drive the same two laps as the Python example (reference: roll_pp_single).
module WaypointFollowExample
include(joinpath(@__DIR__, "..", "examples", "waypoint_follow.jl"))
end

@testset "waypoint_follow example (match Python)" begin
    rows = ref("roll_pp_single_rows")
    n = 1
    py_x, py_y, py_theta = rows[:, 2n+1], rows[:, 2n+2], rows[:, 2n+3]
    py_lap_time = rows[end, 2n+7n+1]

    W = WaypointFollowExample
    conf = W.load_config()
    planner = W.PurePursuitPlanner(conf, 0.17145 + 0.15875)
    env = quiet(() -> F110Env(map=conf["map_path"], map_ext=conf["map_ext"], num_agents=1, timestep=0.01,
        integrator=Integrator.RK4))
    obs, _, done, _ = reset!(env, [conf["sx"] conf["sy"] conf["stheta"]])
    no_scan_noise!(env)   # the Python reference was noise free too
    maxdev = 0.0
    t = 1
    while !done
        speed, steer = W.plan(planner, obs.poses_x[1], obs.poses_y[1], obs.poses_theta[1], W.WORK["tlad"], W.WORK["vgain"])
        obs, _, done, _ = step!(env, [steer speed])
        t += 1
        maxdev = max(maxdev, abs(obs.poses_x[1] - py_x[t]), abs(obs.poses_y[1] - py_y[t]))
    end
    @test t == size(rows, 1)                 # finished on the same step
    @test env.lap_times[1] ≈ py_lap_time atol = 1e-9
    @test env.lap_counts == [2.0]
    @test maxdev < 1e-9

    # the planner pieces match Python's numba helpers on a simple case
    traj = [0.0 0.0; 1.0 0.0; 2.0 0.0]
    proj, dist, tt, i = W.nearest_point_on_trajectory((1.5, 1.0), traj)
    @test proj == (1.5, 0.0) && dist == 1.0 && tt == 0.5 && i == 1
end
