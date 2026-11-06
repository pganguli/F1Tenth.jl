# Closed-loop runs with the waypoint_follow example's planner. This branch's
# vehicle model departs from Python on purpose, so these check behavior
# (laps finish, crashes are caught, parameter updates apply) rather than
# matching Python's trajectories step for step.
module WaypointFollowExample
include(joinpath(@__DIR__, "..", "examples", "waypoint_follow.jl"))
end

const W = WaypointFollowExample
const CONF = W.load_config()
const PLANNER = W.PurePursuitPlanner(CONF, 0.17145 + 0.15875)
const START = [CONF["sx"] CONF["sy"] CONF["stheta"]]

# Pure pursuit for every car, each with its own speed gain.
function pursuit(obs, gains)
    act = zeros(length(gains), 2)
    for (i, g) in enumerate(gains)
        speed, steer = W.plan(PLANNER, obs.poses_x[i], obs.poses_y[i], obs.poses_theta[i], W.WORK["tlad"], g)
        act[i, :] = [steer, speed]
    end
    return act
end

# Run `policy(obs)` until done or `max_steps`; `events[t]` runs before step t.
function drive(env, start, policy; max_steps, events=Dict{Int,Function}())
    obs, _, done, _ = reset!(env, start)
    no_scan_noise!(env)
    t = 0
    while !done && t < max_steps
        haskey(events, t) && events[t](env)
        obs, _, done, _ = step!(env, policy(obs))
        t += 1
    end
    return obs, done, t
end

@testset "closed-loop runs" begin
    wpts = PLANNER.waypoints
    col(k) = W._col(PLANNER, k)
    opp_start = [wpts[201, col("wpt_xind")] wpts[201, col("wpt_yind")] wpts[201, col("wpt_thind")] + pi / 2]

    @testset "waypoint_follow example, 1 car, 2 laps" begin
        env = quiet(() -> F110Env(map=CONF["map_path"], map_ext=CONF["map_ext"], num_agents=1, timestep=0.01,
            integrator=Integrator.RK4))
        obs, done, _ = drive(env, START, obs -> pursuit(obs, [W.WORK["vgain"]]); max_steps=12000)
        @test done && env.lap_counts == [2.0] && env.collisions == [0.0]
        # Python's lap time for the same run, as a sanity bound: better steering
        # shouldn't change the pace much.
        py_rows = ref("roll_pp_single_rows")
        py_lap_time = py_rows[end, 2+7+1]
        @test isapprox(env.lap_times[1], py_lap_time; rtol=0.02)
    end
    @testset "2 cars, ego rams the opponent" begin
        env = quiet(() -> F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=2))
        obs, done, _ = drive(env, [START; opp_start], obs -> pursuit(obs, [1.5, 0.5]); max_steps=12000)
        @test done && env.collisions == [1.0, 1.0]
    end
    @testset "Euler, 0.02 s steps, LiDAR offset, into a wall" begin
        env = F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1, timestep=0.02, integrator=Integrator.Euler,
            lidar_dist=0.25)
        obs, done, _ = drive(env, START, _ -> [0.35 6.0]; max_steps=3000)
        @test done && env.collisions == [1.0]
        @test env.obs.poses_theta == [0.0]   # as Python, yaw is zeroed on a wall hit
    end
    @testset "update_params! mid-run" begin
        slippery = merge(GYM_DICT, Dict("mu" => 0.6, "m" => 4.5, "length" => 0.8, "width" => 0.5))
        heavy = merge(GYM_DICT, Dict("m" => 5.0, "a_max" => 6.0))
        env = quiet(() -> F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=2, seed=7))
        seen = VehicleParams[]
        events = Dict(100 => (e -> update_params!(e, slippery; index=2)),
            101 => (e -> append!(seen, [c.params for c in e.sim.agents])),
            250 => (e -> update_params!(e, heavy)))
        drive(env, [START; opp_start], obs -> pursuit(obs, [1.5, 0.5]); max_steps=600, events=events)
        @test seen == [GYM_PARAMS, VehicleParams(slippery)]
        @test all(car.params == VehicleParams(heavy) for car in env.sim.agents)
        @test env.sim.params == GYM_PARAMS   # car-car collisions keep the initial body size
    end

    # the planner pieces match Python's numba helpers on a simple case
    traj = [0.0 0.0; 1.0 0.0; 2.0 0.0]
    proj, dist, tt, i = W.nearest_point_on_trajectory((1.5, 1.0), traj)
    @test proj == (1.5, 0.0) && dist == 1.0 && tt == 0.5 && i == 1
end
