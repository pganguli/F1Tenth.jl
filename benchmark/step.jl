# The step! time for 1, 2 and 4 cars on the example map, with LiDAR noise.
# All cars drive on the raceline with the pure pursuit of the example. The
# script measures only the time of step!. Run it from the root of the
# repository (the examples environment has GLMakie, which the example
# loads):
#
#   julia --project=examples -t 1 benchmark/step.jl
#   julia --project=examples -t 4 benchmark/step.jl
using F1Tenth, Logging
module W
include(joinpath(@__DIR__, "..", "examples", "waypoint_follow.jl"))
end
conf = W.load_config()
planner = W.PurePursuitPlanner(conf, W.WHEELBASE)
wp = planner.waypoints; xi, yi, ti = (W._col(planner, k) for k in ("wpt_xind", "wpt_yind", "wpt_thind"))
gain = W.WORK["vgain"]; tlad = W.WORK["tlad"]
function bench(n; steps=3000)
    env = Logging.with_logger(() -> F110Env(map=conf["map_path"], map_ext=conf["map_ext"], num_agents=n, timestep=0.01,
        integrator=F1Tenth.Integrator.RK4), Logging.NullLogger())
    idx = [1 + (k - 1) * 150 for k in 1:n]   # at intervals along the raceline, in its direction
    obs, = reset!(env, hcat(wp[idx, xi], wp[idx, yi], wp[idx, ti] .+ pi/2))
    act = zeros(n, 2)
    function plan!(obs)
        for i in 1:n
            sp, st = W.plan(planner, obs.poses_x[i], obs.poses_y[i], obs.poses_theta[i], tlad, gain)
            act[i, 1], act[i, 2] = st, sp
        end
    end
    for _ in 1:300; plan!(obs); obs, = step!(env, act); end
    t = 0.0
    for _ in 1:steps
        plan!(obs)
        t += @elapsed step!(env, act)
    end
    return t / steps * 1e6, sum(env.collisions)
end
for n in (1, 2, 4)
    us, crashed = bench(n)
    println("$(Threads.nthreads()) thread(s), $n car(s): ", round(us; digits=1), " µs per step!",
        crashed > 0 ? " ($(Int(crashed)) crashed; rerun)" : "")
end
