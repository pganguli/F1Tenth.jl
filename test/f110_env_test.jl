# Replays of the Python rollouts (generate.py, gen_rollouts). Python ran with
# LiDAR noise off; here noise is turned off after the reset step by setting
# each car's scan_rng to nothing (Python's rng=None). The reset step's noisy
# scan can't change any state (every car is at rest), so only its scan is
# skipped. Julia agrees with Python to ~1e-14 over whole runs, so the
# tolerances below are tight on purpose.

const ROLL_FIELDS = (:poses_x, :poses_y, :poses_theta, :linear_vels_x, :linear_vels_y, :ang_vels_z, :collisions,
    :lap_times, :lap_counts, :checkpoint_done)

# A Python row: (action, Dict of per-car fields, reward, done).
function split_row(row, n)
    action = permutedims(reshape(row[1:2n], 2, n))
    fields = Dict(f => row[2n+(k-1)*n+1:2n+k*n] for (k, f) in enumerate(ROLL_FIELDS))
    return action, fields, row[end-1], row[end] == 1
end

julia_fields(obs, info) =
    Dict(f => Float64.(f === :checkpoint_done ? info.checkpoint_done : getfield(obs, f)) for f in ROLL_FIELDS)

no_scan_noise!(env) = foreach(car -> car.scan_rng = nothing, env.sim.agents)

# Replay Python's actions. Returns the max deviation per field, whether every
# discrete output (collisions, laps, done, reward) matched exactly, and the
# max scan deviation over the saved scans.
function replay(name, env; events=Dict{Int,Function}(), scan_every)
    rows = ref("roll_$(name)_rows")
    py_scans = ref("roll_$(name)_scans")
    n = env.num_agents
    _, f0, _, _ = split_row(rows[1, :], n)
    reset!(env, hcat(f0[:poses_x], f0[:poses_y], f0[:poses_theta]))
    no_scan_noise!(env)
    maxdev = Dict(f => 0.0 for f in ROLL_FIELDS)
    discrete_ok = true
    scan_err = 0.0
    k = 1   # py_scans[1] is the reset step's
    done = false
    for t in 1:size(rows, 1)-1
        haskey(events, t - 1) && events[t-1](env)
        action, py, py_reward, py_done = split_row(rows[t+1, :], n)
        obs, reward, done, info = step!(env, action)
        jl = julia_fields(obs, info)
        for f in ROLL_FIELDS
            maxdev[f] = max(maxdev[f], maximum(abs.(jl[f] .- py[f])))
        end
        discrete_ok &= all(jl[f] == py[f] for f in (:collisions, :lap_counts, :checkpoint_done))
        discrete_ok &= done == py_done && reward == py_reward
        if t % scan_every == 0 || done
            k += 1
            scan_err = max(scan_err, maximum(maximum(abs.(obs.scans[i] .- py_scans[k, i, :])) for i in 1:n))
        end
    end
    return (; maxdev, discrete_ok, scan_err, done)
end

function check_replay(r)
    @test r.discrete_ok
    for f in (:poses_x, :poses_y, :poses_theta, :linear_vels_x, :ang_vels_z, :lap_times)
        @test r.maxdev[f] < 1e-10
    end
    @test r.scan_err < 1e-10
end

# params as Python dicts
const GYM_DICT = Dict(String(k) => getfield(GYM_PARAMS, k) for k in F1Tenth.PARAM_KEYS)
const SLIPPERY = merge(GYM_DICT, Dict("mu" => 0.6, "m" => 4.5, "length" => 0.8, "width" => 0.5))
const HEAVY = merge(GYM_DICT, Dict("m" => 5.0, "a_max" => 6.0))

@testset "F110Env rollouts (match Python)" begin
    @testset "waypoint_follow example, 1 car, 2 laps" begin
        env = quiet(() -> F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1, timestep=0.01,
            integrator=Integrator.RK4))
        r = replay("pp_single", env; scan_every=100)
        @test r.done && env.lap_counts == [2.0]
        check_replay(r)
    end
    @testset "2 cars, ego rams the opponent" begin
        env = quiet(() -> F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=2))
        r = replay("pp_two", env; scan_every=50)
        @test r.done
        check_replay(r)
    end
    @testset "Euler, 0.02 s steps, LiDAR offset, into a wall" begin
        env = F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1, timestep=0.02, integrator=Integrator.Euler,
            lidar_dist=0.25)
        r = replay("wall_euler", env; scan_every=10)
        @test r.done && env.collisions == [1.0]
        @test env.obs.poses_theta == [0.0]   # Python zeroes yaw on a wall hit
        check_replay(r)
    end
    @testset "update_params! mid-run" begin
        env = quiet(() -> F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=2, seed=7))
        events = Dict(100 => (e -> update_params!(e, SLIPPERY; index=2)),   # Python index=1
            250 => (e -> update_params!(e, HEAVY)))
        r = replay("params_update", env; events=events, scan_every=50)
        @test all(car.params == VehicleParams(HEAVY) for car in env.sim.agents)
        @test env.sim.params == GYM_PARAMS   # car-car collisions keep the initial body size
        check_replay(r)
    end
end

@testset "F110Env API" begin
    quiet() do
        for name in keys(F1Tenth.BUILTIN_MAPS)
            env = F110Env(map=name, num_agents=1)
            @test env.map_ext == F1Tenth.BUILTIN_MAPS[name] && env.sim.scan_simulator.map !== nothing
        end
        env = F110Env()
        @test env.map_name == "vegas" && env.num_agents == 2 && env.ego_idx == 1 && env.timestep == 0.01
        @test env.integrator == Integrator.RK4 && env.params == VehicleParams()

        # params as a dict, like Python; every key is required
        env = F110Env(map="berlin", params=merge(GYM_DICT, Dict("mu" => 0.9)))
        @test env.params.mu == 0.9
        @test_throws KeyError F110Env(map="berlin", params=delete!(copy(GYM_DICT), "mu"))

        env = F110Env(map="berlin", num_agents=2, ego_idx=2)
        obs, reward, done, info = reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])
        @test obs.ego_idx == 1   # as Python: the env never tells the simulator its ego index
        @test obs[:poses_x] == obs["poses_x"] == obs.poses_x == [0.0, 2.0]
        @test reward == 0.01 && !done && info.checkpoint_done == [false, false]
        @test length(obs.scans) == 2 && all(length.(obs.scans) .== 1080)
        @test env.current_time == 0.01
        @test Set(keys(obs)) == Set([:ego_idx, :scans, :poses_x, :poses_y, :poses_theta, :linear_vels_x,
            :linear_vels_y, :ang_vels_z, :collisions, :lap_times, :lap_counts])
        snap = copy(obs)
        step!(env, [0.2 3.0; 0.0 1.0])
        @test snap.poses_x == [0.0, 2.0] && obs.poses_x != snap.poses_x   # obs is reused, copies are not
        @test_throws ArgumentError reset!(env, [0.0 0.0 0.0])
        @test_throws DimensionMismatch step!(env, [0.0 1.0])
        @test_throws BoundsError update_params!(env, VehicleParams(); index=3)

        # scan noise: N(0, 0.01), restarted on reset
        env = F110Env(map="berlin", num_agents=1)
        a = copy(reset!(env, [0.0 0.0 0.0])[1].scans[1])
        b = copy(reset!(env, [0.0 0.0 0.0])[1].scans[1])
        @test a == b
        no_scan_noise!(env)
        clean = step!(env, [0.0 0.0])[1].scans[1]   # still at rest at the same pose
        @test 0.007 < std(a - clean) < 0.013

        env = F110Env(map="berlin", num_agents=1)
        skirk = joinpath(F1Tenth.MAPS_DIR, "skirk.yaml")
        update_map!(env, skirk, ".png")
        @test env.sim.scan_simulator.map === F1Tenth.load_map(skirk, ".png")

        # stepping allocates nothing (single-threaded; per-car threading
        # allocates its tasks). Forced bounds checks, as under Pkg.test, move
        # a few small scratch vectors to the heap.
        env = F110Env(map="berlin", num_agents=2)
        reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])
        act = [0.1 2.0; -0.1 1.0]
        allocs(env, act) = @allocated step!(env, act)
        allocs(env, act)
        if Threads.nthreads() == 1
            @test allocs(env, act) <= (Base.JLOptions().check_bounds == 1 ? 256 : 0)
        end

        @test_throws ArgumentError render!(env; mode=:rgb_array)
        # before `using GLMakie` (the example test loads it later), render! says what to do
        if Base.get_extension(F1Tenth, :F1TenthGLMakieExt) === nothing
            err = try
                render!(env)
            catch e
                e
            end
            @test err isa ErrorException && occursin("using GLMakie", err.msg)
        end
    end
end
