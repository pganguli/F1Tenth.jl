@testset "F110Env API" begin
    quiet() do
        for name in keys(F1Tenth.BUILTIN_MAPS)
            env = F110Env(map=name, num_agents=1)
            @test env.map_ext == F1Tenth.BUILTIN_MAPS[name] && env.sim.scan_simulator.map !== nothing
        end
        # f1tenth_racetracks maps by name (downloaded on first use)
        for name in keys(F1Tenth.RACETRACKS)
            yaml = F1Tenth.racetrack_yaml(name)
            @test isfile(yaml) && isfile(replace(yaml, ".yaml" => ".png"))
        end
        env = F110Env(map="Spa", num_agents=1)
        obs, = reset!(env, [0.0 0.0 0.0])   # every track's centerline starts at the origin
        @test env.map_ext == ".png" && obs.collisions == [0.0] && 0.5 < minimum(obs.scans[1])

        env = F110Env()
        @test env.map_name == "vegas" && env.num_agents == 2 && env.ego_idx == 1 && env.timestep == 0.01
        @test env.integrator == Integrator.RK4 && env.params == VehicleParams()

        # params as a dict, like Python; every key is required
        env = F110Env(map="berlin", params=merge(DEFAULT_DICT, Dict("mu" => 0.9)))
        @test env.params.mu == 0.9
        @test_throws KeyError F110Env(map="berlin", params=delete!(copy(DEFAULT_DICT), "mu"))

        env = F110Env(map="berlin", num_agents=2, ego_idx=2)
        obs, reward, done, info = reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])
        @test obs.ego_idx == 1   # as Python: the env never tells the simulator its ego index
        @test obs[:poses_x] == obs["poses_x"] == obs.poses_x == [0.0, 2.0]
        @test reward == 0.01 && !done && info.checkpoint_done == [false, false]
        @test length(obs.scans) == 2 && all(length.(obs.scans) .== F1Tenth.LIDAR_BEAMS)
        @test env.current_time == 0.01
        @test Set(keys(obs)) == Set([:ego_idx, :scans, :poses_x, :poses_y, :poses_theta, :linear_vels_x,
            :linear_vels_y, :ang_vels_z, :collisions, :lap_times, :lap_counts, :scan_times])
        snap = copy(obs)
        # the VESC engages once the ramped speed command passes vesc_min_erpm
        for _ in 1:20
            step!(env, [0.2 3.0; 0.0 1.0])
        end
        @test snap.poses_x == [0.0, 2.0] && obs.poses_x != snap.poses_x   # obs is reused, copies are not
        @test_throws ArgumentError reset!(env, [0.0 0.0 0.0])
        @test_throws DimensionMismatch step!(env, [0.0 1.0])
        @test_throws BoundsError update_params!(env, VehicleParams(); index=3)

        # scan noise: N(0, scan_noise), restarted on reset
        env = F110Env(map="berlin", num_agents=1)
        a = copy(reset!(env, [0.0 0.0 0.0])[1].scans[1])
        b = copy(reset!(env, [0.0 0.0 0.0])[1].scans[1])
        @test a == b
        no_scan_noise!(env)
        while env.obs.scan_times[1] == 0.0   # until the next sweep; still at rest at the same pose
            step!(env, [0.0 0.0])
        end
        clean = env.obs.scans[1]
        @test 0.9 < std(a - clean) / env.params.scan_noise < 1.1

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

@testset "LiDAR at scan_rate" begin
    env = F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1)
    reset!(env, [0.7 0.0 1.37])
    # 100 steps of 10 ms at 40 Hz: after the reset's scan, sweeps start every
    # 25 ms from 25 ms and take 18.75 ms, so 39 finish within the second (the
    # scans are noisy, so every new one differs)
    fresh, last = 0, copy(env.obs.scans[1])
    for _ in 1:100
        step!(env, [0.0 2.0])
        env.obs.scans[1] != last && (fresh += 1)
        last = copy(env.obs.scans[1])
    end
    @test fresh == 39
    @test env.obs.scan_times[1] ≈ 0.975   # stamped at the first beam
end

@testset "wall collisions: exact body against the map" begin
    env = F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1)
    m = env.sim.scan_simulator.map
    res = m.resolution
    # brute force: is any point of a fine grid over the body in a wall cell?
    function sampled(x, y, θ, len, wid)
        s, c = sincos(θ)
        for u in range(-len / 2, len / 2; length=150), v in range(-wid / 2, wid / 2; length=80)
            px, py = x + u * c - v * s - m.origin[1], y + u * s + v * c - m.origin[2]
            col, row = floor(Int, px / res), floor(Int, py / res)
            0 <= col < size(m.dt, 2) && 0 <= row < size(m.dt, 1) && m.dt[row+1, col+1] == 0 && return true
        end
        return false
    end
    # sampling can miss slivers thinner than its spacing (4 mm here), so the
    # exact check must agree with it on a slightly smaller and larger body
    rng = F1Tenth.Xoshiro(3)
    agree, hits = true, 0
    for _ in 1:3000
        x, y, θ = 0.7 + 3 * randn(rng), 6 * rand(rng) - 3, 2pi * rand(rng)
        hit = F1Tenth.body_hits_wall(m, x, y, θ, 0.58, 0.31)
        agree &= (!sampled(x, y, θ, 0.58 - 0.008, 0.31 - 0.008) || hit) && (sampled(x, y, θ, 0.58 + 0.008, 0.31 + 0.008) || !hit)
        hits += hit
    end
    @test agree && 100 < hits < 2900

    # all around the car: reversing into a wall is caught (the LiDAR can't see behind)
    reset!(env, [0.7 0.0 1.37])
    done = false
    for _ in 1:2000
        _, _, done, _ = step!(env, [0.0 -2.0])
        done && break
    end
    @test done && env.collisions == [1.0] && env.obs.linear_vels_x == [0.0]
end

@testset "LiDAR sweep" begin
    env = F110Env(map=EXAMPLE_MAP, map_ext=".png", num_agents=1)
    reset!(env, [0.7 0.0 1.37])
    car = env.sim.agents[1]
    sim = car.scan_simulator
    n = length(car.current_scan)
    T = F1Tenth._sweep_time(car)
    @test T ≈ 0.75 / 40   # 270° of a 25 ms revolution
    # a sweep turning in place at `rate` rad/s, poses every 1 ms
    function sweep!(rate)
        car.sweep_dt = 0.001
        car.sweep_poses = [F1Tenth.Pose(0.7, 0.0, 1.37 + rate * 0.001k) for k in 0:19]
        return copy(F1Tenth._trace_sweep!(car))
    end
    snapshot = copy(get_scan!(zeros(n), sim, F1Tenth.Pose(0.7, 0.0, 1.37)))

    # at rest a sweep is the instantaneous scan
    @test sweep!(0.0) ≈ snapshot rtol = 1e-9

    # turning, each beam matches a scan from the pose at that beam's time
    rate = 3.0
    swept = sweep!(rate)
    exact = [get_scan!(zeros(n), sim, F1Tenth.Pose(0.7, 0.0, 1.37 + rate * (i - 1) * T / (n - 1)))[i] for i in 1:n]
    # each run of beams in one 1 ms sub-step shares the pose at the run's
    # middle, so a beam is within 0.5 ms of its time: most match exactly, the
    # rest land a map cell (6.25 cm) or two away
    err = abs.(swept .- exact)
    @test count(err .< 1e-9) > 0.8n && maximum(err) <= 2 * 0.0625 + 1e-9
    @test maximum(abs.(swept .- snapshot)) > 0.05            # and the twist is there

    # without the sweep, the whole scan is taken at the first beam's pose
    car.lidar_sweep = false
    @test sweep!(rate) ≈ snapshot rtol = 1e-9
end
