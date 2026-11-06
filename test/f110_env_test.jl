no_scan_noise!(env) = foreach(car -> car.scan_rng = nothing, env.sim.agents)

# params as Python dicts
const GYM_DICT = Dict(String(k) => getfield(GYM_PARAMS, k) for k in F1Tenth.PARAM_KEYS)

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
        # the VESC engages once the ramped speed command passes vesc_min_erpm
        for _ in 1:20
            step!(env, [0.2 3.0; 0.0 1.0])
        end
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
