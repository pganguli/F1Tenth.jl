# The gym environment, ported from f110_gym/envs/f110_env.py.

const MAPS_DIR = joinpath(dirname(@__DIR__), "maps")

# Python's bundled maps, by name, with their image extensions.
const BUILTIN_MAPS = Dict("berlin" => ".png", "skirk" => ".png", "levine" => ".pgm", "vegas" => ".png")

# Real-world tracks from github.com/f1tenth/f1tenth_racetracks, by name =>
# folder. Its license is GPL-3.0, so the maps are not bundled: Pkg downloads
# them (a pinned commit, see Artifacts.toml) the first time one is used.
const RACETRACKS = Dict(name => name for name in ("Austin", "BrandsHatch", "Budapest", "Catalunya",
    "Hockenheim", "IMS", "Melbourne", "Montreal", "Monza", "MoscowRaceway", "Nuerburgring", "Oschersleben",
    "Sakhir", "SaoPaulo", "Sepang", "Shanghai", "Silverstone", "Sochi", "Spa", "Spielberg", "YasMarina",
    "Zandvoort"))
RACETRACKS["MexicoCity"] = "Mexico City"

function racetrack_yaml(name::AbstractString)
    root = artifact"f1tenth_racetracks"
    return joinpath(root, only(readdir(root)), RACETRACKS[name], name * "_map.yaml")
end

# render window size
const WINDOW_W = 1000
const WINDOW_H = 800

"""
    F110Env(; seed=12345, map="vegas", map_ext=".png", params=VehicleParams(), num_agents=2,
              timestep=0.01, ego_idx=1, integrator=Integrator.RK4, lidar_dist=0.0, lidar_sweep=true)

The F1TENTH gym environment, with the keyword arguments of Python's
`gym.make('f110_gym:f110-v0', **kwargs)`:

- `seed`: seed of the LiDAR noise
- `map`: `"berlin"`, `"skirk"`, `"levine"`, `"vegas"`, a track from
  [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) such as
  `"Austin"` or `"Spa"` (downloaded on first use; see `F1Tenth.RACETRACKS`),
  or the path of a map yaml file without its extension
- `map_ext`: extension of the map image, such as `".png"` (built-in maps
  default to their own)
- `params`: a [`VehicleParams`](@ref) or a dict with all of its keys
- `num_agents`: number of cars
- `timestep`: physics time step in seconds
- `ego_idx`: index of the ego car (1-based)
- `integrator`: `Integrator.RK4` or `Integrator.Euler`
- `lidar_dist`: distance from the rear axle to the LiDAR
- `lidar_sweep` (not in Python): take each LiDAR beam from the pose at its
  own time in the sweep, as the real sensor does (`true`), or the whole scan
  from the pose at its first beam (`false`)

Drive it with [`reset!`](@ref), [`step!`](@ref) and [`render!`](@ref).
"""
mutable struct F110Env
    seed::Int
    map_name::String
    map_path::String
    map_ext::String
    params::VehicleParams
    num_agents::Int
    timestep::Float64
    ego_idx::Int
    integrator::Integrator.T
    lidar_dist::Float64
    # race state
    current_time::Float64
    collisions::Vector{Float64}
    lap_times::Vector{Float64}
    lap_counts::Vector{Float64}
    near_starts::Vector{Bool}
    toggle_list::Vector{Float64}
    checkpoint_done::Vector{Bool}
    start_xs::Vector{Float64}
    start_ys::Vector{Float64}
    start_thetas::Vector{Float64}
    start_rot::SMatrix{2,2,Float64,4}
    sim::Simulator
    obs::Observation
    # rendering
    renderer::Union{Nothing,EnvRenderer}
    render_callbacks::Vector{Any}
    wall_start::Float64   # wall-clock time matching sim time 0, for :human pacing
end

function F110Env(; seed::Integer=12345, map::Union{Nothing,AbstractString}=nothing,
    map_ext::Union{Nothing,AbstractString}=nothing, params=VehicleParams(), num_agents::Integer=2,
    timestep::Real=0.01, ego_idx::Integer=1, integrator::Integrator.T=Integrator.RK4, lidar_dist::Real=0.0,
    lidar_sweep::Bool=true)
    1 <= ego_idx <= num_agents || throw(ArgumentError("ego_idx must be in 1:$num_agents, got $ego_idx"))
    map_name = map === nothing ? "vegas" : String(map)
    if haskey(BUILTIN_MAPS, map_name)
        map_path = joinpath(MAPS_DIR, map_name * ".yaml")
        map_ext = something(map_ext, BUILTIN_MAPS[map_name])
    elseif haskey(RACETRACKS, map_name)
        map_path = racetrack_yaml(map_name)
        map_ext = something(map_ext, ".png")
    else
        map_path = map_name * ".yaml"
        map_ext = something(map_ext, ".png")
    end
    params = convert(VehicleParams, params)
    n = Int(num_agents)
    # like Python, the simulator is not told the ego index: obs.ego_idx is always 1
    sim = Simulator(params, n, seed; time_step=timestep, integrator=integrator, lidar_dist=lidar_dist,
        lidar_sweep=lidar_sweep)
    set_map!(sim, map_path, map_ext)
    s = sim.obs
    lap_times, lap_counts = zeros(n), zeros(n)
    obs = Observation(s.ego_idx, s.scans, s.poses_x, s.poses_y, s.poses_theta, s.linear_vels_x, s.linear_vels_y,
        s.ang_vels_z, s.collisions, lap_times, lap_counts, s.scan_times)
    return F110Env(seed, map_name, map_path, _normalize_ext(map_ext), params, n, timestep, ego_idx, integrator,
        lidar_dist, 0.0, s.collisions, lap_times, lap_counts, fill(true, n), zeros(n), fill(false, n), zeros(n),
        zeros(n), zeros(n), one(SMatrix{2,2,Float64}), sim, obs, nothing, Any[], NaN)
end

Base.show(io::IO, env::F110Env) = print(io, "F110Env(map=\"", env.map_name, "\", num_agents=", env.num_agents,
    ", timestep=", env.timestep, ", integrator=", env.integrator, ")")

# Lap counting: a car "toggles" each time it enters or leaves a zone around
# its start; two toggles make a lap, and four mean it has finished.
function _check_done!(env::F110Env)
    left_t, right_t = 2, 2
    o, R = env.obs, env.start_rot
    for i in 1:env.num_agents
        px = o.poses_x[i] - env.start_xs[i]
        py = o.poses_y[i] - env.start_ys[i]
        dx = R[1, 1] * px + R[1, 2] * py
        dy = R[2, 1] * px + R[2, 2] * py
        dy = dy > left_t ? dy - left_t : dy < -right_t ? -right_t - dy : 0.0
        close = dx^2 + dy^2 <= 0.1
        if close != env.near_starts[i]
            env.near_starts[i] = close
            env.toggle_list[i] += 1
        end
        env.lap_counts[i] = fld(env.toggle_list[i], 2)
        if env.toggle_list[i] < 4
            env.lap_times[i] = env.current_time
        end
        env.checkpoint_done[i] = env.toggle_list[i] >= 4
    end
    return env.collisions[env.ego_idx] != 0 || all(env.checkpoint_done)
end

"""
    step!(env::F110Env, action) -> (obs, reward, done, info)

Advance one time step. `action` is `num_agents × 2`: desired steering angle
in column 1, desired speed in column 2. `reward` is the time step; `done` is
true once the ego car crashes or every car has done two laps;
`info.checkpoint_done[i]` is true once car `i` has.

The returned [`Observation`](@ref) and `info` are overwritten by the next call.
"""
function step!(env::F110Env, action::AbstractMatrix{<:Real})
    step!(env.sim, action)
    env.current_time += env.timestep
    done = _check_done!(env)
    return env.obs, env.timestep, done, (checkpoint_done=env.checkpoint_done,)
end

"""
    reset!(env::F110Env, poses) -> (obs, reward, done, info)

Start a new episode with car `i` at row `i` of `poses` (`num_agents × 3`:
x, y, θ). As in Python, this takes one zero-input step and returns its result.
"""
function reset!(env::F110Env, poses::AbstractMatrix{<:Real})
    env.current_time = 0.0
    fill!(env.collisions, 0.0)
    fill!(env.near_starts, true)
    fill!(env.toggle_list, 0.0)
    env.start_xs .= view(poses, :, 1)
    env.start_ys .= view(poses, :, 2)
    env.start_thetas .= view(poses, :, 3)
    th = -env.start_thetas[env.ego_idx]
    env.start_rot = SMatrix{2,2,Float64}(cos(th), sin(th), -sin(th), cos(th))
    reset!(env.sim, poses)
    return step!(env, zeros(env.num_agents, 2))
end

"""
    update_map!(env, map_path, map_ext)

Switch the simulation to the map yaml at `map_path` with image extension
`map_ext`. (As in Python, an open render window keeps drawing the old map.)
"""
update_map!(env::F110Env, map_path::AbstractString, map_ext::AbstractString) =
    (set_map!(env.sim, map_path, map_ext); nothing)

"""
    update_params!(env, params; index=-1)

Replace the vehicle parameters of car `index`, or of every car when `index`
is negative. Car-to-car collisions keep the body size given at construction,
as in Python.
"""
update_params!(env::F110Env, params; index::Integer=-1) = update_params!(env.sim, params; agent_idx=index)

"""
    add_render_callback!(env, callback)

Call `callback(renderer::EnvRenderer)` on every [`render!`](@ref), before drawing.
"""
add_render_callback!(env::F110Env, callback) = (push!(env.render_callbacks, callback); nothing)

function _pace_to_real_time!(env::F110Env)
    wait = env.wall_start + env.current_time - time()
    if isnan(wait) || wait < -0.1 || wait > 0.1
        env.wall_start = time() - env.current_time
        wait = 0.0
    end
    wait >= 0.001 ? sleep(wait) : yield()   # sleep can't go below 1 ms
    return nothing
end

"""
    render!(env; mode=:human)

Draw the environment in a window (needs `using GLMakie`). Drag to pan, scroll
to zoom. `:human` paces the window to real time: each frame waits until the
wall clock catches up with the sim clock (Python just waits 5 ms per frame).
If drawing falls behind by more than 0.1 s, or after a reset, the clocks are
lined up again rather than rushed. `:human_fast` never waits.
"""
function render!(env::F110Env; mode::Union{Symbol,AbstractString}=:human)
    mode = Symbol(mode)
    mode in (:human, :human_fast) || throw(ArgumentError("mode must be :human or :human_fast, got :$mode"))
    if env.renderer === nothing
        env.renderer = _make_renderer(WINDOW_W, WINDOW_H)
        _renderer_update_map!(env.renderer, env.sim.scan_simulator.map)
    end
    # each car drawn at its own body size (Python draws a fixed 0.58 × 0.31 m)
    _renderer_update_obs!(env.renderer, env.obs, [(car.params.length, car.params.width) for car in env.sim.agents])
    foreach(callback -> callback(env.renderer), env.render_callbacks)
    _renderer_draw!(env.renderer)
    mode === :human ? _pace_to_real_time!(env) : yield()
    return nothing
end
