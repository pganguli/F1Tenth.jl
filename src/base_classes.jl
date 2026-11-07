# RaceCar and Simulator, ported from f110_gym/envs/base_classes.py.

"""
    Integrator.RK4, Integrator.Euler

Integrator for the vehicle dynamics, as Python's `Integrator` enum.
"""
module Integrator
@enum T RK4 = 1 Euler = 2
end

const State = SVector{7,Float64}   # x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle
const Pose = SVector{3,Float64}    # x, y, yaw

@inline _pose(s::State) = Pose(s[1], s[2], s[5])

# The LiDAR's beam angles relative to the car, shared by a simulator's cars
# like the class attributes of Python's RaceCar.
struct LidarGeometry
    scan_angles::Vector{Float64}
end

LidarGeometry(num_beams::Int, fov::Float64, scan_ang_incr::Float64) =
    LidarGeometry([-fov / 2.0 + (i - 1) * scan_ang_incr for i in 1:num_beams])

# The LiDAR is a Hokuyo UST-10LX (specification C-42-04077): 1081 steps over
# 270°, 0.25° apart, swept counterclockwise. Beam directions come from an
# angle table; 14400 entries (ten per 0.25° step) keep the beams 0.25° apart
# and within 0.025° of their true direction (Python's 2000 snap them by up to
# 0.18°).
const LIDAR_BEAMS = 1081
const LIDAR_FOV = 3pi / 2
const LIDAR_THETA_DIS = 14400
_lidar_scan_simulator(num_beams=LIDAR_BEAMS, fov=LIDAR_FOV) = ScanSimulator2D(num_beams, fov; theta_dis=LIDAR_THETA_DIS)

"""
    RaceCar(params, seed; is_ego=false, time_step=0.01, num_beams=1081, fov=3π/2,
            integrator=Integrator.Euler, lidar_dist=0.0, lidar_sweep=true)

One vehicle: the vehicle model, its low-level actuator chain (command
limiter, steering servo, VESC speed loop; see [`F1Tenth.actuate!`](@ref)) and a
LiDAR scanning at `params.scan_rate`. With `lidar_sweep` each beam is taken
from the pose at its own time in the sweep, as the real sensor does;
without, the whole scan is taken from the pose at its first beam. `state` is `(x, y, steer_angle, vel, yaw_angle, yaw_rate,
slip_angle)`. Scan noise is drawn from `scan_rng`; set it to
`nothing` for noise-free scans (as `rng=None` in Python).

Cars made by a [`Simulator`](@ref) share its scan simulator and beam tables,
like the class attributes of Python's `RaceCar`.
"""
mutable struct RaceCar
    params::VehicleParams
    seed::Int
    is_ego::Bool
    time_step::Float64
    num_beams::Int
    fov::Float64
    integrator::Integrator.T
    lidar_dist::Float64
    state::State
    opp_poses::Vector{Pose}
    actuators::ActuatorState
    in_collision::Bool
    scan_rng::Union{Nothing,Xoshiro}
    scan_simulator::ScanSimulator2D
    lidar::LidarGeometry
    current_scan::Vector{Float64}   # the latest LiDAR scan, held between scans
    noise::Vector{Float64}          # scratch for the scan noise
    lidar_sweep::Bool
    clock::Float64                  # time since reset (s)
    scan_wait::Float64              # time until the next sweep starts (s)
    sweeping::Bool
    sweep_start::Float64            # clock at the sweep's first beam
    sweep_dt::Float64               # spacing of sweep_poses (s)
    sweep_poses::Vector{Pose}       # the car's poses through the sweep
    sweep_done::Bool                # a sweep finished this step: trace it
    scan_new::Bool                  # current_scan is new: add the other cars
    scan_time::Float64              # clock at the first beam of current_scan
    scan_origin::Pose               # LiDAR pose at the middle of current_scan
end

function RaceCar(params::VehicleParams, seed::Integer; is_ego::Bool=false, time_step::Real=0.01,
    num_beams::Integer=LIDAR_BEAMS, fov::Real=LIDAR_FOV, integrator::Integrator.T=Integrator.Euler,
    lidar_dist::Real=0.0, lidar_sweep::Bool=true, scan_simulator::ScanSimulator2D=_lidar_scan_simulator(num_beams, fov),
    lidar::LidarGeometry=LidarGeometry(Int(num_beams), Float64(fov), get_increment(scan_simulator)))
    return RaceCar(params, seed, is_ego, time_step, num_beams, fov, integrator, lidar_dist, zero(State), Pose[],
        ActuatorState(), false, Xoshiro(seed), scan_simulator, lidar, zeros(num_beams), zeros(num_beams), lidar_sweep, 0.0, 0.0, false, 0.0, 0.0, sizehint!(Pose[], 64), false, false, 0.0,
        zero(Pose))
end

"""
    update_params!(car::RaceCar, params)

Replace the car's parameters (a `VehicleParams` or a dict). Collisions with
walls and other cars follow a new body size (Python keeps the old one).
"""
update_params!(car::RaceCar, params) = (car.params = convert(VehicleParams, params); nothing)

"""
    set_map!(car::RaceCar, map_path, map_ext)

Set the map of the car's scan simulator (shared with its simulator's other cars).
"""
set_map!(car::RaceCar, map_path::AbstractString, map_ext::AbstractString) =
    set_map!(car.scan_simulator, map_path, map_ext)

"""
    reset!(car::RaceCar, pose)

Put the car at rest at `pose = (x, y, θ)`, reset its actuator chain, clock
and collision flag, and restart its scan noise from `seed`. With a map set,
it takes a first scan there (the car is at rest, so the sweep doesn't
matter); the next sweep starts one scan period later.
"""
function reset!(car::RaceCar, pose)
    reset!(car.actuators)
    car.in_collision = false
    car.state = State(pose[1], pose[2], 0.0, 0.0, pose[3], 0.0, 0.0)
    car.clock = car.scan_time = 0.0
    car.scan_wait = 1 / car.params.scan_rate
    car.sweeping = car.sweep_done = car.scan_new = false
    car.scan_rng = Xoshiro(car.seed)
    if car.scan_simulator.map !== nothing
        car.scan_origin = _lidar_pose(car, _pose(car.state))
        get_scan!(car.current_scan, car.scan_simulator, car.scan_origin)
        _add_scan_noise!(car)
        car.scan_new = true
    end
    return car
end

"""
    ray_cast_agents!(car, scan) -> scan

Shorten `scan` where the other cars (at `car.opp_poses`) block the beams,
seen from the LiDAR's pose at the middle of the scan.
"""
function ray_cast_agents!(car::RaceCar, scan::AbstractVector{Float64})
    pose = car.scan_origin
    for opp_pose in car.opp_poses
        ray_cast!(pose, scan, car.lidar.scan_angles, get_vertices(opp_pose, car.params.length, car.params.width))
    end
    return scan
end

function _integrate(s::State, u::SVector{2,Float64}, p::VehicleParams, dt::Float64, integrator::Integrator.T)
    if integrator == Integrator.RK4
        k1 = vehicle_dynamics_st(s, u, p)
        k2 = vehicle_dynamics_st(s + dt * (k1 / 2), u, p)
        k3 = vehicle_dynamics_st(s + dt * (k2 / 2), u, p)
        k4 = vehicle_dynamics_st(s + dt * k3, u, p)
        return s + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return s + dt * vehicle_dynamics_st(s, u, p)
end

"""
    actuate!(car::RaceCar, steer, speed)

Advance `car` one time step under the planner command `(steer, speed)`:
the command limiter ticks at `command_rate`, the VESC speed loop and the
steering servo run at `vesc_rate`, and the vehicle model is integrated over
each of those sub-steps with the car's integrator. After each sub-step the
body is checked against the walls ([`F1Tenth.body_hits_wall`](@ref)); on a
hit the car stops where it is, keeping its heading, for the rest of the
step. LiDAR sweeps start on the same sub-steps, and the car's pose is
recorded through each sweep.
"""
function actuate!(car::RaceCar, steer::Float64, speed::Float64)
    a, p = car.actuators, car.params
    a.steer_cmd, a.speed_cmd = steer, speed
    n = max(1, round(Int, car.time_step * p.vesc_rate))
    h = car.time_step / n
    force = drive_force_per_amp(p) / p.m
    s = car.state
    car.in_collision = false
    for _ in 1:n
        if a.tick_wait <= h / 2
            _limiter_tick!(a, p)
            a.tick_wait += 1 / p.command_rate
        end
        if car.scan_wait <= h / 2   # a sweep starts
            car.sweeping, car.sweep_start, car.sweep_dt = true, car.clock, h
            empty!(car.sweep_poses)
            push!(car.sweep_poses, _pose(s))
            car.scan_wait += 1 / p.scan_rate
        end
        # the VESC sees the motor speed, here the car's forward speed (no
        # wheel slip state)
        _vesc_step!(a, p.erpm_gain * s[4] * cos(s[7]), h, p)
        servo = (a.steer_out - s[3]) / p.tau_steer
        if !car.in_collision
            s = _integrate(s, SVector(servo, force * a.current), p, h, car.integrator)
            if _hits_wall(car, s)   # stop where it is, keeping the heading
                s = State(s[1], s[2], s[3], 0.0, s[5], 0.0, 0.0)
                stop_motor!(a)
                car.in_collision = true
            end
        end
        a.tick_wait -= h
        car.scan_wait -= h
        car.clock += h
        if car.sweeping
            push!(car.sweep_poses, _pose(s))
            if car.clock - car.sweep_start >= _sweep_time(car) - h / 2
                car.sweeping, car.sweep_done = false, true
            end
        end
    end
    car.state = s
    return car
end

"""
    update_pose!(car, raw_steer, vel) -> scan

Advance the car one time step under the planner command (steering angle,
speed), through its actuator chain. If a LiDAR sweep finished in the step, trace
it (see [`RaceCar`](@ref)); otherwise the previous scan is kept. The scan is
the car's own buffer.
"""
function update_pose!(car::RaceCar, raw_steer::Real, vel::Real)
    actuate!(car, Float64(raw_steer), Float64(vel))
    s = car.state
    # bound yaw angle
    yaw = s[5]
    if yaw > 2 * pi
        yaw -= 2 * pi
    elseif yaw < 0
        yaw += 2 * pi
    end
    car.state = setindex(s, yaw, 5)

    car.sweep_done || return car.current_scan
    car.sweep_done = false
    _trace_sweep!(car)
    _add_scan_noise!(car)
    car.scan_new, car.scan_time = true, car.sweep_start
    return car.current_scan
end

# Seconds the LiDAR takes to sweep its field of view.
_sweep_time(car::RaceCar) = car.fov / 2pi / car.params.scan_rate

function _add_scan_noise!(car::RaceCar)
    car.scan_rng === nothing && return car.current_scan
    noise = randn!(car.scan_rng, car.noise)   # drawn in one batch
    out, sd = car.current_scan, car.params.scan_noise
    @inbounds @simd for i in eachindex(out)
        out[i] += sd * noise[i]
    end
    return out
end

_interpolate(a::Pose, b::Pose, f) =
    Pose(a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * rem2pi(b[3] - a[3], RoundNearest))

# Trace the finished sweep into current_scan. Beams are taken counterclockwise,
# evenly over the sweep; each run of beams inside one sub-step of
# sweep_poses is traced from the pose interpolated to the run's middle.
function _trace_sweep!(car::RaceCar)
    sim, scan, poses = car.scan_simulator, car.current_scan, car.sweep_poses
    if !car.lidar_sweep
        car.scan_origin = _lidar_pose(car, poses[1])
        return get_scan!(scan, sim, car.scan_origin)
    end
    n, last_k = length(scan), length(poses) - 2
    beam_dt, h = _sweep_time(car) / (n - 1), car.sweep_dt
    substep(b) = min(floor(Int, (b - 1) * beam_dt / h), last_k)   # 0-based sub-step of beam b
    theta_dis = Float64(sim.theta_dis)
    i = 1
    while i <= n
        k = substep(i)
        j = i
        while j < n && substep(j + 1) == k
            j += 1
        end
        f = ((i + j) / 2 - 1) * beam_dt / h - k
        pose = _lidar_pose(car, _interpolate(poses[k+1], poses[k+2], f))
        i <= (n + 1) ÷ 2 <= j && (car.scan_origin = pose)
        theta_index = rem(_first_theta_index(sim, pose[3]) + (i - 1) * sim.theta_index_increment, theta_dis)
        _trace_beams!(view(scan, i:j), sim, pose[1], pose[2], theta_index)
        i = j + 1
    end
    return scan
end

"""
    update_opp_poses!(car, opp_poses)

Tell the car where the other cars are (a list of `(x, y, θ)`).
"""
function update_opp_poses!(car::RaceCar, opp_poses)
    resize!(car.opp_poses, length(opp_poses))
    for (i, p) in enumerate(opp_poses)
        car.opp_poses[i] = Pose(p[1], p[2], p[3])
    end
    return car
end

"""
    update_scan!(car, scan) -> scan

If `scan` is new, add the other cars to it.
"""
function update_scan!(car::RaceCar, scan::AbstractVector{Float64})
    car.scan_new || return scan
    car.scan_new = false
    return ray_cast_agents!(car, scan)
end

# Where the LiDAR is when the car is at `pose`.
_lidar_pose(car::RaceCar, pose) =
    Pose(pose[1] + car.lidar_dist * cos(pose[3]), pose[2] + car.lidar_dist * sin(pose[3]), pose[3])

# Whether the car's body at state `s` overlaps a wall (see body_hits_wall).
function _hits_wall(car::RaceCar, s::State)
    m = car.scan_simulator.map
    return m !== nothing && body_hits_wall(m, s[1], s[2], s[5], car.params.length, car.params.width)
end

# ---------------------------------------------------------------------------

"""
    Observation

What [`step!`](@ref) and [`reset!`](@ref) return: the entries of Python's
observation dict (`ego_idx`, `scans`, `poses_x`, `poses_y`, `poses_theta`,
`linear_vels_x`, `linear_vels_y`, `ang_vels_z`, `collisions`, `lap_times`,
`lap_counts`). Read them as `obs.poses_x`, `obs[:poses_x]` or `obs["poses_x"]`.

`scan_times` (not in Python) is the time since reset of the first beam of
each car's scan in `scans`, as ROS stamps a scan.

The arrays are reused: each step overwrites them. `copy(obs)` keeps a snapshot.
"""
struct Observation
    ego_idx::Int
    scans::Vector{Vector{Float64}}
    poses_x::Vector{Float64}
    poses_y::Vector{Float64}
    poses_theta::Vector{Float64}
    linear_vels_x::Vector{Float64}
    linear_vels_y::Vector{Float64}
    ang_vels_z::Vector{Float64}
    collisions::Vector{Float64}
    lap_times::Vector{Float64}
    lap_counts::Vector{Float64}
    scan_times::Vector{Float64}
end

Base.getindex(o::Observation, k::Symbol) = getfield(o, k)
Base.getindex(o::Observation, k::AbstractString) = getfield(o, Symbol(k))
Base.keys(::Observation) = fieldnames(Observation)
Base.haskey(::Observation, k::Union{Symbol,AbstractString}) = Symbol(k) in fieldnames(Observation)
Base.copy(o::Observation) = Observation((deepcopy(getfield(o, f)) for f in fieldnames(Observation))...)

"""
    Simulator(params, num_agents, seed; time_step=0.01, ego_idx=1,
              integrator=Integrator.RK4, lidar_dist=0.0, lidar_sweep=true)

All the cars of an environment and their interactions. Cars are numbered
from 1.
"""
mutable struct Simulator
    num_agents::Int
    seed::Int
    time_step::Float64
    ego_idx::Int
    params::VehicleParams          # as given at construction
    agent_poses::Vector{Pose}
    agents::Vector{RaceCar}
    collisions::Vector{Float64}
    collision_idx::Vector{Int}
    scan_simulator::ScanSimulator2D
    vertices::Vector{CarVertices}  # scratch for check_collision!
    obs::Observation
end

function Simulator(params, num_agents::Integer, seed::Integer; time_step::Real=0.01, ego_idx::Integer=1,
    integrator::Integrator.T=Integrator.RK4, lidar_dist::Real=0.0, lidar_sweep::Bool=true)
    params = convert(VehicleParams, params)
    n = Int(num_agents)
    scan_sim = _lidar_scan_simulator()
    lidar = LidarGeometry(LIDAR_BEAMS, LIDAR_FOV, get_increment(scan_sim))
    agents = [RaceCar(params, seed; is_ego=(i == ego_idx), time_step=time_step, integrator=integrator,
        lidar_dist=lidar_dist, lidar_sweep=lidar_sweep, scan_simulator=scan_sim, lidar=lidar) for i in 1:n]
    obs = Observation(ego_idx, [a.current_scan for a in agents], (zeros(n) for _ in 1:10)...)
    return Simulator(n, seed, time_step, ego_idx, params, zeros(Pose, n), agents, obs.collisions, fill(-1, n),
        scan_sim, Vector{CarVertices}(undef, n), obs)
end

"""
    set_map!(sim::Simulator, map_path, map_ext)

Set the map used by every car's LiDAR.
"""
set_map!(sim::Simulator, map_path::AbstractString, map_ext::AbstractString) =
    set_map!(sim.scan_simulator, map_path, map_ext)

"""
    update_params!(sim::Simulator, params; agent_idx=-1)

Update car `agent_idx`'s parameters, or every car's when `agent_idx` is negative.
"""
function update_params!(sim::Simulator, params; agent_idx::Integer=-1)
    if agent_idx < 0
        foreach(a -> update_params!(a, params), sim.agents)
    else
        checkbounds(sim.agents, agent_idx)
        update_params!(sim.agents[agent_idx], params)
    end
    return nothing
end

"""
    check_collision!(sim)

GJK check between every pair of cars.
"""
function check_collision!(sim::Simulator)
    for (i, agent) in enumerate(sim.agents)
        sim.vertices[i] = get_vertices(_pose(agent.state), agent.params.length, agent.params.width)
    end
    collision_multiple!(sim.collisions, sim.collision_idx, sim.vertices)
    return sim.collisions
end

"""
    F1Tenth.THREAD_AGENTS[] = true | false

With more than one Julia thread, step each car on its own thread (default
`true`). Set to `false` when running many simulators in parallel yourself.
"""
const THREAD_AGENTS = Ref(true)

function _update_poses!(sim::Simulator, control_inputs::AbstractMatrix)
    if THREAD_AGENTS[] && sim.num_agents > 1 && Threads.nthreads() > 1
        @sync for (i, agent) in enumerate(sim.agents)
            Threads.@spawn update_pose!(agent, control_inputs[i, 1], control_inputs[i, 2])
        end
    else
        for (i, agent) in enumerate(sim.agents)
            update_pose!(agent, control_inputs[i, 1], control_inputs[i, 2])
        end
    end
end

"""
    step!(sim::Simulator, control_inputs) -> Observation

Advance every car one step. `control_inputs` is `num_agents × 2`: desired
steering angle in column 1, desired speed in column 2.
"""
function step!(sim::Simulator, control_inputs::AbstractMatrix{<:Real})
    size(control_inputs) == (sim.num_agents, 2) ||
        throw(DimensionMismatch("control_inputs must be $(sim.num_agents)×2, got $(size(control_inputs))"))
    _update_poses!(sim, control_inputs)
    for (i, agent) in enumerate(sim.agents)
        sim.agent_poses[i] = _pose(agent.state)
    end
    check_collision!(sim)
    for (i, agent) in enumerate(sim.agents)
        # every other car's pose, without allocating
        resize!(agent.opp_poses, sim.num_agents - 1)
        for j in 1:sim.num_agents-1
            agent.opp_poses[j] = sim.agent_poses[j < i ? j : j + 1]
        end
        update_scan!(agent, agent.current_scan)
        agent.in_collision && (sim.collisions[i] = 1.0)
    end
    o = sim.obs
    for (i, agent) in enumerate(sim.agents)
        s = agent.state
        o.poses_x[i], o.poses_y[i], o.poses_theta[i] = s[1], s[2], s[5]
        o.linear_vels_x[i], o.linear_vels_y[i], o.ang_vels_z[i] = s[4], 0.0, s[6]
        o.scan_times[i] = agent.scan_time
    end
    return o
end

"""
    reset!(sim::Simulator, poses)

Put car `i` at row `i` of `poses` (`num_agents × 3`: x, y, θ).
"""
function reset!(sim::Simulator, poses::AbstractMatrix{<:Real})
    size(poses, 1) == sim.num_agents ||
        throw(ArgumentError("Number of poses for reset does not match number of agents."))
    for (i, agent) in enumerate(sim.agents)
        reset!(agent, (poses[i, 1], poses[i, 2], poses[i, 3]))
    end
    return nothing
end
