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

# Per-beam tables that Python's RaceCar keeps as class attributes: beam
# angles, their cosines, and the distance from the LiDAR to the car's edge
# along each beam (for the iTTC check).
struct LidarGeometry
    scan_angles::Vector{Float64}
    cosines::Vector{Float64}
    side_distances::Vector{Float64}
    max_side::Float64   # the farthest car edge from the LiDAR
end

function LidarGeometry(params::VehicleParams, num_beams::Int, fov::Float64, scan_ang_incr::Float64)
    dist_sides = params.width / 2.0
    dist_fr = (params.lf + params.lr) / 2.0
    scan_angles = [-fov / 2.0 + (i - 1) * scan_ang_incr for i in 1:num_beams]
    side_distance(angle) =
        if angle > 0
            if angle < pi / 2   # between 0 and pi/2
                min(dist_sides / sin(angle), dist_fr / cos(angle))
            else                # between pi/2 and pi
                min(dist_sides / cos(angle - pi / 2.0), dist_fr / sin(angle - pi / 2.0))
            end
        else
            if angle > -pi / 2  # between 0 and -pi/2
                min(dist_sides / sin(-angle), dist_fr / cos(-angle))
            else                # between -pi/2 and -pi
                min(dist_sides / cos(-angle - pi / 2), dist_fr / sin(-angle - pi / 2))
            end
        end
    sides = side_distance.(scan_angles)
    return LidarGeometry(scan_angles, cos.(scan_angles), sides, maximum(sides))
end

"""
    RaceCar(params, seed; is_ego=false, time_step=0.01, num_beams=1080, fov=4.7,
            integrator=Integrator.Euler, lidar_dist=0.0)

One vehicle: the vehicle model, its low-level actuator chain (command
limiter, steering servo, VESC speed loop; see [`F1Tenth.actuate!`](@ref)) and a
LiDAR scanning at `params.scan_rate`. `state` is `(x, y, steer_angle, vel, yaw_angle, yaw_rate,
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
    ttc_thresh::Float64
    scan_rng::Union{Nothing,Xoshiro}
    scan_simulator::ScanSimulator2D
    lidar::LidarGeometry
    current_scan::Vector{Float64}   # the latest LiDAR scan, held between scans
    noise::Vector{Float64}          # scratch for the scan noise
    truth_scan::Vector{Float64}     # scratch: noise-free scan for the wall check
    scan_wait::Float64              # time until the next LiDAR scan (s)
    scan_due::Bool                  # a scan was taken this step, at scan_pose
    scan_pose::Pose
end

function RaceCar(params::VehicleParams, seed::Integer; is_ego::Bool=false, time_step::Real=0.01,
    num_beams::Integer=1080, fov::Real=4.7, integrator::Integrator.T=Integrator.Euler, lidar_dist::Real=0.0,
    scan_simulator::ScanSimulator2D=ScanSimulator2D(num_beams, fov),
    lidar::LidarGeometry=LidarGeometry(params, Int(num_beams), Float64(fov), get_increment(scan_simulator)))
    return RaceCar(params, seed, is_ego, time_step, num_beams, fov, integrator, lidar_dist, zero(State), Pose[],
        ActuatorState(), false, 0.005, Xoshiro(seed), scan_simulator, lidar, zeros(num_beams), zeros(num_beams),
        zeros(num_beams), 0.0, false, zero(Pose))
end

"""
    update_params!(car::RaceCar, params)

Replace the car's physical parameters (a `VehicleParams` or a dict).
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

Put the car at rest at `pose = (x, y, θ)`, reset its actuator chain and
collision flag, and restart its scan noise from `seed`.
"""
function reset!(car::RaceCar, pose)
    reset!(car.actuators)
    car.scan_wait, car.scan_due = 0.0, false
    car.in_collision = false
    car.state = State(pose[1], pose[2], 0.0, 0.0, pose[3], 0.0, 0.0)
    car.scan_rng = Xoshiro(car.seed)
    return car
end

"""
    ray_cast_agents!(car, scan) -> scan

Shorten `scan` where the other cars (at `car.opp_poses`) block the beams.
"""
function ray_cast_agents!(car::RaceCar, scan::AbstractVector{Float64})
    pose = _pose(car.state)
    for opp_pose in car.opp_poses
        ray_cast!(pose, scan, car.lidar.scan_angles, get_vertices(opp_pose, car.params.length, car.params.width))
    end
    return scan
end

"""
    check_ttc!(car, current_scan) -> Bool

iTTC check against the environment (not other cars). On a hit the car is
stopped as in Python (velocity, yaw, yaw rate and slip are all zeroed) and
the speed side of its actuator chain is zeroed with it.
"""
function check_ttc!(car::RaceCar, current_scan::AbstractVector{Float64})
    g = car.lidar
    s = car.state
    car.in_collision = check_ttc(current_scan, s[4], g.scan_angles, g.cosines, g.side_distances, car.ttc_thresh)
    if car.in_collision
        car.state = State(s[1], s[2], s[3], 0.0, 0.0, 0.0, 0.0)
        stop_motor!(car.actuators)
    end
    return car.in_collision
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
each of those sub-steps with the car's integrator. The LiDAR's scan times
fall on the same sub-steps; the pose at the latest one is kept for the scan.
"""
function actuate!(car::RaceCar, steer::Float64, speed::Float64)
    a, p = car.actuators, car.params
    a.steer_cmd, a.speed_cmd = steer, speed
    n = max(1, round(Int, car.time_step * p.vesc_rate))
    h = car.time_step / n
    force = drive_force_per_amp(p) / p.m
    s = car.state
    for _ in 1:n
        if a.tick_wait <= h / 2
            _limiter_tick!(a, p)
            a.tick_wait += 1 / p.command_rate
        end
        if car.scan_wait <= h / 2
            car.scan_pose, car.scan_due = _pose(s), true
            car.scan_wait += 1 / p.scan_rate
        end
        # the VESC sees the motor speed, here the car's forward speed (no
        # wheel slip state)
        _vesc_step!(a, p.erpm_gain * s[4] * cos(s[7]), h, p)
        servo = (a.steer_out - s[3]) / p.tau_steer
        s = _integrate(s, SVector(servo, force * a.current), p, h, car.integrator)
        a.tick_wait -= h
        car.scan_wait -= h
    end
    car.state = s
    return car
end

"""
    update_pose!(car, raw_steer, vel) -> scan

Advance the car one time step under the planner command (steering angle,
speed), through its actuator chain. If a LiDAR scan fell in the step, take it
from the pose at its time; otherwise the previous scan is kept. The scan is
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

    out = car.current_scan
    car.scan_due || return out
    get_scan!(out, car.scan_simulator, _lidar_pose(car, car.scan_pose))
    if car.scan_rng !== nothing
        # same as scan!(...; std_dev=0.01), with the noise drawn in one batch
        noise = randn!(car.scan_rng, car.noise)
        @inbounds @simd for i in eachindex(out)
            out[i] += 0.01 * noise[i]
        end
    end
    return out
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

Check the car against the walls (see [`F1Tenth.check_walls!`](@ref)), and if
`scan` was just taken, add the other cars to it.
"""
function update_scan!(car::RaceCar, scan::AbstractVector{Float64})
    check_walls!(car)
    car.scan_due || return scan
    car.scan_due = false
    return ray_cast_agents!(car, scan)
end

# Where the LiDAR is when the car is at `pose`.
_lidar_pose(car::RaceCar, pose) =
    Pose(pose[1] + car.lidar_dist * cos(pose[3]), pose[2] + car.lidar_dist * sin(pose[3]), pose[3])

"""
    check_walls!(car) -> Bool

Wall collision as ground truth, every step whatever the LiDAR rate: Python's
iTTC check ([`check_ttc!`](@ref)) on a noise-free scan from the car's
current pose. Every range in that scan is at least the distance map's value
at the LiDAR (the ray march starts from it), so when that clears the car's
farthest edge by more than `ttc_thresh * |speed|`, no beam can trigger and
the scan is skipped.
"""
function check_walls!(car::RaceCar)
    sim, s = car.scan_simulator, car.state
    sim.map === nothing && throw(ArgumentError("Map is not set for scan simulator."))
    pose = _lidar_pose(car, _pose(s))
    clearance = min(_distance_to_obstacle(sim, pose[1], pose[2]), sim.max_range) - car.lidar.max_side
    if clearance >= car.ttc_thresh * abs(s[4])
        car.in_collision = false
        return false
    end
    return check_ttc!(car, get_scan!(car.truth_scan, sim, pose))
end

# ---------------------------------------------------------------------------

"""
    Observation

What [`step!`](@ref) and [`reset!`](@ref) return: the entries of Python's
observation dict (`ego_idx`, `scans`, `poses_x`, `poses_y`, `poses_theta`,
`linear_vels_x`, `linear_vels_y`, `ang_vels_z`, `collisions`, `lap_times`,
`lap_counts`). Read them as `obs.poses_x`, `obs[:poses_x]` or `obs["poses_x"]`.

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
end

Base.getindex(o::Observation, k::Symbol) = getfield(o, k)
Base.getindex(o::Observation, k::AbstractString) = getfield(o, Symbol(k))
Base.keys(::Observation) = fieldnames(Observation)
Base.haskey(::Observation, k::Union{Symbol,AbstractString}) = Symbol(k) in fieldnames(Observation)
Base.copy(o::Observation) = Observation((deepcopy(getfield(o, f)) for f in fieldnames(Observation))...)

"""
    Simulator(params, num_agents, seed; time_step=0.01, ego_idx=1,
              integrator=Integrator.RK4, lidar_dist=0.0)

All the cars of an environment and their interactions. Cars are numbered
from 1.
"""
mutable struct Simulator
    num_agents::Int
    seed::Int
    time_step::Float64
    ego_idx::Int
    params::VehicleParams          # as given at construction; car-car collisions use its body size
    agent_poses::Vector{Pose}
    agents::Vector{RaceCar}
    collisions::Vector{Float64}
    collision_idx::Vector{Int}
    scan_simulator::ScanSimulator2D
    vertices::Vector{CarVertices}  # scratch for check_collision!
    obs::Observation
end

function Simulator(params, num_agents::Integer, seed::Integer; time_step::Real=0.01, ego_idx::Integer=1,
    integrator::Integrator.T=Integrator.RK4, lidar_dist::Real=0.0)
    params = convert(VehicleParams, params)
    n = Int(num_agents)
    scan_sim = ScanSimulator2D(1080, 4.7)
    lidar = LidarGeometry(params, 1080, 4.7, get_increment(scan_sim))
    agents = [RaceCar(params, seed; is_ego=(i == ego_idx), time_step=time_step, integrator=integrator,
        lidar_dist=lidar_dist, scan_simulator=scan_sim, lidar=lidar) for i in 1:n]
    obs = Observation(ego_idx, [a.current_scan for a in agents], (zeros(n) for _ in 1:9)...)
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
        sim.vertices[i] = get_vertices(_pose(agent.state), sim.params.length, sim.params.width)
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
