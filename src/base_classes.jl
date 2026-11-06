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
    return LidarGeometry(scan_angles, cos.(scan_angles), side_distance.(scan_angles))
end

"""
    RaceCar(params, seed; is_ego=false, time_step=0.01, num_beams=1080, fov=4.7,
            integrator=Integrator.Euler, lidar_dist=0.0)

One vehicle: dynamics, a two-step steering delay, the speed/steering
controller and a LiDAR. `state` is `(x, y, steer_angle, vel, yaw_angle,
yaw_rate, slip_angle)`. Scan noise is drawn from `scan_rng`; set it to
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
    accel::Float64
    steer_angle_vel::Float64
    steer_buffer::Vector{Float64}   # newest first; first steer_buffer_len entries are used
    steer_buffer_len::Int
    steer_buffer_size::Int
    in_collision::Bool
    ttc_thresh::Float64
    scan_rng::Union{Nothing,Xoshiro}
    scan_simulator::ScanSimulator2D
    lidar::LidarGeometry
    current_scan::Vector{Float64}
    noise::Vector{Float64}          # scratch for the scan noise
end

function RaceCar(params::VehicleParams, seed::Integer; is_ego::Bool=false, time_step::Real=0.01,
    num_beams::Integer=1080, fov::Real=4.7, integrator::Integrator.T=Integrator.Euler, lidar_dist::Real=0.0,
    scan_simulator::ScanSimulator2D=ScanSimulator2D(num_beams, fov),
    lidar::LidarGeometry=LidarGeometry(params, Int(num_beams), Float64(fov), get_increment(scan_simulator)))
    steer_buffer_size = 2
    return RaceCar(params, seed, is_ego, time_step, num_beams, fov, integrator, lidar_dist, zero(State), Pose[],
        0.0, 0.0, zeros(steer_buffer_size), 0, steer_buffer_size, false, 0.005, Xoshiro(seed), scan_simulator,
        lidar, zeros(num_beams), zeros(num_beams))
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

Put the car at rest at `pose = (x, y, θ)`, clear its steering delay and
collision flag, and restart its scan noise from `seed`.
"""
function reset!(car::RaceCar, pose)
    car.accel = 0.0
    car.steer_angle_vel = 0.0
    car.in_collision = false
    car.state = State(pose[1], pose[2], 0.0, 0.0, pose[3], 0.0, 0.0)
    car.steer_buffer_len = 0
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
stopped as in Python: velocity, yaw, yaw rate and slip are all zeroed.
"""
function check_ttc!(car::RaceCar, current_scan::AbstractVector{Float64})
    g = car.lidar
    s = car.state
    car.in_collision = check_ttc(current_scan, s[4], g.scan_angles, g.cosines, g.side_distances, car.ttc_thresh)
    if car.in_collision
        car.state = State(s[1], s[2], s[3], 0.0, 0.0, 0.0, 0.0)
        car.accel = 0.0
        car.steer_angle_vel = 0.0
    end
    return car.in_collision
end

# Steering delay: push `raw_steer`, return the command from steer_buffer_size
# calls ago (0 until the buffer has filled).
function _delay_steer!(car::RaceCar, raw_steer::Float64)
    buf, n = car.steer_buffer, car.steer_buffer_len
    full = n == car.steer_buffer_size
    steer = full ? buf[n] : 0.0
    full || (n += 1)
    for j in n:-1:2
        buf[j] = buf[j-1]
    end
    buf[1] = raw_steer
    car.steer_buffer_len = n
    return steer
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
    update_pose!(car, raw_steer, vel) -> scan

Advance the car one time step towards the requested steering angle and
speed, then scan from the new pose. The scan is the car's own buffer,
overwritten on the next call.
"""
function update_pose!(car::RaceCar, raw_steer::Real, vel::Real)
    steer = _delay_steer!(car, Float64(raw_steer))
    p = car.params
    s = car.state
    accl, sv = pid(Float64(vel), steer, s[4], s[3], p.sv_max, p.a_max, p.v_max, p.v_min)
    s = _integrate(s, SVector(sv, accl), p, car.time_step, car.integrator)
    # bound yaw angle
    yaw = s[5]
    if yaw > 2 * pi
        yaw -= 2 * pi
    elseif yaw < 0
        yaw += 2 * pi
    end
    car.state = s = setindex(s, yaw, 5)

    scan_x = s[1] + car.lidar_dist * cos(s[5])
    scan_y = s[2] + car.lidar_dist * sin(s[5])
    out = get_scan!(car.current_scan, car.scan_simulator, Pose(scan_x, scan_y, s[5]))
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

Run the iTTC check on `scan`, then add the other cars to it.
"""
function update_scan!(car::RaceCar, scan::AbstractVector{Float64})
    check_ttc!(car, scan)
    return ray_cast_agents!(car, scan)
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
