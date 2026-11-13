# The 2D LiDAR simulation, a port of f110_gym/envs/laser_models.py. The map
# loading is in maps.jl, and the beam tracing is in lidar_tracing.jl. With the
# parameters of the Python code, the scans are bit-identical to its scans.

"""
    ScanSimulator2D(num_beams, fov; eps=0.0001, theta_dis=2000, max_range=30.0)

A 2D LiDAR simulator. The beam angles come from a table of `theta_dis`
angles over a full turn. Before a scan, call [`set_map!`](@ref).
"""
mutable struct ScanSimulator2D
    num_beams::Int
    fov::Float64
    eps::Float64
    theta_dis::Int
    max_range::Float64
    angle_increment::Float64
    theta_index_increment::Float64
    sines::Vector{Float64}
    cosines::Vector{Float64}
    map::Union{Nothing,MapData}
    orig_x::Float64
    orig_y::Float64
    orig_c::Float64
    orig_s::Float64
    map_height::Int
    map_width::Int
    map_resolution::Float64
    dt::Matrix{Float64}
end

function ScanSimulator2D(num_beams::Integer, fov::Real; eps::Real=0.0001, theta_dis::Integer=2000,
    max_range::Real=30.0)
    angle_increment = fov / (num_beams - 1)
    theta_index_increment = theta_dis * angle_increment / (2.0 * pi)
    # numpy.linspace(0, 2π, theta_dis): i * step, and the last value is exactly 2π
    step = 2pi / (theta_dis - 1)
    theta_arr = [(i - 1) * step + 0.0 for i in 1:theta_dis]
    theta_arr[end] = 2pi
    return ScanSimulator2D(num_beams, fov, eps, theta_dis, max_range, angle_increment, theta_index_increment,
        sin.(theta_arr), cos.(theta_arr), nothing, 0.0, 0.0, 1.0, 0.0, 0, 0, 0.0, zeros(0, 0))
end

"""
    set_map!(scan_sim, map_path, map_ext)

Load the map yaml file at `map_path`, and its image
`<map_path without extension><map_ext>`.
"""
set_map!(sim::ScanSimulator2D, map_path::AbstractString, map_ext::AbstractString) =
    set_map!(sim, load_map(map_path, map_ext))

function set_map!(sim::ScanSimulator2D, m::MapData)
    sim.map = m
    sim.dt = m.dt
    sim.map_height, sim.map_width = size(m.dt)
    sim.map_resolution = m.resolution
    sim.orig_x, sim.orig_y = m.origin[1], m.origin[2]
    sim.orig_s, sim.orig_c = sin(m.origin[3]), cos(m.origin[3])
    return true
end

"""
    get_increment(scan_sim) -> Float64

The angle between adjacent beams, in radians.
"""
get_increment(sim::ScanSimulator2D) = sim.angle_increment

# The position of the first beam in the angle table, as in get_scan of the Python code.
function _first_theta_index(sim::ScanSimulator2D, theta::Float64)
    theta_dis = Float64(sim.theta_dis)
    theta_index = sim.theta_dis * (theta - sim.fov / 2.0) / (2.0 * pi)
    theta_index = rem(theta_index, theta_dis)   # numpy.fmod
    while theta_index < 0
        theta_index += theta_dis
    end
    return theta_index
end

"""
    get_scan!(scan, sim, pose) -> scan

Write a scan with no noise from `pose = (x, y, θ)` into `scan` (length
`num_beams`).
"""
function get_scan!(scan::AbstractVector{Float64}, sim::ScanSimulator2D, pose)
    sim.map === nothing && throw(ArgumentError("Map is not set for scan simulator."))
    length(scan) == sim.num_beams || throw(DimensionMismatch("scan must have $(sim.num_beams) beams"))
    x, y, theta = Float64(pose[1]), Float64(pose[2]), Float64(pose[3])
    return _trace_beams!(scan, sim, x, y, _first_theta_index(sim, theta))
end

"""
    scan!(out, sim, pose, rng; std_dev=0.01) -> out
    scan(sim, pose, rng; std_dev=0.01) -> Vector{Float64}

A scan from `pose`, with N(0, `std_dev`) noise from `rng` added to each
beam. With `rng = nothing`, the scan has no noise.
"""
function scan!(out::AbstractVector{Float64}, sim::ScanSimulator2D, pose, rng::Union{Nothing,AbstractRNG};
    std_dev::Real=0.01)
    get_scan!(out, sim, pose)
    if rng !== nothing
        for i in eachindex(out)
            out[i] += std_dev * randn(rng)
        end
    end
    return out
end
scan(sim::ScanSimulator2D, pose, rng::Union{Nothing,AbstractRNG}; std_dev::Real=0.01) =
    scan!(Vector{Float64}(undef, sim.num_beams), sim, pose, rng; std_dev=std_dev)

# ---------------------------------------------------------------------------
# Checks on scans

"""
    check_ttc(scan, vel, scan_angles, cosines, side_distances, ttc_thresh) -> Bool

An instantaneous time-to-collision check against the environment. It is
true if the iTTC of a beam is in `[0, ttc_thresh)`. (The function does not
use `scan_angles`, as in the Python code.)
"""
function check_ttc(scan::AbstractVector{Float64}, vel::Float64, scan_angles::AbstractVector{Float64},
    cosines::AbstractVector{Float64}, side_distances::AbstractVector{Float64}, ttc_thresh::Float64)
    vel == 0.0 && return false
    @inbounds for i in eachindex(scan)
        ttc = (scan[i] - side_distances[i]) / (vel * cosines[i])
        if ttc < ttc_thresh && ttc >= 0.0
            return true
        end
    end
    return false
end

@inline _cross(v1::Vec2, v2::Vec2) = v1[1] * v2[2] - v1[2] * v2[1]

# The distance from `o` to the edge va-vb, along the beam with the normal
# `v3` = (cos(θ + π/2), sin(θ + π/2)). Inf if the beam does not hit the edge.
@inline function _get_range(o::Vec2, v3::Vec2, va::Vec2, vb::Vec2)
    v1 = o - va
    v2 = vb - va
    denom = v2' * v3
    if abs(denom) > 0.0
        d1 = _cross(v2, v1) / denom
        d2 = (v1' * v3) / denom
        return d1 >= 0.0 && d2 >= 0.0 && d2 <= 1.0 ? d1 : Inf
    elseif abs(_cross(va - o, o - vb)) < 1e-8   # beam along the edge
        return min(sqrt(sum(abs2, va - o)), sqrt(sum(abs2, vb - o)))
    end
    return Inf
end

"""
    get_range(pose, beam_theta, va, vb) -> Float64

The distance from `pose` to the edge `va`-`vb`, along the beam angle
`beam_theta` (in the world frame).
"""
get_range(pose, beam_theta::Real, va, vb) =
    _get_range(Vec2(pose[1], pose[2]), Vec2(cos(beam_theta + pi / 2.0), sin(beam_theta + pi / 2.0)), Vec2(va), Vec2(vb))

# numpy.argmin(abs(scan_angles - a)) for angles at equal intervals: go to the
# nearest index, then examine its neighbors. If two are equal, use the first,
# as argmin does.
function _nearest_angle_index(scan_angles::AbstractVector{Float64}, a::Float64)
    n = length(scan_angles)
    step = (scan_angles[n] - scan_angles[1]) / (n - 1)
    k = clamp(round(Int, (a - scan_angles[1]) / step) + 1, 1, n)
    return argmin(i -> abs(scan_angles[i] - a), max(k - 2, 1):min(k + 2, n))
end

"""
    get_blocked_view_indices(pose, vertices, scan_angles) -> (min_index, max_index)

The range of beam indices that can see the car with the corners `vertices`.
"""
function get_blocked_view_indices(pose, vertices, scan_angles::AbstractVector{Float64})
    px, py, pth = Float64(pose[1]), Float64(pose[2]), Float64(pose[3])
    ego = atan(sin(pth), cos(pth))
    lo, hi = typemax(Int), typemin(Int)
    for v in vertices
        vx, vy = v[1] - px, v[2] - py
        nrm = sqrt(vx^2 + vy^2)
        angle = ego - atan(vy / nrm, vx / nrm)
        if angle > pi
            angle -= 2 * pi
        elseif angle < -pi
            angle += 2 * pi
        end
        ind = _nearest_angle_index(scan_angles, -angle)
        lo, hi = min(lo, ind), max(hi, ind)
    end
    return lo, hi
end

"""
    ray_cast!(pose, scan, scan_angles, vertices) -> scan

Make the beams of `scan` (taken at `pose`) shorter where they hit the car
with the corners `vertices`.
"""
function ray_cast!(pose, scan::AbstractVector{Float64}, scan_angles::AbstractVector{Float64}, vertices)
    o = Vec2(pose[1], pose[2])
    th = Float64(pose[3])
    corners = (Vec2(vertices[1]), Vec2(vertices[2]), Vec2(vertices[3]), Vec2(vertices[4]), Vec2(vertices[1]))
    min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)
    @inbounds for i in min_ind:max_ind
        beam = th + scan_angles[i]
        v3 = Vec2(cos(beam + pi / 2.0), sin(beam + pi / 2.0))
        for j in 1:4
            scan_range = _get_range(o, v3, corners[j], corners[j+1])
            if scan_range < scan[i]
                scan[i] = scan_range
            end
        end
    end
    return scan
end
