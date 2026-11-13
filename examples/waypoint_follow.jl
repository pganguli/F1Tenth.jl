# A pure-pursuit waypoint follower on the example map: a port of
# f1tenth_gym/examples/waypoint_follow.py.
#
#     julia --project=examples examples/waypoint_follow.jl
#
# Before the first run, make the examples environment:
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.add("GLMakie"); Pkg.instantiate()'

using F1Tenth
using GLMakie

const YAML = F1Tenth.YAML

# ---------------------------------------------------------------------------
# Planner functions. The segment indices `i` and the positions `t` are as in
# the Python code: segment i (from 0) goes from waypoint i to i + 1, and
# `i + t` is a fractional position along the trajectory.

"""
    nearest_point_on_trajectory(point, trajectory) -> (projection, dist, t, i)

The point on the polyline `trajectory` (`n × 2`) that is nearest to
`point`, its distance, and its position `t ∈ [0, 1]` along segment `i`
(from 0).
"""
function nearest_point_on_trajectory(point, trajectory)
    n = size(trajectory, 1)
    best_dist, best_i, best_t, best_proj = Inf, 0, 0.0, (0.0, 0.0)
    for i in 1:n-1
        ax, ay = trajectory[i, 1], trajectory[i, 2]
        dx, dy = trajectory[i+1, 1] - ax, trajectory[i+1, 2] - ay
        l2 = dx^2 + dy^2
        t = clamp(((point[1] - ax) * dx + (point[2] - ay) * dy) / l2, 0.0, 1.0)
        px, py = ax + t * dx, ay + t * dy
        dist = sqrt((point[1] - px)^2 + (point[2] - py)^2)
        if dist < best_dist
            best_dist, best_i, best_t, best_proj = dist, i - 1, t, (px, py)
        end
    end
    return best_proj, best_dist, best_t, best_i
end

# The intersection of the segment `start`-`stop` with the circle (point,
# radius): the two line parameters, or nothing if they do not intersect.
_dot(a, b) = a[1] * b[1] + a[2] * b[2]

function _circle_hits(start, stop, point, radius)
    V = stop .- start
    a = _dot(V, V)
    b = 2.0 * _dot(V, start .- point)
    c = _dot(start, start) + _dot(point, point) - 2.0 * _dot(start, point) - radius^2
    disc = b^2 - 4a * c
    disc < 0 && return nothing
    disc = sqrt(disc)
    return (-b - disc) / (2.0a), (-b + disc) / (2.0a), V
end

"""
    first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=false)

The first point along `trajectory`, from the fractional position `t`, at
the distance `radius` from `point`. Returns `(point, i, t)` with `i` from 0,
or `(nothing, nothing, nothing)`.
"""
function first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=false)
    n = size(trajectory, 1)
    row(k) = (trajectory[mod(k, n)+1, 1], trajectory[mod(k, n)+1, 2])
    start_i = floor(Int, t)
    start_t = t % 1.0
    for i in start_i:n-2
        start = row(i)
        stop = row(i + 1) .+ 1e-6
        hits = _circle_hits(start, stop, point, radius)
        hits === nothing && continue
        t1, t2, V = hits
        if i == start_i
            0.0 <= t1 <= 1.0 && t1 >= start_t && return start .+ t1 .* V, i, t1
            0.0 <= t2 <= 1.0 && t2 >= start_t && return start .+ t2 .* V, i, t2
        elseif 0.0 <= t1 <= 1.0
            return start .+ t1 .* V, i, t1
        elseif 0.0 <= t2 <= 1.0
            return start .+ t2 .* V, i, t2
        end
    end
    if wrap
        # continue from the start of the trajectory
        for i in -1:start_i-1
            start = row(i)
            stop = row(i + 1) .+ 1e-6
            hits = _circle_hits(start, stop, point, radius)
            hits === nothing && continue
            t1, t2, V = hits
            0.0 <= t1 <= 1.0 && return start .+ t1 .* V, i, t1
            0.0 <= t2 <= 1.0 && return start .+ t2 .* V, i, t2
        end
    end
    return nothing, nothing, nothing
end

"""
    get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase) -> (speed, steer)
"""
function get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase)
    waypoint_y = sin(-pose_theta) * (lookahead_point[1] - position[1]) +
                 cos(-pose_theta) * (lookahead_point[2] - position[2])
    speed = lookahead_point[3]
    abs(waypoint_y) < 1e-6 && return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance^2)
    steering_angle = atan(wheelbase / radius)
    return speed, steering_angle
end

"""
    PurePursuitPlanner(conf, wheelbase)

The planner of the example. `conf` is the yaml configuration of the
example, as a Dict.
"""
mutable struct PurePursuitPlanner
    wheelbase::Float64
    conf::Dict{String,Any}
    waypoints::Matrix{Float64}
    max_reacquire::Float64
    drawn_waypoints::Any
end

function PurePursuitPlanner(conf, wheelbase)
    return PurePursuitPlanner(wheelbase, conf, load_waypoints(conf), 20.0, nothing)
end

"""
    load_waypoints(conf) -> Matrix{Float64}

Read the raceline csv file that `conf` gives (with the delimiter and the
header rows from `conf`).
"""
function load_waypoints(conf)
    lines = readlines(conf["wpt_path"])[conf["wpt_rowskip"]+1:end]
    rows = [parse.(Float64, split(l, conf["wpt_delim"])) for l in lines if !isempty(strip(l))]
    return permutedims(reduce(hcat, rows))
end

_col(planner, key) = planner.conf[key] + 1   # the yaml indices start at 0

function _waypoint_xy(planner)
    w = planner.waypoints
    return hcat(w[:, _col(planner, "wpt_xind")], w[:, _col(planner, "wpt_yind")])
end

"""
    render_waypoints(planner, renderer)

Draw the raceline in the render window (a function for the render
callback).
"""
function render_waypoints(planner::PurePursuitPlanner, e)
    if planner.drawn_waypoints === nothing
        xy = _waypoint_xy(planner)
        pts = [Point2f(50.0 * xy[i, 1], 50.0 * xy[i, 2]) for i in axes(xy, 1)]
        planner.drawn_waypoints = scatter!(e.axis, pts; color=RGBf(183 / 255, 193 / 255, 222 / 255),
            markersize=2, markerspace=:pixel)
    end
    return nothing
end

function _get_current_waypoint(planner, lookahead_distance, position, _theta)
    wpts = _waypoint_xy(planner)
    vcol = _col(planner, "wpt_vind")
    _, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
    if nearest_dist < lookahead_distance
        _, i2, _ = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i + t, true)
        i2 === nothing && return nothing
        k = mod(i2, size(wpts, 1)) + 1
        return (wpts[k, 1], wpts[k, 2], planner.waypoints[i+1, vcol])
    elseif nearest_dist < planner.max_reacquire
        return (wpts[i+1, 1], wpts[i+1, 2], planner.waypoints[i+1, vcol])
    end
    return nothing
end

"""
    plan(planner, pose_x, pose_y, pose_theta, lookahead_distance, vgain) -> (speed, steer)
"""
function plan(planner::PurePursuitPlanner, pose_x, pose_y, pose_theta, lookahead_distance, vgain)
    position = (pose_x, pose_y)
    lookahead_point = _get_current_waypoint(planner, lookahead_distance, position, pose_theta)
    lookahead_point === nothing && return 4.0, 0.0
    speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, planner.wheelbase)
    return vgain * speed, steering_angle
end

# ---------------------------------------------------------------------------

# The tuning of the Python example (lookahead tlad = 0.82 m, speed gain
# vgain = 1.375) is for steering with no lag, and for linear tires that
# always have grip. With the realistic car, the servo lag and the slew limit
# of the command make the pure pursuit weave. The weave increases with the
# speed. A lookahead of 0.82 m hits a wall from vgain 0.4, 1.2 m from 0.65,
# and 1.5 m from 0.8. (At these speeds, the tires are far from their grip
# limit.) The values 1.5 m and 0.6 give some margin.
const WORK = Dict("tlad" => 1.5, "vgain" => 0.6)

function load_config()
    conf = YAML.load_file(joinpath(@__DIR__, "config_example_map.yaml"))
    conf["map_path"] = joinpath(@__DIR__, "example_map")
    conf["wpt_path"] = joinpath(@__DIR__, "example_waypoints.csv")
    return conf
end

# The pure pursuit steers with the wheelbase of the car.
const WHEELBASE = VehicleParams().lf + VehicleParams().lr

function main()
    conf = load_config()
    planner = PurePursuitPlanner(conf, WHEELBASE)

    function render_callback(e)
        # the camera follows the car
        xs = [p[1] for p in e.cars[1][]]
        ys = [p[2] for p in e.cars[1][]]
        top, bottom, left, right = maximum(ys), minimum(ys), minimum(xs), maximum(xs)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800
        render_waypoints(planner, e)
    end

    env = F110Env(map=conf["map_path"], map_ext=conf["map_ext"], num_agents=1, timestep=0.01,
        integrator=Integrator.RK4)
    add_render_callback!(env, render_callback)
    obs, step_reward, done, info = reset!(env, [conf["sx"] conf["sy"] conf["stheta"]])
    render!(env)

    laptime = 0.0
    start = time()
    while !done
        speed, steer = plan(planner, obs.poses_x[1], obs.poses_y[1], obs.poses_theta[1], WORK["tlad"], WORK["vgain"])
        obs, step_reward, done, info = step!(env, [steer speed])
        laptime += step_reward
        render!(env; mode=:human)
    end
    println("Sim elapsed time: ", laptime, " Real elapsed time: ", time() - start)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
