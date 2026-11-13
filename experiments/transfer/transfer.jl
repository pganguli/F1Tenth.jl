# Cross-simulator transfer. This script tunes (or trains) controllers on one
# simulator, then races them without changes on the other simulators. The
# results show if the realistic car changes which controllers look best.
#
#   julia --project=experiments/transfer -e 'using Pkg; Pkg.instantiate()'
#   julia --project=experiments/transfer -t auto experiments/transfer/transfer.jl [options]
#
# Options:
#   --maps example,Spa,Monza   tracks: "example" (examples/example_map) or
#                              f1tenth_racetracks names (default: example)
#   --controllers pp,ftg,mpc,mpc-lag,rl  which controllers (default: all five)
#   --seeds 5                  seeds for each transfer race
#   --mu 0.6,0.8               add realistic-car conditions with these friction scales
#   --quick                    small grids, short training, 2 seeds: checks the pipeline
#   --out DIR                  folder for the CSV files (default: experiments/transfer/results/<time>)
#
# The strict Python port is on the archive/python-compat branch. The script
# extracts the src/ folder of that branch (with `git archive`) into .compat/
# and loads it as a second module, `Strict`, next to the `F1Tenth` module of
# this branch.

using Dates, Logging, Printf, Random, SHA, Statistics

const ROOT = dirname(dirname(@__DIR__))
const COMPAT_REF = "archive/python-compat"
const COMPAT_DIR = joinpath(@__DIR__, ".compat")

function extract_compat()
    rev = readchomp(`git -C $ROOT rev-parse $COMPAT_REF`)
    stamp = joinpath(COMPAT_DIR, "REV")
    isfile(stamp) && readchomp(stamp) == rev && return rev
    rm(COMPAT_DIR; force=true, recursive=true)
    mkpath(COMPAT_DIR)
    run(pipeline(`git -C $ROOT archive $rev src maps Artifacts.toml`, `tar -x -C $COMPAT_DIR`))
    write(stamp, rev * "\n")
    return rev
end
const COMPAT_REV = extract_compat()

import F1Tenth   # the realistic car (this branch)

module Compat
include(joinpath(@__DIR__, ".compat", "src", "F1Tenth.jl"))
end
const Strict = Compat.F1Tenth   # the strict Python port

const MAX_STEER = 0.4189   # the steering limit of both cars (rad)

# ---------------------------------------------------------------------------
# Options

function parse_args(args)
    o = Dict{String,Any}("maps" => ["example"], "controllers" => ["pp", "ftg", "mpc", "mpc-lag", "rl"], "seeds" => 5,
        "mu" => Float64[], "quick" => false, "out" => nothing)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--quick"
            o["quick"] = true
        elseif a in ("--maps", "--controllers", "--seeds", "--mu", "--out") && i < length(args)
            v = args[i+=1]
            o[a[3:end]] = a in ("--maps", "--controllers") ? String.(split(v, ',')) :
                          a == "--seeds" ? parse(Int, v) :
                          a == "--mu" ? parse.(Float64, split(v, ',')) : v
        else
            error("unknown or incomplete option: $a")
        end
        i += 1
    end
    o["quick"] && (o["seeds"] = min(o["seeds"], 2))
    return o
end

# ---------------------------------------------------------------------------
# Conditions: a simulator and the car that it simulates.

struct Condition
    name::String
    sim::Module
    params::Any
end

wheelbase(c::Condition) = c.params.lf + c.params.lr
# The steering range of the car: what the command limiter lets through, or the mechanical limits.
steer_range(c::Condition) = c.sim === F1Tenth ? (c.params.steer_cmd_min, c.params.steer_cmd_max) :
                            (c.params.s_min, c.params.s_max)

"""
NUC5 in the strict port. It has the body of the realistic car, its Magic
Formula tires made linear at zero slip (cornering stiffness `B*C*D` for each
unit of load, as CommonRoad's `C_Sf` and `C_Sr`), the peak acceleration of
its motor and the top speed of its command limiter. But it keeps the linear
tires and the actuators of Python (P speed loop, bang-bang steering,
`v_switch` power decrease). Thus it isolates the fidelity of the model from
the choice of car.
"""
function strict_nuc5()
    r = F1Tenth.VehicleParams()
    return Strict.VehicleParams(; mu=r.mu, C_Sf=r.Bf * r.Cf * r.Df, C_Sr=r.Br * r.Cr * r.Dr, lf=r.lf, lr=r.lr,
        h=r.h, m=r.m, I=r.I, s_min=r.s_min, s_max=r.s_max, sv_min=r.sv_min, sv_max=r.sv_max,
        a_max=r.current_max * F1Tenth.drive_force_per_amp(r) / r.m, v_max=r.speed_cmd_max,
        width=r.width, length=r.length)
end

function conditions(mus)
    cs = [Condition("strict", Strict, Strict.VehicleParams()),   # the default car of Python
        Condition("strict-nuc5", Strict, strict_nuc5()),
        Condition("realistic", F1Tenth, F1Tenth.VehicleParams())]
    for mu in mus
        p = F1Tenth.VehicleParams()
        fields = Dict(k => getfield(p, k) for k in fieldnames(F1Tenth.VehicleParams))
        fields[:mu] = mu
        push!(cs, Condition("realistic-mu$(mu)", F1Tenth, F1Tenth.VehicleParams(; fields...)))
    end
    return cs
end

# ---------------------------------------------------------------------------
# Tracks: a map and the path that the controllers follow. On the example map,
# the path is the raceline (it stays 0.40 m or more from the walls). The
# racelines of f1tenth_racetracks come to 0.2 m from the walls over a large
# part of the lap, which is less than half the diagonal of a car body (0.31 m
# or more): a controller that follows them closely hits the walls. Thus on
# these tracks the path is the centerline (1 m or more from the walls), at
# the raceline speed of the nearest raceline point.

struct Track
    name::String
    map::String       # the value for F110Env(map=...)
    map_ext::String
    xy::Matrix{Float64}       # the path (the raceline or the centerline)
    s::Vector{Float64}        # distance along the path (m)
    heading::Vector{Float64}  # direction of the path (rad)
    v::Vector{Float64}        # reference speed (m/s)
    length::Float64
    wall::Any                 # wall(x, y): distance to the nearest wall (m), from the map
end

function load_track(name)
    if name == "example"
        map = joinpath(ROOT, "examples", "example_map")
        csv = joinpath(ROOT, "examples", "example_waypoints.csv")
    else
        yaml = F1Tenth.racetrack_yaml(name)   # downloads the racetracks the first time
        map = name
        csv = replace(yaml, "_map.yaml" => "_raceline.csv")
    end
    readrows(f, delim) = [parse.(Float64, split(l, delim)) for l in readlines(f) if !isempty(strip(l)) && !startswith(l, '#')]
    w = permutedims(reduce(hcat, readrows(csv, ';')))   # s, x, y, psi, kappa, vx, ax
    xy, v = w[:, 2:3], w[:, 6]
    if name != "example"   # the centerline (x, y, widths), at the speed of the nearest raceline point
        c = permutedims(reduce(hcat, readrows(replace(csv, "_raceline.csv" => "_centerline.csv"), ',')))
        v = [v[argmin((xy[:, 1] .- c[i, 1]) .^ 2 .+ (xy[:, 2] .- c[i, 2]) .^ 2)] for i in axes(c, 1)]
        xy = c[:, 1:2]
    end
    n = size(xy, 1)
    # the direction from each point to the next (the files do not agree on how psi is measured)
    heading = [atan(xy[mod1(i + 1, n), 2] - xy[i, 2], xy[mod1(i + 1, n), 1] - xy[i, 1]) for i in 1:n]
    seg = [hypot(xy[mod1(i + 1, n), 1] - xy[i, 1], xy[mod1(i + 1, n), 2] - xy[i, 2]) for i in 1:n]
    s = [0.0; cumsum(seg[1:end-1])]
    len = sum(seg)
    # the distance transform of the map (the two simulators load the same map file)
    sc = with_logger(() -> F1Tenth.F110Env(; map=map, map_ext=".png", num_agents=1), NullLogger()).sim.scan_simulator
    wall = F1Tenth.DTLookup{F1Tenth._is_rotated(sc)}(sc)
    return Track(name, map, ".png", xy, s, heading, v, len, wall)
end

start_pose(t::Track, i=1) = (t.xy[i, 1], t.xy[i, 2], t.heading[i])

function nearest_index(t::Track, x, y)
    best, bi = Inf, 1
    @inbounds for i in axes(t.xy, 1)
        d = (t.xy[i, 1] - x)^2 + (t.xy[i, 2] - y)^2
        d < best && ((best, bi) = (d, i))
    end
    return bi
end

# The nearest raceline point within `window` points of `guess`, and its squared distance.
@inline function nearest_local(t::Track, x, y, guess, window)
    n = size(t.xy, 1)
    best, bi = Inf, guess
    @inbounds for k in -window:window
        i = mod1(guess + k, n)
        d = (t.xy[i, 1] - x)^2 + (t.xy[i, 2] - y)^2
        d < best && ((best, bi) = (d, i))
    end
    return bi, best
end

# The distance from raceline point i to raceline point j, forward or back (shortest way).
function progress(t::Track, i, j)
    d = t.s[j] - t.s[i]
    return d > t.length / 2 ? d - t.length : d < -t.length / 2 ? d + t.length : d
end

wrap(a) = mod(a + pi, 2pi) - pi

# ---------------------------------------------------------------------------
# Controllers. A controller is a struct of tuned values, with
#     init(ctrl, ctx, seed) -> state   (the state for one race; default: nothing)
#     act(ctrl, state, obs, ctx) -> (steering angle, speed)
# `ctx` has the track, the wheelbase and the steering range of the car, and
# the beam layout of the LiDAR. These are properties of the car that you
# measure. You do not tune them.

struct Ctx
    track::Track
    wheelbase::Float64
    steer_min::Float64
    steer_max::Float64
    num_beams::Int
    fov::Float64
end

init(_, ::Ctx, _) = nothing
stochastic(_) = false   # true if the result changes with the seed

# Grid-tuned controllers also have
#     grid_axes(C, quick) -> (values of parameter 1, values of parameter 2)
#     lower_bounds(C) -> the smallest values that a grid can extend to
#     param_names(C), params(ctrl)

"""Pure pursuit on the raceline (the planner of the example). It steers to the
raceline point `lookahead` meters away, at `vgain` times the raceline speed."""
struct PurePursuit
    lookahead::Float64
    vgain::Float64
end

params(c::PurePursuit) = (c.lookahead, c.vgain)
param_names(::Type{PurePursuit}) = ("lookahead", "vgain")
grid_axes(::Type{PurePursuit}, quick) = quick ? (0.8:0.4:2.4, 0.4:0.2:1.4) : (0.6:0.2:2.6, 0.3:0.1:1.5)
lower_bounds(::Type{PurePursuit}) = (0.2, 0.1)

function act(c::PurePursuit, _, obs, ctx::Ctx)
    t = ctx.track
    x, y, th = obs.poses_x[1], obs.poses_y[1], obs.poses_theta[1]
    i = nearest_index(t, x, y)
    n = size(t.xy, 1)
    j = i
    for k in 1:n
        j = mod1(i + k, n)
        hypot(t.xy[j, 1] - x, t.xy[j, 2] - y) >= c.lookahead && break
    end
    dx, dy = t.xy[j, 1] - x, t.xy[j, 2] - y
    lateral = -sin(th) * dx + cos(th) * dy
    steer = atan(2.0 * ctx.wheelbase * lateral / (dx^2 + dy^2))
    return steer, c.vgain * t.v[i]
end

"""Follow the gap, a reactive LiDAR controller. It removes a bubble around the
nearest obstacle and steers `gain` times the angle to the middle of the
widest gap ahead. It starts from the speed `vmax` and goes slower when it
steers more."""
struct FollowGap
    vmax::Float64
    gain::Float64
end

const FG_MAX_RANGE = 10.0   # clip ranges here (the no-return values of both LiDARs are larger)
const FG_HALF_FOV = pi / 2  # look only ahead
const FG_BUBBLE = 0.35      # m
const FG_GAP_MIN = 1.5      # a gap is a series of beams farther than this (m)
const FG_SMOOTH = 2         # half width of the moving average (beams)

stochastic(::FollowGap) = true
params(c::FollowGap) = (c.vmax, c.gain)
param_names(::Type{FollowGap}) = ("vmax", "gain")
grid_axes(::Type{FollowGap}, quick) = quick ? (2.0:1.0:7.0, 0.4:0.2:1.2) : (2.0:0.5:8.0, 0.3:0.1:1.3)
lower_bounds(::Type{FollowGap}) = (0.5, 0.1)

function act(c::FollowGap, _, obs, ctx::Ctx)
    scan = obs.scans[1]
    n, inc = ctx.num_beams, ctx.fov / (ctx.num_beams - 1)
    angle(i) = -ctx.fov / 2 + (i - 1) * inc
    lo = max(1, ceil(Int, (ctx.fov / 2 - FG_HALF_FOV) / inc) + 1)
    hi = min(n, n - lo + 1)
    # no-return values count as far, too-near values as zero
    r = [scan[i] < 0.05 ? 0.0 : min(scan[i], FG_MAX_RANGE) for i in lo:hi]
    m = length(r)
    s = [mean(@view r[max(1, k - FG_SMOOTH):min(m, k + FG_SMOOTH)]) for k in 1:m]
    # the bubble around the nearest point
    kc = argmin(s)
    half = s[kc] <= FG_BUBBLE ? pi / 4 : asin(FG_BUBBLE / s[kc])
    nb = ceil(Int, half / inc)
    s[max(1, kc - nb):min(m, kc + nb)] .= 0.0
    # the widest gap
    best_len, best_start, k = 0, 0, 1
    while k <= m
        if s[k] > FG_GAP_MIN
            k0 = k
            while k <= m && s[k] > FG_GAP_MIN
                k += 1
            end
            (k - k0) > best_len && ((best_len, best_start) = (k - k0, k0))
        else
            k += 1
        end
    end
    target = best_len == 0 ? argmax(s) : best_start + (best_len - 1) ÷ 2
    steer = clamp(c.gain * angle(lo + target - 1), -MAX_STEER, MAX_STEER)
    speed = c.vmax * max(0.35, 1.0 - 0.65 * abs(steer) / MAX_STEER)
    return steer, speed
end

"""
MPPI (model predictive path integral control), a sampling MPC. Each 0.05 s
it samples `MPPI_K` smooth steering sequences around its last plan. It
predicts each sequence over `horizon` seconds with a kinematic bicycle model
(the tires do not slip). Then it averages the samples, weighted by their
costs. The cost is the squared distance to the raceline, the heading error,
the steering changes, and the distance to the walls when it is less than
`MPPI_MARGIN` (from the map). The speed command is `vgain` times the
raceline speed. The model uses the speed command as the speed.

- `MPPI{false}` (`mpc`): the usual simple MPC model. The steering follows
  its command at once.
- `MPPI{true}` (`mpc-lag`): the model also has the published actuators of
  NUC5. The steering command moves at a maximum of `steer_rate`, and the
  servo follows it as a first-order lag with time constant `tau_steer`. The
  controller keeps an estimate of the two values from its own commands,
  because the observation does not give the steering angle.
"""
struct MPPI{Lag}
    vgain::Float64
    horizon::Float64
end

const MPPI_DT = 0.05     # plan step and replan period (s)
const MPPI_K = 96        # samples
const MPPI_SIGMA = 0.1   # steering noise of the samples (rad)
const MPPI_RHO = 0.9     # correlation of the noise from one plan step to the next
const MPPI_W_LAT = 10.0  # cost weights
const MPPI_W_HEAD = 0.5
const MPPI_W_DU = 2.0
const MPPI_W_WALL = 1000.0
const MPPI_MARGIN = 0.4  # wall distance below which the wall cost starts (m): half the car's diagonal and 0.1 m
const MPPI_TEMP = 0.2    # temperature, as a fraction of (mean cost - least cost)
const MPPI_ITERS = 2     # refinements of the plan at each replan
const NUC5 = F1Tenth.VehicleParams()   # the actuator values of MPPI{true}

mutable struct MPPIState
    rng::Xoshiro
    u::Vector{Float64}     # the plan: steering command for each plan step
    eps::Matrix{Float64}   # the noise of each sample
    cost::Vector{Float64}
    idx::Int               # raceline point nearest the car
    next_plan::Float64     # sim time of the next replan
    last_time::Float64     # sim time of the last call
    steer::Float64         # the steering command
    cmd_est::Float64       # estimates (MPPI{true}): steering command after the slew limit,
    angle_est::Float64     # and steering angle
end

stochastic(::MPPI) = true
params(c::MPPI) = (c.vgain, c.horizon)
param_names(::Type{<:MPPI}) = ("vgain", "horizon")
grid_axes(::Type{<:MPPI}, quick) = quick ? (0.4:0.2:1.4, 0.5:0.5:1.5) : (0.3:0.1:1.5, 0.5:0.25:1.5)
lower_bounds(::Type{<:MPPI}) = (0.1, 0.25)

function init(c::MPPI, ::Ctx, seed)
    H = max(2, round(Int, c.horizon / MPPI_DT))
    return MPPIState(Xoshiro(seed), zeros(H), zeros(MPPI_K, H), zeros(MPPI_K), 0, -Inf, 0.0, 0.0, 0.0, 0.0)
end

# One `dt` of the actuator model: (command after the slew limit, steering angle).
@inline function _actuators(cmd, angle, target, dt)
    cmd += clamp(target - cmd, -NUC5.steer_rate * dt, NUC5.steer_rate * dt)
    angle += (cmd - angle) * (1 - exp(-dt / NUC5.tau_steer))
    return cmd, angle
end

function act(c::MPPI{Lag}, st::MPPIState, obs, ctx::Ctx) where {Lag}
    t, L = ctx.track, ctx.wheelbase
    x, y, th = obs.poses_x[1], obs.poses_y[1], obs.poses_theta[1]
    st.idx = st.idx == 0 ? nearest_index(t, x, y) : first(nearest_local(t, x, y, st.idx, 40))
    v = c.vgain * t.v[st.idx]
    time = obs.lap_times[1]   # the sim time until the race is done
    # the actuators since the last call, with the command given then
    Lag && ((st.cmd_est, st.angle_est) = _actuators(st.cmd_est, st.angle_est, st.steer, time - st.last_time))
    st.last_time = time
    if time >= st.next_plan - 1e-9
        st.next_plan = time + MPPI_DT
        H = length(st.u)
        st.u[1:end-1] .= @view st.u[2:end]   # move the last plan one step forward
        for _ in 1:MPPI_ITERS
            for k in 1:MPPI_K
                px, py, pth, idx, prev, cost, e = x, y, th, st.idx, st.steer, 0.0, 0.0
                cmd, angle = st.cmd_est, st.angle_est
                for h in 1:H
                    # smooth noise (AR(1), standard deviation MPPI_SIGMA); sample 1 is the plan itself
                    e = k == 1 ? 0.0 : h == 1 ? MPPI_SIGMA * randn(st.rng) :
                        MPPI_RHO * e + sqrt(1 - MPPI_RHO^2) * MPPI_SIGMA * randn(st.rng)
                    st.eps[k, h] = e
                    δ = clamp(st.u[h] + e, ctx.steer_min, ctx.steer_max)
                    Lag ? ((cmd, angle) = _actuators(cmd, angle, δ, MPPI_DT)) : (angle = δ)
                    px += v * cos(pth) * MPPI_DT
                    py += v * sin(pth) * MPPI_DT
                    pth += v / L * tan(angle) * MPPI_DT
                    idx, d2 = nearest_local(t, px, py, idx, 15)
                    cost += MPPI_W_LAT * d2 + MPPI_W_HEAD * wrap(pth - t.heading[idx])^2 + MPPI_W_DU * (δ - prev)^2 +
                            MPPI_W_WALL * max(0.0, MPPI_MARGIN - t.wall(px, py))^2
                    prev = δ
                end
                st.cost[k] = cost
            end
            cmin = minimum(st.cost)
            λ = max(1e-9, MPPI_TEMP * (mean(st.cost) - cmin))
            w = exp.(-(st.cost .- cmin) ./ λ)
            w ./= sum(w)
            for h in 1:H
                st.u[h] = clamp(st.u[h] + sum(w[k] * st.eps[k, h] for k in 1:MPPI_K), ctx.steer_min, ctx.steer_max)
            end
        end
        st.steer = st.u[1]
    end
    return st.steer, v
end

"""
A linear policy on LiDAR features, trained with ARS V2 (augmented random
search with state normalization, Mania et al. 2018). The features are the
nearest range in each of `RL_BINS` sectors over the full field of view
(divided by 10 m), the speed (divided by 10 m/s) and a constant. The policy
normalizes the features (but not the constant) with the means `mu` and
standard deviations `sd` from training. The steering is
`MAX_STEER * tanh(W[1, :] ⋅ z)`. The speed is from `RL_VMIN` to `RL_VMAX`,
through a sigmoid of `W[2, :] ⋅ z`.
"""
struct LinearPolicy
    W::Matrix{Float64}
    mu::Vector{Float64}
    sd::Vector{Float64}
end

const RL_BINS = 27
const RL_NF = RL_BINS + 2
const RL_VMIN, RL_VMAX = 1.0, 10.0

stochastic(::LinearPolicy) = true

function init(::LinearPolicy, ctx::Ctx, _)
    inc = ctx.fov / (ctx.num_beams - 1)
    bins = [clamp(floor(Int, (i - 1) * inc / (ctx.fov / RL_BINS)) + 1, 1, RL_BINS) for i in 1:ctx.num_beams]
    return (bins=bins, f=zeros(RL_NF), z=zeros(RL_NF))
end

function features!(st, obs)
    f = st.f
    f[1:RL_BINS] .= 1.0
    scan = obs.scans[1]
    @inbounds for i in eachindex(st.bins)
        r = scan[i] < 0.05 ? 0.0 : min(scan[i], 10.0) / 10.0
        f[st.bins[i]] = min(f[st.bins[i]], r)
    end
    f[RL_BINS+1] = obs.linear_vels_x[1] / 10.0
    f[RL_NF] = 1.0
    return f
end

function act(c::LinearPolicy, st, obs, ::Ctx)
    f, z = features!(st, obs), st.z
    @inbounds for k in 1:RL_NF-1
        z[k] = (f[k] - c.mu[k]) / c.sd[k]
    end
    z[RL_NF] = 1.0
    z1 = sum(c.W[1, k] * z[k] for k in 1:RL_NF)
    z2 = sum(c.W[2, k] * z[k] for k in 1:RL_NF)
    return MAX_STEER * tanh(z1), RL_VMIN + (RL_VMAX - RL_VMIN) / (1 + exp(-z2))
end

const GRID_CONTROLLERS = Dict("pp" => PurePursuit, "ftg" => FollowGap, "mpc" => MPPI{false},
    "mpc-lag" => MPPI{true})

# ---------------------------------------------------------------------------
# Environments. Each thread borrows an environment from a pool and gives it
# back. To make a new map's distance transform for each race takes too long
# on the large racetrack maps.

struct Pool
    lock::ReentrantLock
    envs::Dict{String,Vector{Any}}
    track::Track
end
Pool(track) = Pool(ReentrantLock(), Dict{String,Vector{Any}}(), track)

function make_env(cond::Condition, track::Track)
    M = cond.sim
    return with_logger(NullLogger()) do
        M.F110Env(; map=track.map, map_ext=track.map_ext, params=cond.params, num_agents=1, timestep=0.01,
            integrator=M.Integrator.RK4, seed=1)
    end
end

function with_env(f, pool::Pool, cond::Condition)
    env = lock(pool.lock) do
        v = get!(pool.envs, cond.name, Any[])
        isempty(v) ? nothing : pop!(v)
    end
    env === nothing && (env = make_env(cond, pool.track))
    try
        return f(env)
    finally
        lock(() -> push!(pool.envs[cond.name], env), pool.lock)
    end
end

# Start a race: the LiDAR noise seed (reset! makes the noise generator again from it) and the pose.
function start!(cond, env, seed, pose)
    env.sim.agents[1].seed = seed
    return cond.sim.reset!(env, [pose[1] pose[2] pose[3]])
end

ctx_for(cond, env, track) =
    Ctx(track, wheelbase(cond), steer_range(cond)..., env.sim.scan_simulator.num_beams, env.sim.scan_simulator.fov)

"""
Race two laps (until the env is `done`), from the first raceline point.
Returns `(finished, crashed, race_time)`. `race_time` is NaN if the car does
not do the two laps. A race slower than 1 m/s on average is a timeout.
"""
function race(cond::Condition, pool::Pool, ctrl; seed)
    track = pool.track
    with_env(pool, cond) do env
        M = cond.sim
        ctx = ctx_for(cond, env, track)
        st = init(ctrl, ctx, seed)
        tmax = 2 * track.length / 1.0 + 10.0
        obs, _, done, info = start!(cond, env, seed, start_pose(track))
        action = zeros(1, 2)
        t = 0.0
        while !done && t < tmax
            action[1, 1], action[1, 2] = act(ctrl, st, obs, ctx)
            obs, dt, done, info = M.step!(env, action)
            t += dt
        end
        finished = info.checkpoint_done[1]
        (finished=finished, crashed=obs.collisions[1] != 0, race_time=finished ? env.lap_times[1] : NaN)
    end
end

# ---------------------------------------------------------------------------
# Grid tuning

# The score of a setting on a condition: the mean race time if all seeds
# finish, else Inf.
score(rs) = all(r -> r.finished, rs) ? mean(r.race_time for r in rs) : Inf

const MAX_EXTEND = 8   # extensions of the grid in each direction, at most

"""
Race each setting of the grid on each condition. If the best setting of a
condition is on an edge of the grid, add one more row or column of settings
past that edge (not below `lower_bounds`), and do it again. Thus no best
setting stays on an edge, unless it is at a lower bound or the grid already
extended `MAX_EXTEND` times in that direction.
"""
function tune_grid(C, conds, pool, quick)
    a1, a2 = collect.(Float64, grid_axes(C, quick))
    st1, st2 = a1[2] - a1[1], a2[2] - a2[1]
    lb1, lb2 = lower_bounds(C)
    seeds = stochastic(C(a1[1], a2[1])) ? (quick ? (1:1) : (1:3)) : (1:1)
    runs = Dict{Tuple{String,Float64,Float64},Vector{Any}}()
    extended = Dict(:lo1 => 0, :hi1 => 0, :lo2 => 0, :hi2 => 0)
    while true
        jobs = [(c, p1, p2, s) for c in conds for p1 in a1 for p2 in a2 for s in seeds
                if !haskey(runs, (c.name, p1, p2))]
        out = Vector{Any}(undef, length(jobs))
        Threads.@threads :dynamic for j in eachindex(jobs)
            c, p1, p2, s = jobs[j]
            out[j] = race(c, pool, C(p1, p2); seed=s)
        end
        for (j, (c, p1, p2, s)) in enumerate(jobs)
            push!(get!(runs, (c.name, p1, p2), Any[]), (seed=s, out[j]...))
        end
        grow = Set{Symbol}()
        for c in conds
            S = [score(runs[(c.name, p1, p2)]) for p1 in a1, p2 in a2]
            isfinite(minimum(S)) || continue
            i, j = Tuple(argmin(S))
            i == length(a1) && push!(grow, :hi1)
            i == 1 && a1[1] - st1 >= lb1 - 1e-9 && push!(grow, :lo1)
            j == length(a2) && push!(grow, :hi2)
            j == 1 && a2[1] - st2 >= lb2 - 1e-9 && push!(grow, :lo2)
        end
        filter!(d -> extended[d] < MAX_EXTEND, grow)
        isempty(grow) && break
        for d in grow
            extended[d] += 1
            d === :hi1 && push!(a1, round(a1[end] + st1; digits=6))
            d === :lo1 && pushfirst!(a1, round(a1[1] - st1; digits=6))
            d === :hi2 && push!(a2, round(a2[end] + st2; digits=6))
            d === :lo2 && pushfirst!(a2, round(a2[1] - st2; digits=6))
        end
    end
    pts = [(p1, p2) for p1 in a1 for p2 in a2]
    S = Dict(c.name => [score(runs[(c.name, p...)]) for p in pts] for c in conds)
    return pts, runs, S, extended
end

# ---------------------------------------------------------------------------
# ARS training

const RL_DIRS, RL_TOP = 16, 8   # directions for each iteration, and how many update the policy
const RL_NOISE, RL_STEP = 0.03, 0.01
const RL_EVAL_EVERY, RL_EVAL_STARTS = 10, 8   # how often to score the policy, and on how many episodes
const RL_EVAL_RACES = 2                        # two-lap races in each score (seeds 1, 2)
const RL_EPISODE = 8.0          # s
const RL_CRASH = 50.0           # penalty for a crash (m of progress)

"""
One training episode: from raceline point `i0`, drive `RL_EPISODE` seconds.
The reward is the progress along the raceline (m), less `RL_CRASH` for a
crash. Also returns the sum and the sum of squares of the features, and the
number of steps, for the state normalization.
"""
function rollout(cond, pool, pol; i0, seed)
    t = pool.track
    with_env(pool, cond) do env
        M = cond.sim
        ctx = ctx_for(cond, env, t)
        st = init(pol, ctx, seed)
        obs, _, done, _ = start!(cond, env, seed, start_pose(t, i0))
        idx, total, time, n = i0, 0.0, 0.0, 0
        s1, s2 = zeros(RL_NF), zeros(RL_NF)
        action = zeros(1, 2)
        while !done && time < RL_EPISODE
            action[1, 1], action[1, 2] = act(pol, st, obs, ctx)
            s1 .+= st.f
            s2 .+= st.f .^ 2
            n += 1
            obs, dt, done, _ = M.step!(env, action)
            time += dt
            j = first(nearest_local(t, obs.poses_x[1], obs.poses_y[1], idx, 40))
            total += progress(t, idx, j)
            idx = j
        end
        (reward=obs.collisions[1] != 0 ? total - RL_CRASH : total, s1=s1, s2=s2, n=n)
    end
end

"""
Train a `LinearPolicy` with ARS (V2-t) on one condition. Each
`RL_EVAL_EVERY` iterations, score the policy with `RL_EVAL_RACES` two-lap
races (seeds 1, 2, ...; the transfer races use other seeds) and
`RL_EVAL_STARTS` episodes from points at equal intervals along the raceline.
The score is, in this order: the number of finished races, the mean race
time (if all races finish; less is better), and the total episode reward.
Returns the policy with the best score, and the mean reward of each
iteration.
"""
function train_ars(cond, pool; seed, iters)
    rng = Xoshiro(seed)
    n = size(pool.track.xy, 1)
    W = zeros(2, RL_NF)
    mu, sd = zeros(RL_NF), ones(RL_NF)
    S1, S2, N = zeros(RL_NF), zeros(RL_NF), 0   # feature statistics of every state seen
    curve = Float64[]
    best, best_score = LinearPolicy(copy(W), copy(mu), copy(sd)), (-1, -Inf, -Inf)
    for _ in 1:iters
        δs = [randn(rng, 2, RL_NF) for _ in 1:RL_DIRS]
        starts = rand(rng, 1:n, RL_DIRS)
        noise_seeds = rand(rng, 1:10^6, RL_DIRS)
        out = Matrix{Any}(undef, RL_DIRS, 2)
        Threads.@threads :dynamic for j in 1:2RL_DIRS
            d, side = (j + 1) ÷ 2, isodd(j) ? 1 : 2
            sgn = side == 1 ? 1.0 : -1.0
            out[d, side] = rollout(cond, pool, LinearPolicy(W .+ sgn * RL_NOISE .* δs[d], mu, sd); i0=starts[d],
                seed=noise_seeds[d])
        end
        r = map(o -> o.reward, out)
        top = sortperm(vec(maximum(r; dims=2)); rev=true)[1:RL_TOP]
        σ = std(vec(r[top, :]))
        σ > 0 && (W .+= RL_STEP / (RL_TOP * σ) .* sum((r[d, 1] - r[d, 2]) .* δs[d] for d in top))
        for o in out
            S1 .+= o.s1
            S2 .+= o.s2
            N += o.n
        end
        mu = S1 ./ N
        sd = max.(sqrt.(max.(S2 ./ N .- mu .^ 2, 0.0)), 1e-3)
        push!(curve, mean(r))
        if length(curve) % RL_EVAL_EVERY == 0 || length(curve) == iters
            pol = LinearPolicy(copy(W), copy(mu), copy(sd))
            sc = zeros(RL_EVAL_STARTS)
            races = Vector{Any}(undef, RL_EVAL_RACES)
            Threads.@threads :dynamic for k in 1:RL_EVAL_STARTS+RL_EVAL_RACES
                if k <= RL_EVAL_STARTS
                    sc[k] = rollout(cond, pool, pol; i0=1 + (k - 1) * n ÷ RL_EVAL_STARTS, seed=k).reward
                else
                    races[k-RL_EVAL_STARTS] = race(cond, pool, pol; seed=k - RL_EVAL_STARTS)
                end
            end
            nfin = count(r -> r.finished, races)
            score = (nfin, nfin == RL_EVAL_RACES ? -mean(r.race_time for r in races) : -Inf, sum(sc))
            score > best_score && ((best, best_score) = (pol, score))
        end
    end
    return best, curve
end

# ---------------------------------------------------------------------------
# Statistics and output

# Spearman rank correlation, with average ranks for equal values (Inf values are equal and last).
function ranks(x)
    p = sortperm(x)
    r = similar(x, Float64)
    i = 1
    while i <= length(x)
        j = i
        while j < length(x) && x[p[j+1]] == x[p[i]]
            j += 1
        end
        r[p[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return r
end
spearman(a, b) = cor(ranks(a), ranks(b))

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        foreach(r -> println(io, join(r, ',')), rows)
    end
end

fmt_time(x) = isfinite(x) ? @sprintf("%.2f", x) : "—"

function print_transfer(conds, cands, res, label)
    println("Tuned on (row) → raced on (column): finished races, mean two-lap time (s)")
    @printf("  %-18s %-18s", "", label)
    foreach(c -> @printf("%-18s", c.name), conds)
    println()
    for a in conds
        @printf("  %-18s %-18s", a.name, cands[a.name].label)
        isempty(cands[a.name].ctrls) && (println(); continue)
        for b in conds
            rs = res[(a.name, b.name)]
            fin = [r.race_time for r in rs if r.finished]
            @printf("%-18s", "$(length(fin))/$(length(rs))  " * fmt_time(isempty(fin) ? Inf : mean(fin)))
        end
        println()
    end
end

# Race the candidates of each condition (one tuned setting, or several trained
# policies) on every condition, with `seeds`.
function transfer(conds, pool, cands, seeds)
    jobs = [(a, b, k, s) for a in conds for b in conds for k in eachindex(cands[a.name].ctrls) for s in seeds]
    out = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for j in eachindex(jobs)
        a, b, k, s = jobs[j]
        out[j] = race(b, pool, cands[a.name].ctrls[k]; seed=s)
    end
    res = Dict((a.name, b.name) => Any[] for a in conds for b in conds)
    rows = []
    for (j, (a, b, k, s)) in enumerate(jobs)
        push!(res[(a.name, b.name)], out[j])
        push!(rows, (a.name, k, b.name, s, out[j].finished, out[j].crashed, out[j].race_time))
    end
    return res, rows
end

function main(args=ARGS)
    o = parse_args(args)
    quick = o["quick"]
    conds = conditions(o["mu"])
    out = something(o["out"], joinpath(@__DIR__, "results", Dates.format(now(), "yyyymmdd-HHMMSS")))
    mkpath(out)
    # one line for each start (a restart in the same folder adds a line)
    open(joinpath(out, "provenance.txt"), "a") do io
        println(io, now(), ": F1Tenth.jl ", readchomp(`git -C $ROOT rev-parse HEAD`), ", ", COMPAT_REF, " ",
            COMPAT_REV, ", transfer.jl sha256 ", bytes2hex(open(SHA.sha256, @__FILE__)), ", julia ", VERSION, ", ",
            Threads.nthreads(), " threads, args: ", join(args, ' '))
    end
    println("Conditions: ", join((c.name for c in conds), ", "), "; ", Threads.nthreads(), " threads; output: ", out)

    # warm up the two simulators on one thread (the first use selects the LiDAR tracer)
    let pool = Pool(load_track("example"))
        foreach(c -> race(c, pool, PurePursuit(1.5, 0.3); seed=1), conds[[1, 3]])
    end

    for mapname in o["maps"]
        track = load_track(mapname)
        pool = Pool(track)
        for key in o["controllers"]
            # each map and controller has its own folder. A folder with a `done` file is complete: skip it.
            piece = joinpath(out, "$mapname-$key")
            if isfile(joinpath(piece, "done"))
                println("\n== $key on $mapname: already done (", piece, ")")
                continue
            end
            grid_rows, transfer_rows, rank_rows, curve_rows, extend_rows = [], [], [], [], []
            t0 = time()
            if haskey(GRID_CONTROLLERS, key)
                C = GRID_CONTROLLERS[key]
                pts, runs, S, extended = tune_grid(C, conds, pool, quick)
                for ((cname, p1, p2), rs) in runs, r in rs
                    push!(grid_rows, (mapname, key, cname, p1, p2, r.seed, r.finished, r.crashed, r.race_time))
                end
                push!(extend_rows, (mapname, key, extended[:lo1], extended[:hi1], extended[:lo2], extended[:hi2]))
                @printf("\n== %s on %s: %d settings × %d conditions in %.0f s (grid extended: %s)\n", key,
                    mapname, length(pts), length(conds), time() - t0,
                    join(("$k $v" for (k, v) in sort(collect(extended)) if v > 0), ", ") |> s -> isempty(s) ? "no" : s)
                println("Rank agreement over the grid (Spearman; crashes are equal and last), and settings that",
                    " finish on the row's simulator but not on the column's:")
                @printf("  %-18s", "")
                foreach(c -> @printf("%-22s", c.name), conds)
                println()
                for a in conds
                    @printf("  %-18s", a.name)
                    for b in conds
                        ok_a, ok_b = isfinite.(S[a.name]), isfinite.(S[b.name])
                        lost = count(ok_a .& .!ok_b)
                        ρ = spearman(S[a.name], S[b.name])
                        @printf("ρ=%5.2f  %3d/%-3d lost  ", ρ, lost, count(ok_a))
                        push!(rank_rows, (mapname, key, a.name, b.name, ρ, count(ok_a), lost))
                    end
                    println()
                end
                cands = Dict(map(conds) do c
                    i = argmin(S[c.name])
                    ok = isfinite(S[c.name][i])
                    c.name => (ctrls=ok ? [C(pts[i]...)] : Any[],
                        label=ok ? join((@sprintf("%.2f", v) for v in pts[i]), ", ") : "none finish")
                end)
                label = join(param_names(C), ", ")
            else
                key == "rl" || error("unknown controller: $key")
                iters, ntrain = quick ? (40, 1) : (300, 3)
                cands = Dict{String,Any}()
                for c in conds
                    pols = Any[]
                    for ts in 1:ntrain
                        pol, curve = train_ars(c, pool; seed=ts, iters=iters)
                        push!(pols, pol)
                        foreach(((it, r),) -> push!(curve_rows, (mapname, c.name, ts, it, r)), enumerate(curve))
                    end
                    cands[c.name] = (ctrls=pols, label="$ntrain policies")
                end
                @printf("\n== LinearPolicy (ARS) on %s: %d policies × %d conditions, %d iterations, in %.0f s\n",
                    mapname, ntrain, length(conds), iters, time() - t0)
                label = "trained"
            end
            seeds = 101:100+o["seeds"]
            res, rows = transfer(conds, pool, cands, seeds)
            print_transfer(conds, cands, res, label)
            for (a, k, b, s, fin, cr, rt) in rows
                tag = cands[a].label
                push!(transfer_rows, (mapname, key, a, "\"$tag\"", k, b, s, fin, cr, rt))
            end
            mkpath(piece)
            for (name, rows) in (("grid", grid_rows), ("grid_extensions", extend_rows), ("transfer", transfer_rows),
                ("rank_agreement", rank_rows), ("rl_curves", curve_rows))
                write_csv(joinpath(piece, name * ".csv"), CSV_HEADERS[name], rows)
            end
            write(joinpath(piece, "done"), "")
            flush(stdout)
        end
    end

    # all the folders together: one file of each kind in `out`
    for (name, header) in CSV_HEADERS
        open(joinpath(out, name * ".csv"), "w") do io
            println(io, join(header, ','))
            for d in sort(readdir(out; join=true))
                f = joinpath(d, name * ".csv")
                isfile(f) && foreach(l -> println(io, l), Iterators.drop(eachline(f), 1))
            end
        end
    end
    println("\nWrote the CSV files and provenance.txt to ", out)
end

const CSV_HEADERS = Dict(
    "grid" => ("map", "controller", "condition", "p1", "p2", "seed", "finished", "crashed", "race_time"),
    "grid_extensions" => ("map", "controller", "lo1", "hi1", "lo2", "hi2"),
    "transfer" => ("map", "controller", "tuned_on", "setting", "policy", "raced_on", "seed", "finished",
        "crashed", "race_time"),
    "rank_agreement" => ("map", "controller", "a", "b", "spearman", "finish_on_a", "finish_on_a_not_b"),
    "rl_curves" => ("map", "condition", "train_seed", "iteration", "mean_reward"))

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
