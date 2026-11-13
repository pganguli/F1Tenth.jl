# Beam tracing for ScanSimulator2D: the trace_ray / get_scan loop of the
# Python code, made fast with no change to the output bits.
#
# The tracer sphere-traces each beam through the distance table. It moves
# forward by the distance to the nearest obstacle, until that distance is
# less than `eps` or the beam goes past `max_range`. Each step needs the
# result of the previous lookup. Thus with one beam at a time, the CPU waits
# for the memory. The two tracers below march many beams at the same time:
#
# - `_trace_interleaved!` keeps 16 beams active in turn (scalar code).
# - `_trace_simd!` moves 16 beams forward in each SIMD vector, with masked
#   gathers.
#
# Each beam does the arithmetic of the Python code exactly, in the same
# order. Thus the two tracers and the Python code agree bit for bit.
# `TRACER[]` selects one. `:auto` measures the time of the two, one time in
# each session, and keeps the faster one. (SIMD needs fast gathers, for
# example AVX-512.)

"""
    F1Tenth.TRACER[] = :auto | :simd | :scalar

The LiDAR tracer. The two tracers give identical scans. `:auto` (the
default) selects the faster one for this CPU when you use it for the first
time.
"""
const TRACER = Ref(:auto)
const _SIMD_FASTER = Ref{Union{Nothing,Bool}}(nothing)

const _LANES = 16   # beams in flight, for both tracers

# The distance to the nearest obstacle at a world point, as
# distance_transform() of the Python code. Points that are not on the map
# read dt[end, end] (dt[-1, -1] in numpy). With `Rotated = false` (zero yaw
# at the origin, as in all the maps of the package), the lookup does not do
# the identity rotation. The coordinates are identical.
struct DTLookup{Rotated}
    dt::Matrix{Float64}
    ox::Float64
    oy::Float64
    oc::Float64
    os::Float64
    h::Int
    w::Int
    res::Float64
    w_res::Float64
    h_res::Float64
end

function DTLookup{Rotated}(sim) where {Rotated}
    h, w, res = sim.map_height, sim.map_width, sim.map_resolution
    return DTLookup{Rotated}(sim.dt, sim.orig_x, sim.orig_y, sim.orig_c, sim.orig_s, h, w, res, w * res, h * res)
end

_is_rotated(sim) = !(sim.orig_c == 1.0 && sim.orig_s == 0.0)

@inline function (L::DTLookup{Rotated})(x::Float64, y::Float64) where {Rotated}
    x_trans = x - L.ox
    y_trans = y - L.oy
    if Rotated
        x_rot = x_trans * L.oc + y_trans * L.os
        y_rot = -x_trans * L.os + y_trans * L.oc
    else
        x_rot, y_rot = x_trans, y_trans
    end
    if x_rot < 0 || x_rot >= L.w_res || y_rot < 0 || y_rot >= L.h_res
        return @inbounds L.dt[L.h, L.w]
    end
    c = min(unsafe_trunc(Int, x_rot / L.res), L.w - 1)
    r = min(unsafe_trunc(Int, y_rot / L.res), L.h - 1)
    return @inbounds L.dt[r+1, c+1]
end

# The position of the next beam in the angle table, as the Python code
# calculates it.
@inline function _next_theta_index(theta_index::Float64, incr::Float64, theta_dis::Float64)
    theta_index += incr
    while theta_index >= theta_dis
        theta_index -= theta_dis
    end
    return theta_index
end

# Trace each beam of a scan from (x0, y0) into `scan`, from the position
# `theta_index` in the angle table. The function dispatches on the rotation
# of the map, thus Julia compiles each tracer for a concrete lookup.
function _trace_beams!(scan::AbstractVector{Float64}, sim, x0::Float64, y0::Float64, theta_index::Float64)
    if _is_rotated(sim)
        return _trace_beams!(scan, sim, DTLookup{true}(sim), x0, y0, theta_index)
    end
    return _trace_beams!(scan, sim, DTLookup{false}(sim), x0, y0, theta_index)
end

function _trace_beams!(scan, sim, lookup::DTLookup, x0, y0, theta_index)
    if _use_simd()
        return _trace_simd!(scan, sim, lookup, x0, y0, theta_index)
    end
    return _trace_interleaved!(scan, sim, lookup, x0, y0, theta_index)
end

function _use_simd()
    TRACER[] === :simd && return true
    TRACER[] === :scalar && return false
    faster = _SIMD_FASTER[]
    return faster === nothing ? (_SIMD_FASTER[] = _simd_is_faster()) : faster
end

# ---------------------------------------------------------------------------
# Scalar tracer: _LANES beams are active. When a beam is complete, its lane
# takes the next beam. Thus the CPU always has independent lookups to
# overlap.

function _trace_interleaved!(scan, sim, lookup::DTLookup, x0::Float64, y0::Float64, theta_index::Float64)
    eps, max_range = sim.eps, sim.max_range
    theta_dis = Float64(sim.theta_dis)
    incr = sim.theta_index_increment
    K = _LANES
    xs = MVector{K,Float64}(undef)
    ys = MVector{K,Float64}(undef)
    cs = MVector{K,Float64}(undef)
    ss = MVector{K,Float64}(undef)
    ds = zeros(MVector{K,Float64})     # d = 0: every lane starts out "finished"...
    tots = zeros(MVector{K,Float64})
    beams = fill(-1, MVector{K,Int})   # ...with no beam to write (-1); 0 = lane retired
    d0 = lookup(x0, y0)                # every beam starts at the scan origin
    next = 1
    in_flight = K
    @inbounds while in_flight > 0
        for k in 1:K
            b = beams[k]
            b == 0 && continue
            d, tot = ds[k], tots[k]
            if d > eps && tot <= max_range
                x = xs[k] + d * cs[k]
                y = ys[k] + d * ss[k]
                d = lookup(x, y)
                xs[k], ys[k], ds[k], tots[k] = x, y, d, tot + d
                continue
            end
            # the beam is complete: record it, and give the lane the next beam
            b > 0 && (scan[b] = tot > max_range ? max_range : tot)
            if next > length(scan)
                beams[k] = 0
                in_flight -= 1
                continue
            end
            ti = unsafe_trunc(Int, theta_index) + 1
            ss[k], cs[k] = sim.sines[ti], sim.cosines[ti]
            xs[k], ys[k], ds[k], tots[k] = x0, y0, d0, d0
            beams[k] = next
            next += 1
            theta_index = _next_theta_index(theta_index, incr, theta_dis)
        end
    end
    return scan
end

# ---------------------------------------------------------------------------
# SIMD tracer: each group of _LANES adjacent beams marches together, one
# lane for each beam, until all lanes are complete. Adjacent beams take
# similar numbers of steps, thus few lanes have no work.

function _trace_simd!(scan, sim, L::DTLookup{Rotated}, x0::Float64, y0::Float64,
    theta_index::Float64) where {Rotated}
    N = _LANES
    V, I = Vec{N,Float64}, Vec{N,Int}
    eps, max_range = sim.eps, sim.max_range
    theta_dis = Float64(sim.theta_dis)
    incr = sim.theta_index_increment
    table = vec(L.dt)
    off_map = I(L.h * L.w)   # linear index of dt[h, w]
    lane = I(ntuple(identity, Val(N)))
    d0 = L(x0, y0)
    sines = MVector{N,Float64}(undef)
    cosines = MVector{N,Float64}(undef)
    nb = length(scan)
    @inbounds for first in 1:N:nb
        n = min(N, nb - first + 1)
        for k in 1:N
            if k <= n
                ti = unsafe_trunc(Int, theta_index) + 1
                sines[k], cosines[k] = sim.sines[ti], sim.cosines[ti]
                theta_index = _next_theta_index(theta_index, incr, theta_dis)
            else
                sines[k], cosines[k] = 0.0, 0.0
            end
        end
        s, c = V(Tuple(sines)), V(Tuple(cosines))
        x, y = V(x0), V(y0)
        d = V(d0)
        tot = d
        active = (lane <= n) & (d > eps) & (tot <= max_range)
        while any(active)
            x = vifelse(active, x + d * c, x)
            y = vifelse(active, y + d * s, y)
            x_trans = x - L.ox
            y_trans = y - L.oy
            if Rotated
                x_rot = x_trans * L.oc + y_trans * L.os
                y_rot = (-x_trans) * L.os + y_trans * L.oc
            else
                x_rot, y_rot = x_trans, y_trans
            end
            on_map = (x_rot >= 0.0) & (x_rot < L.w_res) & (y_rot >= 0.0) & (y_rot < L.h_res)
            col = min(convert(I, x_rot / L.res), L.w - 1)
            row = min(convert(I, y_rot / L.res), L.h - 1)
            # each lane gets an index in the bounds, also the masked lanes
            idx = vifelse(active & on_map, row + col * L.h + 1, off_map)
            dn = vgather(table, idx, active)
            d = vifelse(active, dn, d)
            tot = vifelse(active, tot + dn, tot)
            active = active & (d > eps) & (tot <= max_range)
        end
        out = vifelse(tot > max_range, V(max_range), tot)
        for k in 1:n
            scan[first+k-1] = out[k]
        end
    end
    return scan
end

# ---------------------------------------------------------------------------

# Measure the time of the two tracers in a synthetic room with obstacles.
function _simd_is_faster()
    raw = fill(0xff, 400, 400)
    raw[[1, end], :] .= 0x00
    raw[:, [1, end]] .= 0x00
    for i in 1:40, j in 1:40
        if (7i + 13j) % 17 == 0
            raw[10i-5:10i-3, 10j-5:10j-3] .= 0x00
        end
    end
    sim = ScanSimulator2D(1080, 4.7)
    set_map!(sim, MapData(raw, distance_transform(raw, 0.05), 0.05, SVector(0.0, 0.0, 0.0)))
    lookup = DTLookup{false}(sim)
    out = zeros(sim.num_beams)
    poses = [(2.0 + 0.7k, 3.0 + 0.5k, 0.4k) for k in 1:20]
    function best_time(tracer!)
        return minimum(1:3) do _
            @elapsed for p in poses
                tracer!(out, sim, lookup, p[1], p[2], _first_theta_index(sim, p[3]))
            end
        end
    end
    best_time(_trace_interleaved!)   # compile both first
    best_time(_trace_simd!)
    return best_time(_trace_simd!) < best_time(_trace_interleaved!)
end
