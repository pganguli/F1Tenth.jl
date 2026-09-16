using StaticArrays
import NearestNeighbors
using Random: AbstractRNG, randn

struct LiDARConfig
    num_beams::Int
    fov::Float64
    max_range::Float64
    noise_std::Float64
end

LiDARConfig(; num_beams::Int=1080, fov::Float64=4.7, max_range::Float64=30.0,
    noise_std::Float64=0.01) = LiDARConfig(num_beams, fov, max_range, noise_std)

struct CarFootprint
    length::Float64
    width::Float64
end

CarFootprint(; length::Float64=0.58, width::Float64=0.31) = CarFootprint(length, width)

# Corner order: front-left, front-right, rear-right, rear-left (vehicle
# frame: +x forward, +y left, ψ measured counterclockwise from +x).
function car_vertices(state::State, car::CarFootprint)
    x, y, _, _, ψ, _, _ = state
    c, s = cos(ψ), sin(ψ)
    hl, hw = car.length / 2, car.width / 2
    local_corners = (SVector(hl, hw), SVector(hl, -hw), SVector(-hl, -hw), SVector(-hl, hw))
    return [SVector(x + c * p[1] - s * p[2], y + s * p[1] + c * p[2]) for p in local_corners]
end

# Standard ray/segment intersection: ray = origin + t*dir (t >= 0),
# segment = seg.p1 + u*(seg.p2 - seg.p1) (0 <= u <= 1). Returns the ray
# parameter t of the intersection, or `nothing` if none exists.
function _ray_segment_intersection(origin::SVector{2,Float64}, dir::SVector{2,Float64}, seg::Segment)
    s = seg.p2 - seg.p1
    rxs = dir[1] * s[2] - dir[2] * s[1]
    if abs(rxs) < 1e-12
        return nothing
    end
    qp = seg.p1 - origin
    t = (qp[1] * s[2] - qp[2] * s[1]) / rxs
    u = (qp[1] * dir[2] - qp[2] * dir[1]) / rxs
    return (t >= 0 && 0 <= u <= 1) ? t : nothing
end

function cast_ray(origin::SVector{2,Float64}, angle::Float64, bvh::TrackBVH,
    max_range::Float64; step::Float64=0.1)
    dir = SVector(cos(angle), sin(angle))
    margin = bvh.max_half_length + step
    best_t = max_range
    t = step
    while t <= max_range
        sample = origin + t * dir
        for i in NearestNeighbors.inrange(bvh.tree, sample, margin)
            t_hit = _ray_segment_intersection(origin, dir, bvh.segments[i])
            if t_hit !== nothing && t_hit < best_t
                best_t = t_hit
            end
        end
        t > best_t && break
        t += step
    end
    return best_t
end

function _cast_ray_against_segments(origin::SVector{2,Float64}, angle::Float64,
    segments::Vector{Segment}, max_range::Float64)
    dir = SVector(cos(angle), sin(angle))
    best = max_range
    for seg in segments
        t_hit = _ray_segment_intersection(origin, dir, seg)
        if t_hit !== nothing && t_hit < best
            best = t_hit
        end
    end
    return best
end

function _other_agent_segments(other_agents::Vector{State}, car::CarFootprint)
    segments = Segment[]
    for agent in other_agents
        verts = car_vertices(agent, car)
        for i in 1:4
            push!(segments, Segment(verts[i], verts[mod1(i + 1, 4)]))
        end
    end
    return segments
end

function scan(state::State, bvh::TrackBVH, other_agents::Vector{State},
    car::CarFootprint, cfg::LiDARConfig; rng::Union{Nothing,AbstractRNG}=nothing)
    x, y, _, _, ψ, _, _ = state
    origin = SVector(x, y)
    other_segments = _other_agent_segments(other_agents, car)
    angles = cfg.num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=cfg.num_beams)
    ranges = Vector{Float64}(undef, cfg.num_beams)
    for (i, angle) in enumerate(angles)
        t_wall = cast_ray(origin, angle, bvh, cfg.max_range)
        t_agents = _cast_ray_against_segments(origin, angle, other_segments, cfg.max_range)
        ranges[i] = min(t_wall, t_agents)
    end
    if rng !== nothing && cfg.noise_std > 0
        ranges .+= cfg.noise_std .* randn(rng, cfg.num_beams)
    end
    return ranges
end
