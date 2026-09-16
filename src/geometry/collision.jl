using StaticArrays
using LinearAlgebra: dot, normalize, hypot
import NearestNeighbors

function _sat_axes(vertices::Vector{SVector{2,Float64}})
    n = length(vertices)
    return [normalize(SVector(-(e[2]), e[1]))
            for e in (vertices[mod1(i + 1, n)] - vertices[i] for i in 1:n)]
end

function _project(vertices::Vector{SVector{2,Float64}}, axis::SVector{2,Float64})
    projections = (dot(v, axis) for v in vertices)
    return extrema(projections)
end

function collides(vertices_a::Vector{SVector{2,Float64}}, vertices_b::Vector{SVector{2,Float64}})
    for axis in vcat(_sat_axes(vertices_a), _sat_axes(vertices_b))
        min_a, max_a = _project(vertices_a, axis)
        min_b, max_b = _project(vertices_b, axis)
        if max_a < min_b || max_b < min_a
            return false
        end
    end
    return true
end

function check_car_collisions(states::Vector{State}, car::CarFootprint)
    n = length(states)
    verts = [car_vertices(s, car) for s in states]
    flags = falses(n)
    for i in 1:n-1, j in i+1:n
        if collides(verts[i], verts[j])
            flags[i] = true
            flags[j] = true
        end
    end
    return flags
end

# Standard segment/segment intersection test (both parameters in [0, 1]).
function _segments_intersect(a::Segment, b::Segment)
    r = a.p2 - a.p1
    s = b.p2 - b.p1
    rxs = r[1] * s[2] - r[2] * s[1]
    abs(rxs) < 1e-12 && return false
    qp = b.p1 - a.p1
    t = (qp[1] * s[2] - qp[2] * s[1]) / rxs
    u = (qp[1] * r[2] - qp[2] * r[1]) / rxs
    return 0 <= t <= 1 && 0 <= u <= 1
end

function check_track_collision(state::State, bvh::TrackBVH, car::CarFootprint)
    verts = car_vertices(state, car)
    query_radius = hypot(car.length, car.width) / 2 + bvh.max_half_length
    candidates = NearestNeighbors.inrange(bvh.tree, SVector(state[1], state[2]), query_radius)
    car_edges = [Segment(verts[i], verts[mod1(i + 1, 4)]) for i in 1:4]
    for idx in candidates
        seg = bvh.segments[idx]
        for edge in car_edges
            _segments_intersect(edge, seg) && return true
        end
    end
    return false
end
