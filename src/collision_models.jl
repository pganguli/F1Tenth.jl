# The geometry of the car body and the GJK collision checks, a port of
# f110_gym/envs/collision_models.py.

const Vec2 = SVector{2,Float64}
const CarVertices = SVector{4,Vec2}

"""
    get_vertices(pose, length, width) -> SVector{4,SVector{2,Float64}}

The corners of a car body at `pose = (x, y, θ)`, in the order of the Python
code: rear-left, rear-right, front-right, front-left.
"""
@inline function get_vertices(pose, length::Real, width::Real)
    x, y, th = Float64(pose[1]), Float64(pose[2]), Float64(pose[3])
    c, s = cos(th), sin(th)
    hl, hw = length / 2, width / 2
    corner(a, b) = Vec2(c * a + -s * b + x, s * a + c * b + y)
    return CarVertices(corner(-hl, hw), corner(-hl, -hw), corner(hl, -hw), corner(hl, hw))
end

@inline _perpendicular(p::Vec2) = Vec2(p[2], -p[1])
@inline _triple_product(a::Vec2, b::Vec2, c::Vec2) = b * (a' * c) - a * (b' * c)

@inline function _avg_point(v)
    s = v[1]
    for i in 2:length(v)
        s += v[i]
    end
    return s / length(v)
end

@inline function _furthest(v, d::Vec2)
    best, bi = v[1]' * d, 1
    for i in 2:length(v)
        p = v[i]' * d
        if p > best
            best, bi = p, i
        end
    end
    return bi
end

@inline _support(v1, v2, d::Vec2) = v1[_furthest(v1, d)] - v2[_furthest(v2, -d)]

"""
    collision(vertices1, vertices2) -> Bool

A GJK overlap test between two convex polygons. Each polygon is a list of
2D points.
"""
function collision(vertices1, vertices2)
    v1 = _as_points(vertices1)
    v2 = _as_points(vertices2)
    d = _avg_point(v1) - _avg_point(v2)
    if d[1] == 0 && d[2] == 0
        d = Vec2(1.0, d[2])
    end
    a = _support(v1, v2, d)
    s1 = a                 # simplex[0]
    s2 = a                 # simplex[1]
    index = 1
    if d' * a <= 0
        return false
    end
    d = -a
    iter_count = 0
    while iter_count < 1000
        a = _support(v1, v2, d)
        index += 1
        if d' * a <= 0
            return false
        end
        ao = -a
        if index < 3
            s2 = a
            ab = s1 - a
            d = _triple_product(ab, ao, ab)
            if sqrt(d[1]^2 + d[2]^2) < 1e-10
                d = _perpendicular(ab)
            end
            continue
        end
        # simplex = (s1, s2, a): c = s1, b = s2
        ab = s2 - a
        ac = s1 - a
        acperp = _triple_product(ab, ac, ac)
        if acperp' * ao >= 0
            d = acperp
        else
            abperp = _triple_product(ac, ab, ab)
            if abperp' * ao < 0
                return true
            end
            s1 = s2
            d = abperp
        end
        s2 = a
        index -= 1
        iter_count += 1
    end
    return false
end

_as_points(v::SVector{N,Vec2}) where {N} = v
_as_points(v::AbstractVector{<:AbstractVector}) = [Vec2(p) for p in v]
_as_points(v::AbstractMatrix) = [Vec2(v[i, 1], v[i, 2]) for i in axes(v, 1)]

"""
    collision_multiple(vertices) -> (collisions::Vector{Float64}, collision_idx::Vector{Int})

GJK checks on each pair of bodies. `collisions[i]` is `1.0` if body `i`
touches a different body. `collision_idx[i]` is the last body that it hit,
or `-1`.
"""
function collision_multiple(vertices::AbstractVector)
    n = length(vertices)
    collisions = zeros(n)
    collision_idx = fill(-1, n)
    collision_multiple!(collisions, collision_idx, vertices)
    return collisions, collision_idx
end
collision_multiple(vertices::AbstractArray{<:Real,3}) =
    collision_multiple([vertices[i, :, :] for i in axes(vertices, 1)])

function collision_multiple!(collisions::AbstractVector{Float64}, collision_idx::AbstractVector{Int},
    vertices::AbstractVector)
    fill!(collisions, 0.0)
    fill!(collision_idx, -1)
    n = length(vertices)
    for i in 1:n-1, j in i+1:n
        if collision(vertices[i], vertices[j])
            collisions[i] = 1.0
            collisions[j] = 1.0
            collision_idx[i] = j
            collision_idx[j] = i
        end
    end
    return collisions, collision_idx
end
