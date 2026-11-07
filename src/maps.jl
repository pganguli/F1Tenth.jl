# Map loading for the LiDAR: a ROS map (yaml + image) read the way Python's
# ScanSimulator2D.set_map does, plus its distance transform. Loaded maps are
# cached, so many envs on one map share one copy.

"""
    MapData

A loaded occupancy map. `raw` holds the image's gray values with row 1 at
the bottom of the picture (Python flips it the same way); `dt` is the
distance in meters from each cell to the nearest obstacle.
"""
struct MapData
    raw::Matrix{UInt8}
    dt::Matrix{Float64}
    resolution::Float64
    origin::SVector{3,Float64}   # x, y, yaw of cell (1, 1)
end

const _MAP_CACHE = Dict{Tuple{String,String,Float64,Int},MapData}()
const _MAP_CACHE_LOCK = ReentrantLock()

# Python's env documents extensions without the dot ("png") but only works with it.
_normalize_ext(ext::AbstractString) = startswith(ext, ".") ? String(ext) : "." * ext

_map_image_path(map_path, map_ext) = first(splitext(map_path)) * _normalize_ext(map_ext)

"""
    load_map(map_path, map_ext) -> MapData

Load the map yaml at `map_path` and its image `<map_path without
extension><map_ext>`. Pixels `<= 128` are obstacles. Cached by path and
modification time.
"""
function load_map(map_path::AbstractString, map_ext::AbstractString)
    yaml_path = abspath(map_path)
    img_path = abspath(_map_image_path(map_path, map_ext))
    key = (yaml_path, img_path, mtime(yaml_path) + mtime(img_path), filesize(img_path))
    return lock(_MAP_CACHE_LOCK) do
        get!(() -> _read_map(yaml_path, img_path), _MAP_CACHE, key)
    end
end

function _read_map(yaml_path::String, img_path::String)
    meta = YAML.load_file(yaml_path)
    resolution = Float64(meta["resolution"])
    origin = SVector{3,Float64}(meta["origin"][1], meta["origin"][2], meta["origin"][3])
    raw = _flipped_gray(FileIO.load(img_path))
    return MapData(raw, distance_transform(raw, resolution), resolution, origin)
end

# Gray values 0-255, flipped so row 1 is the bottom row of the picture.
# 8-bit grayscale (every bundled map) is copied byte for byte.
function _flipped_gray(img::AbstractMatrix{ColorTypes.Gray{ColorTypes.N0f8}})
    h = size(img, 1)
    bytes = reinterpret(UInt8, img)
    return [bytes[h-r+1, c] for r in 1:h, c in axes(img, 2)]
end

function _flipped_gray(img::AbstractMatrix)
    h = size(img, 1)
    to_byte(px) = round(UInt8, clamp(Float64(ColorTypes.gray(convert(ColorTypes.Gray, px))), 0, 1) * 255)
    return [to_byte(img[h-r+1, c]) for r in 1:h, c in axes(img, 2)]
end

# ---------------------------------------------------------------------------
# Car against the map

"""
    body_hits_wall(map, x, y, θ, length, width) -> Bool

Whether a car body, the `length` × `width` rectangle centered at `(x, y)`
with heading `θ`, overlaps a wall cell of `map` (pixel value `<= 128`).
Exact for the grid: the rectangle is tested against each nearby wall
cell's square. Cells off the map are free.
"""
function body_hits_wall(m::MapData, x::Real, y::Real, θ::Real, length::Real, width::Real)
    res, dt = m.resolution, m.dt
    h, w = size(dt)
    # the body in the map's frame, with cell (1, 1) at the origin
    so, co = sincos(m.origin[3])
    dx, dy = x - m.origin[1], y - m.origin[2]
    px, py = dx * co + dy * so, -dx * so + dy * co
    s, c = sincos(θ - m.origin[3])
    a, b = length / 2, width / 2
    # far from every wall: the distance map, measured between cell centers,
    # is within res * √2 of the distance from any point of a cell
    col, row = floor(Int, px / res), floor(Int, py / res)
    if 0 <= col < w && 0 <= row < h && @inbounds(dt[row+1, col+1]) - res * sqrt(2) > hypot(a, b)
        return false
    end
    # separating-axis test of each wall cell's square against the rectangle
    ext_x, ext_y = a * abs(c) + b * abs(s), a * abs(s) + b * abs(c)
    half = res / 2
    reach = half * (abs(c) + abs(s))   # the square's half-extent along the body's axes
    for col in max(floor(Int, (px - ext_x) / res), 0):min(floor(Int, (px + ext_x) / res), w - 1),
        row in max(floor(Int, (py - ext_y) / res), 0):min(floor(Int, (py + ext_y) / res), h - 1)
        @inbounds dt[row+1, col+1] == 0 || continue
        qx, qy = (col + 0.5) * res - px, (row + 0.5) * res - py
        (abs(qx) > half + ext_x || abs(qy) > half + ext_y) && continue
        (abs(qx * c + qy * s) > a + reach || abs(qy * c - qx * s) > b + reach) && continue
        return true
    end
    return false
end

# ---------------------------------------------------------------------------
# Distance transform

"""
    distance_transform(raw, resolution) -> Matrix{Float64}

Distance in meters from each cell to the nearest obstacle (value `<= 128`).
Exact Euclidean (Felzenszwalb & Huttenlocher), equal to Python's
`resolution * scipy.ndimage.distance_transform_edt(bitmap)`.
"""
function distance_transform(raw::Matrix{UInt8}, resolution::Float64)
    # squared distances: 0 at obstacles, "far" elsewhere (finite, so the
    # envelope arithmetic stays NaN-free)
    sq = [px <= 0x80 ? 0.0 : 1e20 for px in raw]
    # one pass down the columns, then transpose so the pass along rows also
    # walks contiguous memory
    _edt_columns!(sq)
    sq = permutedims(sq)
    _edt_columns!(sq)
    return resolution .* sqrt.(permutedims(sq))
end

# 1D transform of every column of `a`, in place, split across threads.
function _edt_columns!(a::Matrix{Float64})
    n, ncols = size(a)
    ntasks = clamp(ncols ÷ 16, 1, Threads.nthreads())
    chunk = cld(ncols, ntasks)
    @sync for lo in 1:chunk:ncols
        Threads.@spawn begin
            v = Vector{Int}(undef, n)
            z = Vector{Float64}(undef, n + 1)
            f = Vector{Float64}(undef, n)
            for c in lo:min(lo + chunk - 1, ncols)
                col = view(a, :, c)
                copyto!(f, col)
                _edt_1d!(col, f, v, z)
            end
        end
    end
    return a
end

# Lower envelope of parabolas: d[q] = min over p of (q - p)^2 + f[p].
# `v` and `z` are scratch (parabola vertices and their boundaries).
function _edt_1d!(d, f, v, z)
    n = length(f)
    k = 1
    @inbounds begin
        v[1] = 1
        z[1] = -Inf
        z[2] = Inf
        for q in 2:n
            fq = f[q] + q * q
            s = (fq - (f[v[k]] + v[k] * v[k])) / (2q - 2v[k])
            while s <= z[k]
                k -= 1
                s = (fq - (f[v[k]] + v[k] * v[k])) / (2q - 2v[k])
            end
            k += 1
            v[k] = q
            z[k] = s
            z[k+1] = Inf
        end
        k = 1
        for q in 1:n
            while z[k+1] < q
                k += 1
            end
            d[q] = (q - v[k])^2 + f[v[k]]
        end
    end
    return d
end
