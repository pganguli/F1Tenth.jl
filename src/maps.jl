# Map loading for the LiDAR: a ROS map (yaml file and image), read as
# ScanSimulator2D.set_map of the Python code reads it, and its distance
# transform. The package keeps loaded maps in a cache. Thus many envs on one
# map share one copy.

"""
    MapData

A loaded occupancy map. `raw` has the gray values of the image, with row 1
at the bottom of the image (the Python code turns it over in the same way).
`dt` is the distance in meters from each cell to the nearest obstacle.
"""
struct MapData
    raw::Matrix{UInt8}
    dt::Matrix{Float64}
    resolution::Float64
    origin::SVector{3,Float64}   # x, y, yaw of cell (1, 1)
end

const _MAP_CACHE = Dict{Tuple{String,String,Float64,Int},MapData}()
const _MAP_CACHE_LOCK = ReentrantLock()

# The documentation of the Python env gives extensions without the dot
# ("png"), but the env works only with the dot.
_normalize_ext(ext::AbstractString) = startswith(ext, ".") ? String(ext) : "." * ext

_map_image_path(map_path, map_ext) = first(splitext(map_path)) * _normalize_ext(map_ext)

"""
    load_map(map_path, map_ext) -> MapData

Load the map yaml file at `map_path`, and its image `<map_path without
extension><map_ext>`. Pixels `<= 128` are obstacles. The cache uses the path
and the modification time.
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

# Gray values 0-255, turned over so that row 1 is the bottom row of the
# image. The function copies 8-bit grayscale images (all the maps of the
# package) byte for byte.
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
# The car against the map

"""
    body_hits_wall(map, x, y, θ, length, width) -> Bool

True if a car body overlaps a wall cell of `map` (pixel value `<= 128`).
The body is the `length` × `width` rectangle with its center at `(x, y)`
and the heading `θ`. The test is exact for the grid: it compares the
rectangle with the square of each wall cell near it. Cells that are not on
the map are free.
"""
function body_hits_wall(m::MapData, x::Real, y::Real, θ::Real, length::Real, width::Real)
    res, dt = m.resolution, m.dt
    h, w = size(dt)
    # the body in the frame of the map, with cell (1, 1) at the origin
    so, co = sincos(m.origin[3])
    dx, dy = x - m.origin[1], y - m.origin[2]
    px, py = dx * co + dy * so, -dx * so + dy * co
    s, c = sincos(θ - m.origin[3])
    a, b = length / 2, width / 2
    # far from all walls: the distance map (measured between cell centers)
    # is correct to res * √2 for all points of a cell
    col, row = floor(Int, px / res), floor(Int, py / res)
    if 0 <= col < w && 0 <= row < h && @inbounds(dt[row+1, col+1]) - res * sqrt(2) > hypot(a, b)
        return false
    end
    # a separating-axis test of the square of each wall cell against the rectangle
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

The distance in meters from each cell to the nearest obstacle (value
`<= 128`). It is the exact Euclidean distance (Felzenszwalb &
Huttenlocher), equal to
`resolution * scipy.ndimage.distance_transform_edt(bitmap)` in the Python
code.
"""
function distance_transform(raw::Matrix{UInt8}, resolution::Float64)
    # squared distances: 0 at obstacles, and "far" at all other cells (a
    # finite value, thus the envelope arithmetic gives no NaN)
    sq = [px <= 0x80 ? 0.0 : 1e20 for px in raw]
    # one pass down the columns, then a transpose. Thus the pass along the
    # rows also reads contiguous memory.
    _edt_columns!(sq)
    sq = permutedims(sq)
    _edt_columns!(sq)
    return resolution .* sqrt.(permutedims(sq))
end

# The 1D transform of each column of `a`, in place, divided between threads.
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

# The lower envelope of parabolas: d[q] = min over p of (q - p)^2 + f[p].
# `v` and `z` are work arrays (the parabola vertices and their boundaries).
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
