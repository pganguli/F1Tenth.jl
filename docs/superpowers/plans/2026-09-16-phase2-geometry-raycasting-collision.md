# Phase 2: Geometry, Raycasting & Collision Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a track/map representation, a KD-tree-accelerated 2D LiDAR
raycaster, and SAT/segment-intersection collision detection to
`F1Tenth.jl`, wired into a multi-agent per-tick step function — without
modifying anything under `src/dynamics/`.

**Architecture:** A `Track` is built from a centerline with per-point
left/right width, offset into inner/outer boundary polylines and
flattened into a static `Vector{Segment}`. `NearestNeighbors.jl`'s
`KDTree` indexes those segments' midpoints for broad-phase pruning; a
sample-marched raycaster does exact ray-segment intersection against
the pruned candidates. Vehicle bodies are 4-vertex rectangles (from
pose + a new `CarFootprint`); car-vs-car collision is SAT, car-vs-track
collision reuses the same KD-tree broad-phase with exact
segment-segment intersection. `step_multi_agent!` wraps Phase 1's
`step_dynamics` per agent, then computes scans and collision flags for
the whole agent list. `to_lazysets` is a thin, separate export of the
track's convex slices as `LazySets.VPolygon`s for Phase 5.

**Tech Stack:** Julia 1.13 (repo compat floor 1.10.10),
NearestNeighbors.jl (new), LazySets.jl (new), StaticArrays.jl,
LinearAlgebra/DelimitedFiles/Random (stdlib), Test (stdlib).

**Spec:** `docs/superpowers/specs/2026-09-16-phase2-geometry-raycasting-collision-design.md`

## Global Constraints

- No Python API compatibility is required or attempted — only
  behavioral fidelity where the spec calls for it (per project-wide
  instruction; this phase explicitly departs from the Python
  reference's grid/EDT raycasting and GJK collision algorithms — see
  spec's Context section).
- **No file under `src/dynamics/` may be modified.** All Phase 2 code
  lives under `src/geometry/`.
- Tracks are **closed loops** (typical racetrack topology). Open/non-
  looping tracks are out of scope for Phase 2.
- `VehicleParams` (Phase 1) has no car-body-footprint fields (only
  axle distances `lᶠ`/`lʳ`, not physical length/width). Phase 2
  introduces its own `CarFootprint` struct for collision/raycasting
  geometry, kept separate from `VehicleParams`.
- Default `CarFootprint`: `length=0.58, width=0.31` (F1TENTH reference
  vehicle footprint, from Python `f1tenth_gym`'s default vehicle
  params — a starting value, not a fidelity claim).
- Default `LiDARConfig`: `num_beams=1080, fov=4.7, max_range=30.0,
  noise_std=0.01` (Python reference defaults, same caveat).
- Geometry files are `include`d directly into the `F1Tenth` module
  (no submodule), matching Phase 1's convention — call sites use
  `F1Tenth.Track`, not `F1Tenth.Geometry.Track`.

---

### Task 1: `Segment`/`Track` types, CSV loading, boundary derivation

**Files:**
- Create: `src/geometry/track.jl`
- Modify: `Project.toml` (add `DelimitedFiles`, `LinearAlgebra` stdlib deps)
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/track_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `StaticArrays.SVector` (already a dependency).
- Produces: `struct Segment` with fields `p1::SVector{2,Float64},
  p2::SVector{2,Float64}`.
- Produces: `struct Track` with fields `centerline::Vector{SVector{2,Float64}},
  w_left::Vector{Float64}, w_right::Vector{Float64},
  inner::Vector{SVector{2,Float64}}, outer::Vector{SVector{2,Float64}},
  segments::Vector{Segment}`.
- Produces: `build_track(centerline::Vector{SVector{2,Float64}},
  w_left::Vector{Float64}, w_right::Vector{Float64}) -> Track`.
- Produces: `load_track(path::AbstractString) -> Track`, reading a
  comma-separated file with columns `x, y, w_left, w_right` (an
  optional non-numeric header row is skipped automatically).

- [ ] **Step 1: Add stdlib dependencies**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add(["DelimitedFiles", "LinearAlgebra"])'
```
Expected: `Project.toml`/`Manifest.toml` gain entries for both.

- [ ] **Step 2: Write the failing test**

Create `test/geometry/track_test.jl`:

```julia
using F1Tenth
using StaticArrays
using LinearAlgebra: norm
using Test

@testset "build_track: segment count and boundary offsets on a large circle" begin
    # Large radius so polygon-tangent curvature error is negligible —
    # isolates the offset math from discretization error.
    R = 100.0
    n = 360
    w = 1.5
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    w_left = fill(w, n)
    w_right = fill(w, n)

    track = F1Tenth.build_track(centerline, w_left, w_right)

    @test length(track.segments) == 2n
    @test length(track.inner) == n
    @test length(track.outer) == n

    # Every outer point should sit ~w further from the origin than the
    # centerline point at the same index; every inner point ~w closer.
    for i in 1:n
        @test isapprox(norm(track.outer[i]), R + w; atol=1e-2)
        @test isapprox(norm(track.inner[i]), R - w; atol=1e-2)
    end
end

@testset "build_track: variable width is respected per-point" begin
    R = 100.0
    n = 360
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    w_left = [i <= n ÷ 2 ? 1.0 : 3.0 for i in 1:n]
    w_right = fill(1.0, n)

    track = F1Tenth.build_track(centerline, w_left, w_right)

    @test isapprox(norm(track.outer[1]), R + 1.0; atol=1e-2)
    @test isapprox(norm(track.outer[n ÷ 2 + 1]), R + 3.0; atol=1e-2)
end

@testset "load_track: reads CSV with header" begin
    mktempdir() do dir
        path = joinpath(dir, "track.csv")
        open(path, "w") do io
            println(io, "x,y,w_left,w_right")
            println(io, "0.0,0.0,1.0,1.0")
            println(io, "1.0,0.0,1.0,1.0")
            println(io, "1.0,1.0,1.0,1.0")
            println(io, "0.0,1.0,1.0,1.0")
        end
        track = F1Tenth.load_track(path)
        @test length(track.centerline) == 4
        @test track.centerline[1] == SVector(0.0, 0.0)
        @test length(track.segments) == 8
    end
end
```

Modify `test/runtests.jl`:

```julia
using F1Tenth
using Test

@testset "F1Tenth.jl" begin
    include("params_test.jl")
    include("mtk_models_test.jl")
    include("step_test.jl")
    include("solve_test.jl")
    include("geometry/track_test.jl")
end
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `build_track`/`load_track` not defined.

- [ ] **Step 4: Implement `track.jl`**

Create `src/geometry/track.jl`:

```julia
using StaticArrays
using LinearAlgebra: norm
using DelimitedFiles: readdlm

struct Segment
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
end

struct Track
    centerline::Vector{SVector{2,Float64}}
    w_left::Vector{Float64}
    w_right::Vector{Float64}
    inner::Vector{SVector{2,Float64}}
    outer::Vector{SVector{2,Float64}}
    segments::Vector{Segment}
end

function build_track(centerline::Vector{SVector{2,Float64}},
    w_left::Vector{Float64}, w_right::Vector{Float64})
    n = length(centerline)
    inner = Vector{SVector{2,Float64}}(undef, n)
    outer = Vector{SVector{2,Float64}}(undef, n)
    for i in 1:n
        prev = centerline[mod1(i - 1, n)]
        next = centerline[mod1(i + 1, n)]
        tangent = next - prev
        tangent = tangent / norm(tangent)
        normal_left = SVector(-tangent[2], tangent[1])
        outer[i] = centerline[i] + normal_left * w_left[i]
        inner[i] = centerline[i] - normal_left * w_right[i]
    end
    segments = Segment[]
    for i in 1:n
        j = mod1(i + 1, n)
        push!(segments, Segment(outer[i], outer[j]))
        push!(segments, Segment(inner[i], inner[j]))
    end
    return Track(centerline, w_left, w_right, inner, outer, segments)
end

function load_track(path::AbstractString)
    raw = readdlm(path, ',')
    start_row = all(x -> x isa Number, raw[1, 1:4]) ? 1 : 2
    data = Float64.(raw[start_row:end, 1:4])
    n = size(data, 1)
    centerline = [SVector(data[i, 1], data[i, 2]) for i in 1:n]
    return build_track(centerline, data[:, 3], data[:, 4])
end
```

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")
include("geometry/track.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE

function __init__()
    __mtk_models_init__()
end

end
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/geometry/track.jl test/geometry/track_test.jl test/runtests.jl
git commit -m "Add Track type, CSV loading, and boundary segment derivation"
```

---

### Task 2: KD-tree spatial index over boundary segments

**Files:**
- Create: `src/geometry/bvh.jl`
- Modify: `Project.toml` (add `NearestNeighbors` dependency)
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/bvh_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.Segment` (Task 1).
- Produces: `struct TrackBVH` with fields `tree::NearestNeighbors.KDTree,
  segments::Vector{Segment}, max_half_length::Float64`.
- Produces: `build_bvh(segments::Vector{Segment}) -> TrackBVH`.

- [ ] **Step 1: Add NearestNeighbors dependency**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add("NearestNeighbors")'
```
Expected: `Project.toml`/`Manifest.toml` updated.

- [ ] **Step 2: Write the failing test**

Create `test/geometry/bvh_test.jl`:

```julia
using F1Tenth
using StaticArrays
using NearestNeighbors: inrange
using Test

@testset "build_bvh: inrange query finds the expected segment" begin
    segments = [
        F1Tenth.Segment(SVector(0.0, 0.0), SVector(1.0, 0.0)),
        F1Tenth.Segment(SVector(10.0, 0.0), SVector(11.0, 0.0)),
    ]
    bvh = F1Tenth.build_bvh(segments)

    @test bvh.max_half_length == 0.5

    near_first = inrange(bvh.tree, SVector(0.5, 0.0), 1.0)
    @test near_first == [1]

    near_second = inrange(bvh.tree, SVector(10.5, 0.0), 1.0)
    @test near_second == [2]

    near_neither = inrange(bvh.tree, SVector(5.0, 0.0), 1.0)
    @test isempty(near_neither)
end
```

Modify `test/runtests.jl` (add after `track_test.jl`):

```julia
    include("geometry/track_test.jl")
    include("geometry/bvh_test.jl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `build_bvh` not defined.

- [ ] **Step 4: Implement `bvh.jl`**

Create `src/geometry/bvh.jl`:

```julia
using StaticArrays
using LinearAlgebra: norm
using NearestNeighbors: KDTree

struct TrackBVH
    tree::KDTree
    segments::Vector{Segment}
    max_half_length::Float64
end

function build_bvh(segments::Vector{Segment})
    midpoints = [(seg.p1 + seg.p2) / 2 for seg in segments]
    tree = KDTree(midpoints)
    max_half_length = maximum(norm(seg.p2 - seg.p1) / 2 for seg in segments)
    return TrackBVH(tree, segments, max_half_length)
end
```

Modify `src/F1Tenth.jl` to include `"geometry/bvh.jl"` after
`"geometry/track.jl"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/geometry/bvh.jl test/geometry/bvh_test.jl test/runtests.jl
git commit -m "Add KD-tree spatial index over track boundary segments"
```

---

### Task 3: Ray-segment intersection and the LiDAR raycaster

**Files:**
- Create: `src/geometry/raycast.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/raycast_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.Segment`, `F1Tenth.TrackBVH` (Task 2);
  `F1Tenth.State`, `F1Tenth.VehicleParams` (Phase 1).
- Produces: `cast_ray(origin::SVector{2,Float64}, angle::Float64,
  bvh::TrackBVH, max_range::Float64; step::Float64=0.1) -> Float64`.
- Produces: `struct LiDARConfig` with fields `num_beams::Int,
  fov::Float64, max_range::Float64, noise_std::Float64`, and a keyword
  constructor with the Global-Constraints defaults.
- Produces: `struct CarFootprint` with fields `length::Float64,
  width::Float64`, and a keyword constructor with the
  Global-Constraints defaults.
- Produces: `car_vertices(state::State, car::CarFootprint) ->
  Vector{SVector{2,Float64}}` (4 corners, front-left/front-right/
  rear-right/rear-left order).
- Produces: `scan(state::State, bvh::TrackBVH,
  other_agents::Vector{State}, car::CarFootprint, cfg::LiDARConfig;
  rng::Union{Nothing,AbstractRNG}=nothing) -> Vector{Float64}`.

- [ ] **Step 1: Write the failing tests**

Create `test/geometry/raycast_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Random: MersenneTwister
using Test

function _square_room_bvh(half_side::Float64)
    h = half_side
    segments = [
        F1Tenth.Segment(SVector(-h, -h), SVector(h, -h)),
        F1Tenth.Segment(SVector(h, -h), SVector(h, h)),
        F1Tenth.Segment(SVector(h, h), SVector(-h, h)),
        F1Tenth.Segment(SVector(-h, h), SVector(-h, -h)),
    ]
    return F1Tenth.build_bvh(segments)
end

@testset "cast_ray: hits a known wall at the expected distance" begin
    bvh = _square_room_bvh(10.0)
    t = F1Tenth.cast_ray(SVector(0.0, 0.0), 0.0, bvh, 30.0)
    @test isapprox(t, 10.0; atol=1e-6)

    t_up = F1Tenth.cast_ray(SVector(0.0, 0.0), π / 2, bvh, 30.0)
    @test isapprox(t_up, 10.0; atol=1e-6)
end

@testset "cast_ray: returns max_range when the wall is out of range" begin
    bvh = _square_room_bvh(10.0)
    t = F1Tenth.cast_ray(SVector(0.0, 0.0), 0.0, bvh, 5.0)
    @test t == 5.0
end

@testset "car_vertices: axis-aligned car at origin" begin
    car = F1Tenth.CarFootprint(; length=2.0, width=1.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    verts = F1Tenth.car_vertices(state, car)
    @test length(verts) == 4
    @test SVector(1.0, 0.5) in verts
    @test SVector(-1.0, -0.5) in verts
end

@testset "scan: sees the wall and a blocking agent" begin
    bvh = _square_room_bvh(10.0)
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.0, max_range=30.0, noise_std=0.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # fov=0.0 with num_beams=3 sends all 3 beams straight along +x.
    ranges_no_agents = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg)
    @test all(isapprox.(ranges_no_agents, 10.0; atol=1e-6))

    blocking_agent = F1Tenth.State(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ranges_with_agent = F1Tenth.scan(state, bvh, [blocking_agent], car, cfg)
    @test all(r -> r < 10.0 - 1e-6, ranges_with_agent)
end

@testset "scan: deterministic noise via explicit rng" begin
    bvh = _square_room_bvh(10.0)
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=30.0, noise_std=0.05)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    r1 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r2 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r3 = F1Tenth.scan(state, bvh, F1Tenth.State[], car, cfg; rng=MersenneTwister(2))
    @test r1 == r2
    @test r1 != r3
end
```

Modify `test/runtests.jl` (add after `bvh_test.jl`):

```julia
    include("geometry/bvh_test.jl")
    include("geometry/raycast_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `cast_ray`/`LiDARConfig`/`CarFootprint`/`car_vertices`/`scan` not defined.

- [ ] **Step 3: Implement `raycast.jl`**

Create `src/geometry/raycast.jl`:

```julia
using StaticArrays
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
```

Note: `cast_ray` calls `NearestNeighbors.inrange`, so `raycast.jl` needs
`import NearestNeighbors` (not just `using NearestNeighbors: KDTree` as
in `bvh.jl`). Add `import NearestNeighbors` at the top of `raycast.jl`.

Modify `src/F1Tenth.jl` to include `"geometry/raycast.jl"` after
`"geometry/bvh.jl"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If `car_vertices` order-dependent tests fail, check the
corner order against the test's expected `(1.0, 0.5)`/`(-1.0, -0.5)`
values (front-right and rear-left corners of a `length=2.0, width=1.0`
car at the origin, facing +x) rather than changing the test.

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/geometry/raycast.jl test/geometry/raycast_test.jl test/runtests.jl
git commit -m "Add ray-segment intersection and LiDAR scan simulation"
```

---

### Task 4: SAT car-vs-car and segment-based car-vs-track collision

**Files:**
- Create: `src/geometry/collision.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/collision_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.State` (Phase 1); `F1Tenth.Segment`,
  `F1Tenth.TrackBVH` (Task 2); `F1Tenth.CarFootprint`,
  `F1Tenth.car_vertices` (Task 3).
- Produces: `collides(vertices_a::Vector{SVector{2,Float64}},
  vertices_b::Vector{SVector{2,Float64}}) -> Bool` (SAT, works for any
  convex polygon pair, used here with 4-vertex rectangles).
- Produces: `check_car_collisions(states::Vector{State},
  car::CarFootprint) -> Vector{Bool}` (all-pairs, O(n²)).
- Produces: `check_track_collision(state::State, bvh::TrackBVH,
  car::CarFootprint) -> Bool`.

- [ ] **Step 1: Write the failing tests**

Create `test/geometry/collision_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Test

@testset "collides: SAT on axis-aligned rectangles" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    overlapping = [SVector(1.0, 1.0), SVector(3.0, 1.0), SVector(3.0, 3.0), SVector(1.0, 3.0)]
    separated = [SVector(5.0, 5.0), SVector(7.0, 5.0), SVector(7.0, 7.0), SVector(5.0, 7.0)]
    touching = [SVector(2.0, 0.0), SVector(4.0, 0.0), SVector(4.0, 2.0), SVector(2.0, 2.0)]

    @test F1Tenth.collides(a, overlapping)
    @test !F1Tenth.collides(a, separated)
    @test F1Tenth.collides(a, touching)
end

@testset "collides: SAT on a rotated rectangle" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    # A 45-degree-rotated square centered on a's center (1,1), sized to
    # clearly overlap a's interior (not just touch at a boundary point,
    # which would make the test fragile to floating-point rounding).
    rotated = [SVector(1.0, -0.2), SVector(2.2, 1.0), SVector(1.0, 2.2), SVector(-0.2, 1.0)]
    @test F1Tenth.collides(a, rotated)
end

@testset "check_car_collisions: all-pairs over an agent list" begin
    car = F1Tenth.CarFootprint(; length=0.5, width=0.3)
    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # overlaps agent 1
        F1Tenth.State(20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0), # isolated
    ]
    flags = F1Tenth.check_car_collisions(states, car)
    @test flags == [true, true, false]
end

@testset "check_track_collision: penetrating vs clear pose" begin
    half_side = 5.0
    h = half_side
    segments = [
        F1Tenth.Segment(SVector(-h, -h), SVector(h, -h)),
        F1Tenth.Segment(SVector(h, -h), SVector(h, h)),
        F1Tenth.Segment(SVector(h, h), SVector(-h, h)),
        F1Tenth.Segment(SVector(-h, h), SVector(-h, -h)),
    ]
    bvh = F1Tenth.build_bvh(segments)
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)

    clear = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test !F1Tenth.check_track_collision(clear, bvh, car)

    penetrating = F1Tenth.State(h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test F1Tenth.check_track_collision(penetrating, bvh, car)
end
```

Modify `test/runtests.jl` (add after `raycast_test.jl`):

```julia
    include("geometry/raycast_test.jl")
    include("geometry/collision_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `collides`/`check_car_collisions`/`check_track_collision` not defined.

- [ ] **Step 3: Implement `collision.jl`**

Create `src/geometry/collision.jl`:

```julia
using StaticArrays
using LinearAlgebra: dot, normalize
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
    query_radius = car.length + bvh.max_half_length
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
```

Modify `src/F1Tenth.jl` to include `"geometry/collision.jl"` after
`"geometry/raycast.jl"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/geometry/collision.jl test/geometry/collision_test.jl test/runtests.jl
git commit -m "Add SAT car-vs-car and segment-based car-vs-track collision"
```

---

### Task 5: `step_multi_agent!` — wiring Phase 1 dynamics with Phase 2 geometry

**Files:**
- Create: `src/geometry/step2.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/step2_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.step_dynamics` (Phase 1, unmodified);
  `F1Tenth.TrackBVH` (Task 2); `F1Tenth.scan`, `F1Tenth.CarFootprint`,
  `F1Tenth.LiDARConfig` (Task 3); `F1Tenth.check_car_collisions`,
  `F1Tenth.check_track_collision` (Task 4).
- Produces: `step_multi_agent!(states::Vector{State},
  actions::Vector{Action}, bvh::TrackBVH, params::VehicleParams,
  car::CarFootprint, dt::Float64, cfg::LiDARConfig) ->
  (Vector{State}, Vector{Vector{Float64}}, Vector{Bool})`.

- [ ] **Step 1: Write the failing tests**

Create `test/geometry/step2_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Test

function _square_room_bvh(half_side::Float64)
    h = half_side
    segments = [
        F1Tenth.Segment(SVector(-h, -h), SVector(h, -h)),
        F1Tenth.Segment(SVector(h, -h), SVector(h, h)),
        F1Tenth.Segment(SVector(h, h), SVector(-h, h)),
        F1Tenth.Segment(SVector(-h, h), SVector(-h, -h)),
    ]
    return F1Tenth.build_bvh(segments)
end

@testset "step_multi_agent!: Phase 1 state evolution is unaffected" begin
    bvh = _square_room_bvh(50.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.0)

    state = F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.1, 1.0)
    dt = 0.01

    expected = F1Tenth.step_dynamics(state, action, params, dt)
    new_states, _, _ = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, dt, cfg)

    @test new_states[1] == expected
end

@testset "step_multi_agent!: flags a car driven into the wall" begin
    bvh = _square_room_bvh(1.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)

    # Already at the wall boundary and driving forward into it.
    state = F1Tenth.State(0.9, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.0, 0.0)

    _, _, collisions = F1Tenth.step_multi_agent!([state], [action], bvh, params, car, 0.1, cfg)
    @test collisions[1]
end

@testset "step_multi_agent!: flags two cars driven into each other" begin
    bvh = _square_room_bvh(50.0)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)

    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    actions = [F1Tenth.Action(0.0, 0.0), F1Tenth.Action(0.0, 0.0)]

    _, scans, collisions = F1Tenth.step_multi_agent!(states, actions, bvh, params, car, 0.01, cfg)
    @test collisions == [true, true]
    @test length(scans) == 2
    @test length(scans[1]) == cfg.num_beams
end
```

Modify `test/runtests.jl` (add after `collision_test.jl`):

```julia
    include("geometry/collision_test.jl")
    include("geometry/step2_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `step_multi_agent!` not defined.

- [ ] **Step 3: Implement `step2.jl`**

Create `src/geometry/step2.jl`:

```julia
function step_multi_agent!(states::Vector{State}, actions::Vector{Action},
    bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64, cfg::LiDARConfig)
    n = length(states)
    new_states = [step_dynamics(states[i], actions[i], params, dt) for i in 1:n]

    scans = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        others = [new_states[j] for j in 1:n if j != i]
        scans[i] = scan(new_states[i], bvh, others, car, cfg)
    end

    car_flags = check_car_collisions(new_states, car)
    collisions = [car_flags[i] || check_track_collision(new_states[i], bvh, car) for i in 1:n]

    return new_states, scans, collisions
end
```

Modify `src/F1Tenth.jl` to include `"geometry/step2.jl"` after
`"geometry/collision.jl"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the "driven into the wall" test doesn't flag a
collision, check that `car.length` combined with the pose in the test
actually places a body vertex past `x=1.0`; adjust the test's `x`
coordinate closer to the wall rather than changing
`check_track_collision`'s logic.

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/geometry/step2.jl test/geometry/step2_test.jl test/runtests.jl
git commit -m "Add step_multi_agent!, wiring Phase 1 dynamics with Phase 2 geometry"
```

---

### Task 6: `to_lazysets` export for Phase 5

**Files:**
- Create: `src/geometry/lazysets_export.jl`
- Modify: `Project.toml` (add `LazySets` dependency)
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/geometry/lazysets_export_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.Track` (Task 1).
- Produces: `to_lazysets(track::Track) -> Vector{LazySets.VPolygon}`,
  one convex quad per centerline segment slice (`outer[i], outer[i+1],
  inner[i+1], inner[i]`, wrapping for the closed loop).

- [ ] **Step 1: Add LazySets dependency**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add("LazySets")'
```
Expected: `Project.toml`/`Manifest.toml` updated.

- [ ] **Step 2: Write the failing test**

Create `test/geometry/lazysets_export_test.jl`:

```julia
using F1Tenth
using StaticArrays
using LazySets: VPolygon, an_element, ∈
using Test

@testset "to_lazysets: one convex polygon per centerline slice" begin
    R = 100.0
    n = 360
    w = 1.5
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    track = F1Tenth.build_track(centerline, fill(w, n), fill(w, n))

    polys = F1Tenth.to_lazysets(track)

    @test length(polys) == n
    @test all(p -> p isa VPolygon, polys)

    # The centerline waypoint itself must lie inside (or on the boundary
    # of) its own slice polygon.
    @test track.centerline[1] ∈ polys[1] || track.centerline[1] ∈ polys[n]
end
```

Modify `test/runtests.jl` (add after `step2_test.jl`):

```julia
    include("geometry/step2_test.jl")
    include("geometry/lazysets_export_test.jl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `to_lazysets` not defined.

- [ ] **Step 4: Implement `lazysets_export.jl`**

Create `src/geometry/lazysets_export.jl`:

```julia
import LazySets

function to_lazysets(track::Track)
    n = length(track.centerline)
    return [LazySets.VPolygon([track.outer[i], track.outer[mod1(i + 1, n)],
                track.inner[mod1(i + 1, n)], track.inner[i]])
            for i in 1:n]
end
```

Modify `src/F1Tenth.jl` to include `"geometry/lazysets_export.jl"`
after `"geometry/step2.jl"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the membership test fails because the quad's vertex
winding order confuses `VPolygon`'s internal convex-hull handling, try
listing the vertices in strict counter-clockwise order for that slice
(`inner[i], outer[i], outer[i+1], inner[i+1]`) instead — check
`LazySets.VPolygon`'s docstring (`?LazySets.VPolygon`) for whether it
requires a specific winding.

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/geometry/lazysets_export.jl test/geometry/lazysets_export_test.jl test/runtests.jl
git commit -m "Add to_lazysets track export for Phase 5 reachability guards"
```

---

### Task 7: Module exports and full-suite verification

**Files:**
- Modify: `src/F1Tenth.jl`

**Interfaces:**
- Consumes: everything produced in Tasks 1-6.
- Produces: public names exported from `F1Tenth` in addition to
  Phase 1's exports: `Track`, `Segment`, `build_track`, `load_track`,
  `TrackBVH`, `build_bvh`, `LiDARConfig`, `CarFootprint`,
  `car_vertices`, `cast_ray`, `scan`, `collides`,
  `check_car_collisions`, `check_track_collision`,
  `step_multi_agent!`, `to_lazysets`.

- [ ] **Step 1: Add exports**

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")
include("geometry/track.jl")
include("geometry/bvh.jl")
include("geometry/raycast.jl")
include("geometry/collision.jl")
include("geometry/step2.jl")
include("geometry/lazysets_export.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE
export Track, Segment, build_track, load_track
export TrackBVH, build_bvh
export LiDARConfig, CarFootprint, car_vertices, cast_ray, scan
export collides, check_car_collisions, check_track_collision
export step_multi_agent!
export to_lazysets

function __init__()
    __mtk_models_init__()
end

end
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS (all testsets from Tasks 1-6, plus all Phase 1 testsets
still passing unchanged).

- [ ] **Step 3: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl
git commit -m "Export Phase 2 public API from F1Tenth module"
```
