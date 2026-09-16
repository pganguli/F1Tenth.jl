# Phase 3: Decoupled Abstract Renderer Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a decoupled rendering layer to `F1Tenth.jl` — one
`AbstractRenderer` interface (`init!`/`render!`/`close!`/`reset!`) with
three implementations (`NullRenderer`, `GLMakieRenderer`,
`CairoMakieRenderer`) — that consumes `step_multi_agent!`'s per-tick
`(states, scans, collisions)` output and the `Track` used to build its
`bvh`, without modifying any file under `src/dynamics/` or
`src/geometry/`.

**Architecture:** `AbstractRenderer` and `NullRenderer` live directly
in the main module (zero extra runtime dependency). `GLMakieRenderer`/
`CairoMakieRenderer` are empty constructor stubs in the main module,
implemented as Julia package extensions (`ext/F1TenthGLMakieExt.jl`,
`ext/F1TenthCairoMakieExt.jl`) that only load when the caller also has
`GLMakie`/`CairoMakie` loaded (`[weakdeps]` in `Project.toml`), so a
headless/training user never pays either backend's install/precompile
cost. `GLMakieRenderer` builds its `Figure`/`Axis` once and updates
Makie `Observable`s incrementally per tick (live view). `CairoMakieRenderer`
buffers each tick's data cheaply in `render!` and does all drawing in
`close!`, emitting a single summary SVG (full trails) and, if
configured, one SVG per tick for a vector "flip-book" replay.

**Tech Stack:** Julia 1.13 (repo compat floor 1.10.10), GLMakie.jl
(new, weakdep), CairoMakie.jl (new, weakdep), StaticArrays.jl
(existing), Test (stdlib).

**Spec:** `docs/superpowers/specs/2026-09-16-phase3-renderer-engine-design.md`

## Global Constraints

- No file under `src/dynamics/` or `src/geometry/` may be modified.
  All Phase 3 code lives under `src/render/` and `ext/`.
- `render!`'s positional argument shape is fixed:
  `render!(renderer, states::Vector{State}, scans::Vector{Vector{Float64}},
  collisions::Vector{Bool})` — exactly `step_multi_agent!`'s return
  shape, so `render!(renderer, step_multi_agent!(...)...)` works
  directly (spec Decision 1).
- `render!` before `init!`/`reset!` must throw a clear `ErrorException`
  for `GLMakieRenderer`/`CairoMakieRenderer`; `NullRenderer` is always
  a safe no-op regardless of lifecycle state (spec Decision 1/2).
- `CarFootprint` and `LiDARConfig` are supplied once at each renderer's
  construction (`car=`/`cfg=` keywords), never per-`render!`-call —
  `LiDARConfig` is required to reconstruct beam angles for drawing scan
  rays, since `render!`'s `scans` argument carries only numeric ranges
  (spec's Decision-3 amendment).
- `GLMakie`/`CairoMakie` must be `[weakdeps]` + `[extensions]` entries
  in `Project.toml`, never plain `[deps]` (spec Decision 3).
- `GLMakieRenderer` updates must be Observable-based incremental
  updates, not full clear-and-redraw per tick (spec Decision 4).
- `CairoMakieRenderer` output is vector (SVG) only — no video/GIF
  encoding, no ffmpeg dependency (spec Non-goals, Decision 5).
- GLMakie UUID: `e9467ef8-e4e7-5192-8a1a-b1aee30e663a`. CairoMakie
  UUID: `13f3f980-e62b-5c42-98c6-ff1f3baf88f0`. (Looked up directly
  from the local Julia General registry; use these exact values rather
  than whatever a fresh `Pkg.add` might resolve to first, to keep
  `Project.toml` edits deterministic.)

---

### Task 1: `AbstractRenderer` interface, `NullRenderer`, shared test fixture

**Files:**
- Create: `src/render/interface.jl`
- Modify: `src/F1Tenth.jl` (include the new file, add exports)
- Create: `test/render/testutils.jl`
- Create: `test/render/null_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `F1Tenth.Track` (Phase 2), `F1Tenth.State` (Phase 1).
- Produces: `abstract type AbstractRenderer end`.
- Produces: `struct NullRenderer <: AbstractRenderer end`.
- Produces: generic functions `init!(renderer, track::Track) ->
  renderer`, `render!(renderer, states::Vector{State},
  scans::Vector{Vector{Float64}}, collisions::Vector{Bool}) ->
  Nothing`, `close!(renderer) -> Nothing`, `reset!(renderer,
  track::Track) -> renderer`, each with a `NullRenderer` method that is
  an immediate no-op.
- Produces: empty constructor stubs `function GLMakieRenderer end`,
  `function CairoMakieRenderer end` (no methods yet — added by Tasks 2
  and 3).
- Produces (test-only): `small_test_track() -> Track`, an 8-point
  circular closed-loop track for reuse by every renderer test file.

- [ ] **Step 1: Write the failing tests**

Create `test/render/testutils.jl`:

```julia
using StaticArrays

function small_test_track()
    n = 8
    R = 5.0
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
end
```

Create `test/render/null_test.jl`:

```julia
using F1Tenth
using Test

include("testutils.jl")

@testset "NullRenderer: full lifecycle is a safe no-op" begin
    track = small_test_track()
    renderer = F1Tenth.NullRenderer()

    F1Tenth.init!(renderer, track)
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    scans = [[1.0, 2.0, 3.0]]
    collisions = [false]
    F1Tenth.render!(renderer, states, scans, collisions)
    F1Tenth.reset!(renderer, track)
    F1Tenth.render!(renderer, states, scans, collisions)
    F1Tenth.close!(renderer)

    @test true  # reaching here without throwing is the assertion
end

@testset "NullRenderer: render! before init! does not throw" begin
    renderer = F1Tenth.NullRenderer()
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    F1Tenth.render!(renderer, states, [[1.0]], [false])
    @test true
end

@testset "NullRenderer: render! allocates zero bytes" begin
    renderer = F1Tenth.NullRenderer()
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    scans = [[1.0]]
    collisions = [false]
    F1Tenth.render!(renderer, states, scans, collisions)  # compile once
    allocs = @allocated F1Tenth.render!(renderer, states, scans, collisions)
    @test allocs == 0
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
    include("geometry/bvh_test.jl")
    include("geometry/raycast_test.jl")
    include("geometry/collision_test.jl")
    include("geometry/step2_test.jl")
    include("geometry/lazysets_export_test.jl")
    include("render/null_test.jl")
    include("exports_test.jl")
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.NullRenderer`/`init!`/`render!`/`close!`/`reset!` not defined.

- [ ] **Step 3: Implement `interface.jl`**

Create `src/render/interface.jl`:

```julia
abstract type AbstractRenderer end

function init! end
function render! end
function close! end
function reset! end

# Implemented by ext/F1TenthGLMakieExt.jl and ext/F1TenthCairoMakieExt.jl
# respectively -- these have no methods until the caller also loads
# GLMakie/CairoMakie. See Project.toml's [weakdeps]/[extensions].
function GLMakieRenderer end
function CairoMakieRenderer end

struct NullRenderer <: AbstractRenderer end

init!(renderer::NullRenderer, track::Track) = renderer
render!(renderer::NullRenderer, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool}) = nothing
close!(renderer::NullRenderer) = nothing
reset!(renderer::NullRenderer, track::Track) = renderer
```

Modify `src/F1Tenth.jl` to add the new `include`, without touching the
`export` lines yet (matching Phase 2's pattern: every task through this
plan uses qualified `F1Tenth.foo` access in its own tests; the export
list is only extended once, in Task 4):

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
include("render/interface.jl")

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/render/interface.jl test/render/testutils.jl test/render/null_test.jl test/runtests.jl
git commit -m "Add AbstractRenderer interface and NullRenderer"
```

---

### Task 2: `GLMakieRenderer` package extension

**Files:**
- Modify: `Project.toml` (add `GLMakie` under `[weakdeps]`/`[extensions]`/`[compat]`/`[extras]`/`[targets]`)
- Create: `ext/F1TenthGLMakieExt.jl`
- Test: `test/render/glmakie_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `F1Tenth.AbstractRenderer`, `F1Tenth.init!`,
  `F1Tenth.render!`, `F1Tenth.close!`, `F1Tenth.reset!`,
  `F1Tenth.GLMakieRenderer` stub (Task 1); `F1Tenth.Track`,
  `F1Tenth.CarFootprint`, `F1Tenth.car_vertices` (Phase 2);
  `F1Tenth.State`, `F1Tenth.LiDARConfig` (Phase 1/2).
- Produces: a method for `F1Tenth.GLMakieRenderer(track::Track;
  camera_mode::Symbol=:fixed, ego_index::Int=1,
  car::CarFootprint=CarFootprint(), cfg::LiDARConfig=LiDARConfig())`
  returning a private `GLMakieRendererImpl <: AbstractRenderer`
  (not exported — reached only via the `GLMakieRenderer` name).
- Produces: `init!`/`render!`/`close!`/`reset!` methods for
  `GLMakieRendererImpl`.

- [ ] **Step 1: Edit `Project.toml`**

Replace the full contents of `Project.toml` with:

```toml
name = "F1Tenth"
uuid = "ec9dc3f2-e932-4b18-a493-a4686694dcbe"
version = "1.0.0-DEV"
authors = ["Prateek Ganguli <prateek.ganguli@gmail.com> and contributors"]

[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[weakdeps]
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"

[extensions]
F1TenthGLMakieExt = "GLMakie"

[compat]
DelimitedFiles = "1.9.1"
DifferentialEquations = "8.1.1"
GLMakie = "0.13.6"
LazySets = "7.1.0"
ModelingToolkit = "11.43.1"
NearestNeighbors = "0.4.29"
StaticArrays = "1.9.20"
Symbolics = "7.39.2"
julia = "1.10.10"

[extras]
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["GLMakie", "Test"]
```

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.resolve(); Pkg.instantiate()'
```
Expected: `Manifest.toml` updates to include GLMakie and its
dependency tree. This step downloads and precompiles GLMakie/Makie,
which is large — expect several minutes, not a hang.

- [ ] **Step 2: Write the failing tests**

Create `test/render/glmakie_test.jl`:

```julia
using F1Tenth
using GLMakie
using Test

include("testutils.jl")

GLMakie.activate!(; visible=false)

function _glmakie_usable()
    try
        fig = Figure()
        display(fig)
        return true
    catch
        return false
    end
end

if _glmakie_usable()
    @testset "GLMakieRenderer: render! updates per-car Observables" begin
        track = small_test_track()
        renderer = F1Tenth.GLMakieRenderer(track;
            car=F1Tenth.CarFootprint(), cfg=F1Tenth.LiDARConfig(; num_beams=3, fov=1.0))

        states1 = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        scans1 = [[1.0, 2.0, 3.0]]
        F1Tenth.render!(renderer, states1, scans1, [false])

        @test length(renderer.car_polys) == 1
        @test length(renderer.car_polys[1][]) == 4
        @test renderer.trail_points[1] == [Point2f(0.0, 0.0)]
        @test renderer.car_colors[1][] == :blue

        states2 = [F1Tenth.State(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(renderer, states2, scans1, [true])

        @test renderer.trail_points[1] == [Point2f(0.0, 0.0), Point2f(1.0, 0.0)]
        @test renderer.car_colors[1][] == :red

        F1Tenth.reset!(renderer, track)
        @test isempty(renderer.car_polys)

        F1Tenth.close!(renderer)
    end

    @testset "GLMakieRenderer: render! before init! throws" begin
        track = small_test_track()
        renderer = F1Tenth.GLMakieRenderer(track)
        renderer.initialized = false
        @test_throws ErrorException F1Tenth.render!(renderer, F1Tenth.State[], Vector{Float64}[], Bool[])
    end
else
    @warn "Skipping GLMakieRenderer tests: no usable OpenGL context in this environment"
end
```

Modify `test/runtests.jl` (add after `null_test.jl`):

```julia
    include("render/null_test.jl")
    include("render/glmakie_test.jl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.GLMakieRenderer(track; ...)` has no method
(the stub from Task 1 has zero methods).

- [ ] **Step 4: Implement `F1TenthGLMakieExt.jl`**

Create `ext/F1TenthGLMakieExt.jl`:

```julia
module F1TenthGLMakieExt

import F1Tenth
using GLMakie

mutable struct GLMakieRendererImpl <: F1Tenth.AbstractRenderer
    camera_mode::Symbol
    ego_index::Int
    car::F1Tenth.CarFootprint
    cfg::F1Tenth.LiDARConfig
    fig::Union{Nothing,Figure}
    ax::Union{Nothing,Axis}
    half_extent::Float64
    car_polys::Vector{Observable{Vector{Point2f}}}
    trails::Vector{Observable{Vector{Point2f}}}
    trail_points::Vector{Vector{Point2f}}
    scan_lines::Vector{Observable{Vector{Point2f}}}
    car_colors::Vector{Observable{Symbol}}
    initialized::Bool
end

function F1Tenth.GLMakieRenderer(track::F1Tenth.Track; camera_mode::Symbol=:fixed,
    ego_index::Int=1, car::F1Tenth.CarFootprint=F1Tenth.CarFootprint(),
    cfg::F1Tenth.LiDARConfig=F1Tenth.LiDARConfig())
    renderer = GLMakieRendererImpl(camera_mode, ego_index, car, cfg, nothing, nothing, 0.0,
        Observable{Vector{Point2f}}[], Observable{Vector{Point2f}}[], Vector{Point2f}[],
        Observable{Vector{Point2f}}[], Observable{Symbol}[], false)
    F1Tenth.init!(renderer, track)
    return renderer
end

function _plot_track!(ax, track::F1Tenth.Track)
    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
    lines!(ax, outer_pts; color=:black)
    lines!(ax, inner_pts; color=:black)
end

function _track_bounds(track::F1Tenth.Track)
    pts = vcat(track.outer, track.inner)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
end

function F1Tenth.init!(renderer::GLMakieRendererImpl, track::F1Tenth.Track)
    if renderer.fig === nothing
        renderer.fig = Figure()
        renderer.ax = Axis(renderer.fig[1, 1]; aspect=DataAspect())
        display(renderer.fig)
    else
        empty!(renderer.ax)
    end
    _plot_track!(renderer.ax, track)
    xmin, xmax, ymin, ymax = _track_bounds(track)
    renderer.half_extent = max(xmax - xmin, ymax - ymin) / 2
    if renderer.camera_mode == :fixed
        limits!(renderer.ax, xmin, xmax, ymin, ymax)
    end
    empty!(renderer.car_polys)
    empty!(renderer.trails)
    empty!(renderer.trail_points)
    empty!(renderer.scan_lines)
    empty!(renderer.car_colors)
    renderer.initialized = true
    return renderer
end

F1Tenth.reset!(renderer::GLMakieRendererImpl, track::F1Tenth.Track) = F1Tenth.init!(renderer, track)

function _ensure_car_plots!(renderer::GLMakieRendererImpl, n::Int)
    isempty(renderer.car_polys) || return
    for _ in 1:n
        car_obs = Observable(Point2f[])
        trail_obs = Observable(Point2f[])
        scan_obs = Observable(Point2f[])
        color_obs = Observable(:blue)
        poly!(renderer.ax, car_obs; color=color_obs)
        lines!(renderer.ax, trail_obs; color=:gray, linestyle=:dot)
        lines!(renderer.ax, scan_obs; color=:red, linewidth=0.5)
        push!(renderer.car_polys, car_obs)
        push!(renderer.trails, trail_obs)
        push!(renderer.trail_points, Point2f[])
        push!(renderer.scan_lines, scan_obs)
        push!(renderer.car_colors, color_obs)
    end
end

function _scan_ray_points(state::F1Tenth.State, beam_ranges::Vector{Float64}, cfg::F1Tenth.LiDARConfig)
    x, y, _, _, ψ, _, _ = state
    angles = cfg.num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=cfg.num_beams)
    points = Point2f[]
    for (angle, r) in zip(angles, beam_ranges)
        push!(points, Point2f(x, y))
        push!(points, Point2f(x + r * cos(angle), y + r * sin(angle)))
        push!(points, Point2f(NaN, NaN))
    end
    return points
end

function F1Tenth.render!(renderer::GLMakieRendererImpl, states::Vector{F1Tenth.State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    renderer.initialized || error("GLMakieRenderer: render! called before init!/reset!")
    n = length(states)
    _ensure_car_plots!(renderer, n)
    for i in 1:n
        verts = F1Tenth.car_vertices(states[i], renderer.car)
        renderer.car_polys[i][] = [Point2f(v[1], v[2]) for v in verts]
        push!(renderer.trail_points[i], Point2f(states[i][1], states[i][2]))
        renderer.trails[i][] = renderer.trail_points[i]
        renderer.car_colors[i][] = collisions[i] ? :red : :blue
        renderer.scan_lines[i][] = _scan_ray_points(states[i], scans[i], renderer.cfg)
    end
    if renderer.camera_mode == :follow
        ex, ey = states[renderer.ego_index][1], states[renderer.ego_index][2]
        h = renderer.half_extent
        limits!(renderer.ax, ex - h, ex + h, ey - h, ey + h)
    end
    return nothing
end

function F1Tenth.close!(renderer::GLMakieRendererImpl)
    renderer.fig === nothing || GLMakie.closeall()
    return nothing
end

end # module
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If `_glmakie_usable()` returns `false` in this
environment (no working OpenGL context), the GLMakie testset is
skipped with a `@warn` and the suite still passes overall — this is
expected per the spec, not a failure to chase.

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml ext/F1TenthGLMakieExt.jl test/render/glmakie_test.jl test/runtests.jl
git commit -m "Add GLMakieRenderer package extension"
```

---

### Task 3: `CairoMakieRenderer` package extension

**Files:**
- Modify: `Project.toml` (add `CairoMakie` under `[weakdeps]`/`[extensions]`/`[compat]`/`[extras]`/`[targets]`)
- Create: `ext/F1TenthCairoMakieExt.jl`
- Test: `test/render/cairomakie_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `F1Tenth.AbstractRenderer`, `F1Tenth.init!`,
  `F1Tenth.render!`, `F1Tenth.close!`, `F1Tenth.reset!`,
  `F1Tenth.CairoMakieRenderer` stub (Task 1); `F1Tenth.Track`,
  `F1Tenth.CarFootprint`, `F1Tenth.car_vertices` (Phase 2);
  `F1Tenth.State`, `F1Tenth.LiDARConfig` (Phase 1/2).
- Produces: a method for `F1Tenth.CairoMakieRenderer(track::Track;
  output_dir::Union{Nothing,String}=nothing,
  summary_path::String="summary.svg",
  car::CarFootprint=CarFootprint(), cfg::LiDARConfig=LiDARConfig())`
  returning a private `CairoMakieRendererImpl <: AbstractRenderer`.
  (`summary_path` pins down where the single always-emitted summary
  SVG goes — the spec calls for "always emit summary.svg" but leaves
  its exact path unspecified since the constructor's only path knob
  was `output_dir` for the optional per-tick frames; `summary_path`
  fills that gap explicitly rather than silently defaulting into the
  caller's current working directory.)
- Produces: `init!`/`render!`/`close!`/`reset!` methods for
  `CairoMakieRendererImpl`.

- [ ] **Step 1: Edit `Project.toml`**

Replace the full contents of `Project.toml` with:

```toml
name = "F1Tenth"
uuid = "ec9dc3f2-e932-4b18-a493-a4686694dcbe"
version = "1.0.0-DEV"
authors = ["Prateek Ganguli <prateek.ganguli@gmail.com> and contributors"]

[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[weakdeps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"

[extensions]
F1TenthCairoMakieExt = "CairoMakie"
F1TenthGLMakieExt = "GLMakie"

[compat]
CairoMakie = "0.15.6"
DelimitedFiles = "1.9.1"
DifferentialEquations = "8.1.1"
GLMakie = "0.13.6"
LazySets = "7.1.0"
ModelingToolkit = "11.43.1"
NearestNeighbors = "0.4.29"
StaticArrays = "1.9.20"
Symbolics = "7.39.2"
julia = "1.10.10"

[extras]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["CairoMakie", "GLMakie", "Test"]
```

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.resolve(); Pkg.instantiate()'
```
Expected: `Manifest.toml` updates to include CairoMakie and its
dependency tree.

- [ ] **Step 2: Write the failing tests**

Create `test/render/cairomakie_test.jl`:

```julia
using F1Tenth
using CairoMakie
using Test

include("testutils.jl")

@testset "CairoMakieRenderer: full lifecycle produces summary and frame SVGs" begin
    track = small_test_track()
    mktempdir() do dir
        summary_path = joinpath(dir, "summary.svg")
        frames_dir = joinpath(dir, "frames")
        renderer = F1Tenth.CairoMakieRenderer(track; output_dir=frames_dir, summary_path=summary_path,
            car=F1Tenth.CarFootprint(), cfg=F1Tenth.LiDARConfig(; num_beams=3, fov=1.0))

        for tick in 1:3
            states = [F1Tenth.State(Float64(tick), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            scans = [[1.0, 2.0, 3.0]]
            F1Tenth.render!(renderer, states, scans, [tick == 3])
        end
        F1Tenth.close!(renderer)

        @test isfile(summary_path)
        @test filesize(summary_path) > 0
        content = read(summary_path, String)
        @test startswith(content, "<?xml") || startswith(content, "<svg")

        frame_files = readdir(frames_dir)
        @test length(frame_files) == 3
    end
end

@testset "CairoMakieRenderer: close! with zero ticks still emits a track-only summary" begin
    track = small_test_track()
    mktempdir() do dir
        summary_path = joinpath(dir, "summary.svg")
        renderer = F1Tenth.CairoMakieRenderer(track; summary_path=summary_path)
        F1Tenth.close!(renderer)
        @test isfile(summary_path)
    end
end

@testset "CairoMakieRenderer: render! before init! throws" begin
    track = small_test_track()
    renderer = F1Tenth.CairoMakieRenderer(track)
    renderer.initialized = false
    @test_throws ErrorException F1Tenth.render!(renderer, F1Tenth.State[], Vector{Float64}[], Bool[])
end
```

Modify `test/runtests.jl` (add after `glmakie_test.jl`):

```julia
    include("render/glmakie_test.jl")
    include("render/cairomakie_test.jl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.CairoMakieRenderer(track; ...)` has no
method.

- [ ] **Step 4: Implement `F1TenthCairoMakieExt.jl`**

Create `ext/F1TenthCairoMakieExt.jl`:

```julia
module F1TenthCairoMakieExt

import F1Tenth
using CairoMakie

mutable struct CairoMakieRendererImpl <: F1Tenth.AbstractRenderer
    output_dir::Union{Nothing,String}
    summary_path::String
    car::F1Tenth.CarFootprint
    cfg::F1Tenth.LiDARConfig
    track::Union{Nothing,F1Tenth.Track}
    history_states::Vector{Vector{F1Tenth.State}}
    history_scans::Vector{Vector{Vector{Float64}}}
    history_collisions::Vector{Vector{Bool}}
    initialized::Bool
end

function F1Tenth.CairoMakieRenderer(track::F1Tenth.Track;
    output_dir::Union{Nothing,String}=nothing, summary_path::String="summary.svg",
    car::F1Tenth.CarFootprint=F1Tenth.CarFootprint(), cfg::F1Tenth.LiDARConfig=F1Tenth.LiDARConfig())
    renderer = CairoMakieRendererImpl(output_dir, summary_path, car, cfg, nothing,
        Vector{F1Tenth.State}[], Vector{Vector{Float64}}[], Vector{Bool}[], false)
    F1Tenth.init!(renderer, track)
    return renderer
end

function F1Tenth.init!(renderer::CairoMakieRendererImpl, track::F1Tenth.Track)
    renderer.track = track
    empty!(renderer.history_states)
    empty!(renderer.history_scans)
    empty!(renderer.history_collisions)
    renderer.output_dir === nothing || mkpath(renderer.output_dir)
    renderer.initialized = true
    return renderer
end

F1Tenth.reset!(renderer::CairoMakieRendererImpl, track::F1Tenth.Track) = F1Tenth.init!(renderer, track)

function F1Tenth.render!(renderer::CairoMakieRendererImpl, states::Vector{F1Tenth.State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    renderer.initialized || error("CairoMakieRenderer: render! called before init!/reset!")
    push!(renderer.history_states, copy(states))
    push!(renderer.history_scans, copy(scans))
    push!(renderer.history_collisions, copy(collisions))
    return nothing
end

function _plot_track!(ax, track::F1Tenth.Track)
    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
    lines!(ax, outer_pts; color=:black)
    lines!(ax, inner_pts; color=:black)
end

function _draw_frame(track::F1Tenth.Track, trails::Vector{Vector{Point2f}},
    states::Vector{F1Tenth.State}, scans::Vector{Vector{Float64}}, collisions::Vector{Bool},
    car::F1Tenth.CarFootprint, cfg::F1Tenth.LiDARConfig)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    _plot_track!(ax, track)
    for i in eachindex(states)
        isempty(trails[i]) || lines!(ax, trails[i]; color=:gray, linestyle=:dot)
        verts = F1Tenth.car_vertices(states[i], car)
        poly!(ax, [Point2f(v[1], v[2]) for v in verts]; color=collisions[i] ? :red : :blue)
        x, y, _, _, ψ, _, _ = states[i]
        num_beams = length(scans[i])
        angles = num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=num_beams)
        for (angle, r) in zip(angles, scans[i])
            lines!(ax, [Point2f(x, y), Point2f(x + r * cos(angle), y + r * sin(angle))];
                color=:red, linewidth=0.5)
        end
    end
    return fig
end

function F1Tenth.close!(renderer::CairoMakieRendererImpl)
    track = renderer.track
    n_ticks = length(renderer.history_states)
    n_cars = n_ticks == 0 ? 0 : length(renderer.history_states[1])

    if n_ticks == 0
        summary_fig = Figure()
        ax = Axis(summary_fig[1, 1]; aspect=DataAspect())
        _plot_track!(ax, track)
    else
        full_trails = [Point2f[] for _ in 1:n_cars]
        for tick in 1:n_ticks, i in 1:n_cars
            s = renderer.history_states[tick][i]
            push!(full_trails[i], Point2f(s[1], s[2]))
        end
        summary_fig = _draw_frame(track, full_trails, renderer.history_states[end],
            renderer.history_scans[end], renderer.history_collisions[end], renderer.car, renderer.cfg)
    end
    save(renderer.summary_path, summary_fig)

    if renderer.output_dir !== nothing
        running_trails = [Point2f[] for _ in 1:n_cars]
        for tick in 1:n_ticks
            for i in 1:n_cars
                s = renderer.history_states[tick][i]
                push!(running_trails[i], Point2f(s[1], s[2]))
            end
            frame_fig = _draw_frame(track, running_trails, renderer.history_states[tick],
                renderer.history_scans[tick], renderer.history_collisions[tick], renderer.car, renderer.cfg)
            path = joinpath(renderer.output_dir, "frame_" * lpad(tick, 4, '0') * ".svg")
            save(path, frame_fig)
        end
    end
    return nothing
end

end # module
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml ext/F1TenthCairoMakieExt.jl test/render/cairomakie_test.jl test/runtests.jl
git commit -m "Add CairoMakieRenderer package extension"
```

---

### Task 4: Module exports and full-suite verification

**Files:**
- Modify: `src/F1Tenth.jl`
- Modify: `test/exports_test.jl`

**Interfaces:**
- Consumes: everything produced in Tasks 1-3.
- Produces: public names exported from `F1Tenth` in addition to Phase
  1/2's exports: `AbstractRenderer`, `NullRenderer`, `GLMakieRenderer`,
  `CairoMakieRenderer`, `init!`, `render!`, `close!`, `reset!`.

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
include("render/interface.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE
export Track, Segment, build_track, load_track
export TrackBVH, build_bvh
export LiDARConfig, CarFootprint, car_vertices, cast_ray, scan
export collides, check_car_collisions, check_track_collision
export step_multi_agent!
export to_lazysets
export AbstractRenderer, NullRenderer, GLMakieRenderer, CairoMakieRenderer
export init!, render!, close!, reset!

function __init__()
    __mtk_models_init__()
end

end
```

- [ ] **Step 2: Extend `exports_test.jl`**

Modify `test/exports_test.jl`:

```julia
using F1Tenth
using Test

@testset "F1Tenth: public API is exported" begin
    exported_names = [
        :VehicleParams, :State, :Action, :step_dynamics, :simulate, :KS_MODE, :ST_MODE,
        :Track, :Segment, :build_track, :load_track,
        :TrackBVH, :build_bvh,
        :LiDARConfig, :CarFootprint, :car_vertices, :cast_ray, :scan,
        :collides, :check_car_collisions, :check_track_collision,
        :step_multi_agent!,
        :to_lazysets,
        :AbstractRenderer, :NullRenderer, :GLMakieRenderer, :CairoMakieRenderer,
        :init!, :render!, :close!, :reset!,
    ]
    for name in exported_names
        @test isdefined(F1Tenth, name)
    end
end
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS (all testsets from Tasks 1-3, plus every Phase 1/Phase 2
testset still passing unchanged).

- [ ] **Step 4: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl test/exports_test.jl
git commit -m "Export Phase 3 renderer API from F1Tenth module"
```
