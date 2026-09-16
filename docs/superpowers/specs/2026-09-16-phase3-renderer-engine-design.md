# Phase 3 — Decoupled Abstract Renderer Engine: Design

## Context

Phase 1 (`src/dynamics/`) provides `State` (`SVector{7,Float64}`: x, y,
δ, v, ψ, ψ̇, β) and `step_dynamics`. Phase 2 (`src/geometry/`) adds
`Track`/`Segment` (centerline + inner/outer boundary polylines +
flattened boundary segments), `CarFootprint`, `TrackBVH`, `scan`, and
`step_multi_agent!(states, actions, bvh, params, car, dt, cfg) ->
(new_states, scans, collisions)` — a stateless, synchronous per-tick
step over all agents.

Phase 3 adds a rendering layer that consumes exactly that per-tick
output (`new_states`, `scans`, `collisions`) plus the `Track` used to
build the `bvh`, and draws it — without modifying any Phase 1 or
Phase 2 file, and without assuming anything about how Phase 4's
not-yet-designed concurrency/channel model will schedule calls into
it. The blueprint names three renderer types: `NullRenderer`,
`GLMakieRenderer` (live/interactive), `CairoMakieRenderer`
(static/vector output). All three exist behind one `AbstractRenderer`
interface so a caller (a training loop, a demo script, a test) can
swap renderers without changing its sim loop.

## Non-goals

- No changes to any file under `src/dynamics/` or `src/geometry/`.
  `step_multi_agent!`'s signature and statelessness are unchanged;
  Phase 3 renderers hold their own mutable state (trail history,
  Observables, output paths) entirely on their own side of the
  interface.
- No Phase 4 (concurrency/channels) design. The interface below is
  called synchronously, once per tick, by whatever loop Phase 4
  eventually builds; how that loop is threaded is out of scope here.
- No video encoding (mp4/gif) and no ffmpeg dependency. CairoMakie's
  deliverable is vector (SVG) output only, per Decision 4.
- No single self-contained animated-SVG (SMIL) file. Deferred; see
  Decision 4.
- No numeric-fidelity claim to `../f1tenth_gym`'s
  `rendering.py`/pyglet-based renderer — this is a from-scratch
  design against Phase 1/2's actual data shapes, not a port.

## Decisions

### 1. Interface: `init!` / `render!` / `close!` / `reset!`

```julia
abstract type AbstractRenderer end

init!(renderer::AbstractRenderer, track::Track) -> renderer
render!(renderer::AbstractRenderer, states::Vector{State},
        scans::Vector{Vector{Float64}}, collisions::Vector{Bool}) -> nothing
close!(renderer::AbstractRenderer) -> nothing
reset!(renderer::AbstractRenderer, track::Track) -> renderer
```

`render!`'s three positional arguments are exactly `step_multi_agent!`'s
return shape (`new_states, scans, collisions`), so a call site is
`render!(renderer, step_multi_agent!(...)...)`. `init!` performs
one-time setup (build a window/canvas, store the track, clear any
history); `reset!` re-runs that setup for a new episode against a
(possibly different) track *without* reconstructing the renderer
object — relevant for training loops that reset often and don't want
to reopen a GLMakie window every episode. `close!` tears down
(flushes files, closes the window).

`render!` called before `init!`/`reset!` throws a clear
`ErrorException` for `GLMakieRenderer`/`CairoMakieRenderer` (tracked
via an internal `initialized::Bool` flag) — except `NullRenderer`,
which is always a safe no-op regardless of lifecycle state, so
training loops that don't care about rendering never have to think
about it.

Agent count is assumed fixed within one episode, i.e. between
`reset!` calls — a renderer is not required to handle
`length(states)` changing between two `render!` calls without an
intervening `reset!`. This mirrors how `CarFootprint` is also supplied
once (at construction, not per-tick — see Decision 3) rather than
threaded through every call.

### 2. `NullRenderer`: zero-cost no-op

```julia
struct NullRenderer <: AbstractRenderer end
```

All four interface functions are immediate no-ops that return without
touching their arguments. This is not merely a test stub: RL/training
loops that step thousands of times per second need a renderer that
costs nothing, and `NullRenderer`'s methods are simple enough that
Julia's compiler can inline them away entirely. Lives in the main
module, no extra dependency.

### 3. `GLMakieRenderer` / `CairoMakieRenderer`: package extensions, not core deps

`GLMakie` and `CairoMakie` are both heavy dependencies. Phase 2 flagged
exactly this tradeoff for `LazySets.jl` (`lazysets_export.jl`'s
deferred-minor about a weakdep) and deferred it; Phase 3 does not
defer it, because unlike `LazySets` (an ~8-line file), rendering is
this phase's entire deliverable and a `NullRenderer`/headless/training
user must not be forced to install or precompile either Makie backend.

Mechanism: the main module declares abstract types and empty
constructor stubs only —

```julia
function GLMakieRenderer end
function CairoMakieRenderer end
```

— plus the four interface generics from Decision 1. Two new Julia
package extensions, `ext/F1TenthGLMakieExt.jl` and
`ext/F1TenthCairoMakieExt.jl`, each add a method to their stub
constructor and define a private concrete struct (`<: AbstractRenderer`,
not exported — accessed only through the exported stub function name)
plus methods for `init!`/`render!`/`close!`/`reset!` dispatched on
that struct. `GLMakie`/`CairoMakie` become `[weakdeps]` +
`[extensions]` entries in `Project.toml`; they are only loaded,
compiled, and their code paths only reachable, when the *caller's*
environment also has that package loaded (`using GLMakie` or `using
CairoMakie` alongside `using F1Tenth`).

A `CarFootprint` is supplied once at each renderer's construction
(`GLMakieRenderer(track; car=CarFootprint())`,
`CairoMakieRenderer(track; ...)`) rather than passed to `render!`,
matching how Phase 2 already threads one fixed `CarFootprint` through
`step_multi_agent!` for the whole sim run.

`render!`'s `scans` argument carries only the numeric range per beam —
it has no way to know the beam *angles* used to produce them, since
those depend on `LiDARConfig.fov` (`scan`'s beams span `ψ - fov/2` to
`ψ + fov/2`), which isn't part of `render!`'s three arguments. Drawing
scan rays therefore requires the angle-defining config too. Rather than
widen `render!`'s signature (which would break the direct
`render!(renderer, step_multi_agent!(...)...)`  call shape from
Decision 1), a `LiDARConfig` is supplied once at construction, exactly
like `CarFootprint`: `GLMakieRenderer(track; car=CarFootprint(),
cfg=LiDARConfig())`. This assumes `num_beams`/`fov` don't change
mid-episode — true of every current caller (`cfg` is itself constructed
once and threaded unchanged through a whole `step_multi_agent!` run) —
and reconstructs each beam's angle the same way `scan` does:
`range(ψ - cfg.fov/2, ψ + cfg.fov/2; length=cfg.num_beams)` (or the
single-beam `[ψ]` case when `cfg.num_beams == 1`), applied to that
tick's `states[i]` heading `ψ`. `CairoMakieRenderer` takes the same
`cfg` keyword for the same reason, since its buffered history also
needs beam angles to draw scan rays in the replayed frames.

### 4. `GLMakieRenderer`: Observable-based incremental live view

`GLMakieRenderer(track; camera_mode=:fixed, ego_index=1,
car=CarFootprint())`. `init!`/`reset!` build the `Figure`/`Axis` once,
plot the static track boundaries once (they never change within an
episode), and wrap everything that *does* change —
per-car body polygon, per-car trail polyline, per-car scan rays,
per-car collision-flag color — in Makie `Observable`s. `render!` only
pushes new values into those `Observable`s each tick; it never
re-plots from scratch. This was chosen over full clear-and-redraw
specifically because Phase 2 already measured the raycaster at
~32–54 Hz (`raycast.jl`'s documented throughput) — adding a
full-scene-replot cost on top of that on every tick risks making
rendering the new bottleneck instead of LiDAR.

`camera_mode` is configurable: `:fixed` (default) sets axis limits
once, from the track's bounding box, at `init!`/`reset!` time and
never moves them; `:follow` recenters the axis on `states[ego_index]`
each `render!` call. Both modes are supported by the same struct
rather than two renderer types, since the only difference is which
axis-limit update (if any) runs inside `render!`.

Per-car trail history (a growing `Vector{Point2f}` per car, appended
each tick, backing the trail-line `Observable`) is the renderer's own
state — `step_multi_agent!` stays stateless and unaware trails exist.
`reset!` clears these buffers; `init!` on first construction starts
them empty.

**Testing** (headless CI): exercised via GLMakie's offscreen/headless
screen mode (`GLMakie.Screen(...; visible=false)`, no window ever
shown), asserting the underlying `Observable`s' `[]` values update
correctly across a short synthetic multi-tick sequence — not by
capturing or comparing rendered pixels. This requires the CI
environment to provide *some* working OpenGL context (e.g. Mesa
software rendering under Xvfb); if the execution environment truly
cannot create one, that specific test file's tests skip with a
warning rather than failing the whole suite. `NullRenderer` and
`CairoMakieRenderer` (Decision 5) do not depend on this and are always
fully tested.

### 5. `CairoMakieRenderer`: buffer-then-draw, vector (SVG) output only

`CairoMakieRenderer(track; output_dir=nothing, car=CarFootprint())`.
Unlike `GLMakieRenderer`, `render!` does **not** draw anything — it
only appends the tick's `(states, scans, collisions)` into cheap
internal `Vector`-of-`Vector` history buffers. All drawing happens in
`close!`:

1. **Always**: build one final `Figure` from the complete accumulated
   history — track boundaries, every car's full trail as a polyline,
   final-tick car-body polygons, collided cars flagged by color — and
   save it as `summary.svg`. If `close!` is called with zero recorded
   ticks (no `render!` calls happened), this still succeeds and
   produces a track-only SVG rather than erroring.
2. **If `output_dir` was given at construction**: additionally replay
   the buffered history tick-by-tick, drawing one fresh `Figure` per
   tick (track + every car's trail-so-far + that tick's car
   positions/scan rays) and saving `frame_0001.svg`, `frame_0002.svg`,
   … into `output_dir` — a vector-only "poor man's animation" a viewer
   can step through, per the project owner's request to use
   CairoMakie specifically for its vector strengths.

`reset!` clears the history buffers (and, if constructed with
`output_dir`, does not delete previously-written frame files from a
prior episode — each episode's `close!` is expected to use a
fresh/distinct `output_dir` if the caller wants per-episode frame
sets kept separate; not enforced by the renderer).

Explicitly deferred (not built in this phase): a single
self-contained animated-SVG (SMIL `<animate>`) file. Makie's Cairo
backend has no native support for it; hand-rolling one is real
implementation work disproportionate to Phase 3's scope. Noted here so
a future phase doesn't need to re-litigate the option.

**Testing**: fully exercised in normal CI (CairoMakie is headless-safe,
no GPU/display needed). Full-lifecycle test against a temp directory:
assert `summary.svg` exists, is non-empty, and starts with a valid SVG
header; when `output_dir` is given, assert the per-tick frame file
count matches the number of `render!` calls.

## Data flow

Typical single-episode usage:

```julia
track = load_track(path)
car = CarFootprint()
bvh = build_bvh(track)
renderer = GLMakieRenderer(track; camera_mode=:follow, ego_index=1, car=car)
init!(renderer, track)
states = initial_states
for tick in 1:n_ticks
    actions = policy(states)
    states, scans, collisions = step_multi_agent!(states, actions, bvh, params, car, dt, cfg)
    render!(renderer, states, scans, collisions)
end
close!(renderer)
```

Multi-episode reuse (e.g. inside a training loop) replaces the
`init!(renderer, track)` line with `reset!(renderer, track)` on every
episode after the first, and calls `close!(renderer)` once at the very
end rather than per episode — this is precisely why `reset!` exists as
a distinct operation from `init!`/`close!` (Decision 1).

## File layout

```
src/render/
  interface.jl      # AbstractRenderer, init!/render!/close!/reset! generics,
                     # NullRenderer, GLMakieRenderer/CairoMakieRenderer stubs
ext/
  F1TenthGLMakieExt.jl
  F1TenthCairoMakieExt.jl
test/render/
  testutils.jl       # shared small-track fixture
  null_test.jl
  glmakie_test.jl
  cairomakie_test.jl
```

`src/F1Tenth.jl` gains one new `include("render/interface.jl")` and
exports `AbstractRenderer, NullRenderer, GLMakieRenderer,
CairoMakieRenderer, init!, render!, close!, reset!`. `Project.toml`
gains `GLMakie`/`CairoMakie` under `[weakdeps]` and a matching
`[extensions]` table; `test/exports_test.jl` is extended to cover the
new exported names, and the test environment (`[extras]`/test
`Project.toml`) adds `GLMakie`/`CairoMakie` as real test-only
dependencies so both extensions are actually exercised in CI despite
being weakdeps at the package level.

## Open questions / deferred items for a future phase

- Single self-contained animated SVG (SMIL) — deferred, see Decision 5.
- Video (mp4/gif) export — explicitly out of scope, see Non-goals.
- How Phase 4's concurrency model calls `render!` (which thread/tier
  of the 3-tier Channel pipeline owns it, whether render calls can be
  dropped/coalesced under backpressure) is entirely undecided here and
  left for Phase 4's own brainstorming.
