# Handoff: F1Tenth.jl

Written for: a fresh agent picking up this project with no memory of prior
sessions. Read this before touching anything.

## What this is

A from-scratch Julia port of the Python `f1tenth_gym` simulator (source at
`../f1tenth_gym` relative to this repo, i.e. a sibling directory — not
part of this package, kept only as the reference implementation to port
from and validate against). No Python API compatibility is required or
attempted — only numeric/behavioral fidelity where the spec calls for it.

The full 5-phase blueprint (given once, not otherwise saved in the repo):

- **Phase 1 — Core Dynamics & Symbolic System.** `ModelingToolkit.jl`
  single-track kinematic/dynamic bicycle ODEs, `DifferentialEquations.jl`
  numerical solves, an allocation-free `SVector`-based discrete step
  function. **Done, merged to `main`.**
- **Phase 2 — Geometry, Raycasting & Collision Engine.** Vector track
  geometry, a KD-tree spatial index, a 2D LiDAR raycaster and collision
  detector. **Done, merged to `main`.** (The original blueprint's
  "LazySets.jl map parsing, BVH/R-tree" guess was revised during
  brainstorming — see below; `LazySets.jl` still appears, but only as a
  Phase-5-facing export, not the core representation.)
- **Phase 3 — Decoupled Abstract Renderer Engine.** `AbstractRenderer`
  with `NullRenderer`, `GLMakieRenderer`, `CairoMakieRenderer`
  implementations. **Not started.**
- **Phase 4 — Async Concurrency & Control Channels.** Multi-threaded
  sim loop, 3-tier `Channel` pipeline (input / physics / render). **Not
  started.**
- **Phase 5 — Reachability Analysis & Telemetry Export.**
  `ReachabilityAnalysis.jl` hybrid-system reachability over the Phase 1
  dynamics, `Arrow.jl` telemetry logging. **Not started.**

## Process this project follows

Every phase goes through: **brainstorming → written spec → written plan →
subagent-driven-development execution**, using the Superpowers skill
plugin (`superpowers:brainstorming`, `superpowers:writing-plans`,
`superpowers:subagent-driven-development`). This is deliberate and the
project owner is invested in it — don't skip straight to code. Specs live
in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`, each
named `YYYY-MM-DD-<topic>-design.md` / `YYYY-MM-DD-<topic>.md`.

Phase 1 and Phase 2's documents are the reference examples of the
expected depth:
- `docs/superpowers/specs/2026-09-16-phase1-core-dynamics-design.md` /
  `docs/superpowers/plans/2026-09-16-phase1-core-dynamics.md`
- `docs/superpowers/specs/2026-09-16-phase2-geometry-raycasting-collision-design.md`
  / `docs/superpowers/plans/2026-09-16-phase2-geometry-raycasting-collision.md`

subagent-driven-development keeps a ledger at
`.superpowers/sdd/<plan-basename>/progress.md` inside whatever worktree
is executing the plan (git-ignored, deleted after a clean final review —
so neither Phase 1's nor Phase 2's ledger still exists; their rulings are
summarized below since they matter beyond one execution).

## Phase 1 — what actually got built

Files: `src/dynamics/{params,mtk_models,step,solve}.jl`, `src/F1Tenth.jl`
(includes + exports), `test/{params,mtk_models,step,solve}_test.jl`.

- `params.jl` — `VehicleParams` (immutable struct, Unicode field names —
  see Notation below), `steering_constraint`/`accl_constraint` (control
  input clamps, ported from Python; note `accl_constraint` is
  deliberately singular, unlike Python's inconsistent
  `steering_constraint`/`accl_constraints`).
- `mtk_models.jl` — `KS_MODE`/`ST_MODE`: two `ModelingToolkit` `System`s
  sharing one 7-dimensional state space `(x, y, δ, v, ψ, ψ̇, β)`.
  `KS_RHS`/`ST_RHS`: allocation-free generated numerical functions, built
  from these at module `__init__()` time (not precompile time — see
  Gotchas below). **These stay deliberately uncompiled** — do not run
  `mtkcompile`/`structural_simplify` on them without reading the comment
  at their definition first.
- `step.jl` — `State`/`Action` type aliases (`SVector{7,Float64}` /
  `SVector{2,Float64}`), `dynamics_rhs` (mode dispatch + the low-speed
  correction below), `step_dynamics` (hand-written zero-allocation RK4).
- `solve.jl` — `simulate`, a thin `DifferentialEquations.jl`/`Tsit5`
  wrapper around the same `dynamics_rhs`, for offline/adaptive-accuracy
  use.

### Load-bearing facts a fresh agent must not re-derive from scratch

1. **The low-speed regime is NOT just "KS mode with the ψ̇/β
   derivatives zeroed."** Python's `vehicle_dynamics_st`, when
   `|v| < 0.5`, zeros only the slip-angle derivative (`β̇`); the yaw-rate
   state's own derivative (`ψ̈`) uses a specific nonzero formula. Task 2's
   `KS_RHS`/`KS_MODE` zero *both* (correct for what they actually
   represent: the standalone Python `vehicle_dynamics_ks`, a true 5-state
   model). The correction lives **only** in `step.jl`'s `dynamics_rhs`
   (see the comment there), not in `KS_RHS` — this was deliberate, so
   `KS_MODE`/`ST_MODE` stay usable as clean hybrid-system-mode building
   blocks for Phase 5. If you're touching Phase 5's reachability work,
   build the `HybridSystem` from `KS_MODE`/`ST_MODE` directly, not from
   `dynamics_rhs`.
2. **`mtkcompile(KS_MODE)` silently drops to 5 unknowns** (eliminates
   `ψ̇`/`β` since their raw equations are `D(ψ̇) ~ 0`/`D(β) ~ 0`), while
   `ST_MODE` stays at 7. This breaks the "both modes share one state
   space, no reset map needed" premise if you compile before assembling a
   `HybridSystem`. Documented in a comment at `mtk_models.jl` and covered
   by a test in `mtk_models_test.jl` (`"KS_MODE / ST_MODE share one
   7-dimensional state space"`).
3. **The spec's `select_mode(state, params) -> Val{:ks}/Val{:st}` seam
   was never implemented as a separate function** — `dynamics_rhs`
   inlines `abs(v) < 0.5` directly (ruled correct: YAGNI, avoids dispatch
   overhead). If Phase 5 wants that named seam, it doesn't exist yet.
4. **`Manifest.toml` is gitignored.** A fresh checkout/worktree needs
   `julia --project=. -e 'import Pkg; Pkg.resolve(); Pkg.instantiate()'`
   before tests will run — don't mistake the resulting `ERROR:
   DifferentialEquations is a direct dependency, but does not appear in
   the manifest` for a real bug.
5. **`test/mtk_models_test.jl` deliberately uses plain ASCII local
   variable names** (`mu`, `lf`, `lr`, `I`, `C_Sf`, `C_Sr`, `a_max`, …),
   decoupled from `VehicleParams`'s Unicode field names — confirmed
   correct in three separate reviews during Phase 1. Don't "fix" this.
6. **`ModelingToolkit.build_function(...; expression=Val{false})` at
   package top-level breaks across precompilation** (a
   `RuntimeGeneratedFunctions` cache `KeyError` in a fresh process). The
   fix pattern used here: build with `expression=Val{true}` to get an
   `Expr`, then `@eval` it into a proper `const` binding inside the
   module's `__init__()`, not at top-level `include` time. See
   `mtk_models.jl`'s comments if you add another symbolic function.

### Notation convention (established over many iterative rename passes — follow it for new code)

Physical quantities use Julia's native Unicode identifiers, matching
standard vehicle-dynamics/math notation, not ASCII abbreviations:
`μ` (friction), `Cᵦᶠ`/`Cᵦʳ` (cornering stiffness), `lᶠ`/`lʳ` (axle
distances), `Iᶻ` (yaw inertia), `δ`/`δ̇` (steering angle/rate) with
`δ⁻`/`δ⁺`/`δ̇⁻`/`δ̇⁺` bounds (superscript minus/plus, not subscript),
`v⁻`/`v⁺`/`vₛ` (velocity bounds/switch), `a⁺` (max accel), `ψ`/`ψ̇`/`ψ̈`
(yaw/yaw-rate/yaw-angular-accel — note `ψ̈` uses the Unicode combining
diaeresis, the standard second-derivative notation), `β` (slip angle),
`𝐱`/`𝐱₀` (bold x, state vector / initial state), `k₁`..`k₄` (RK4 stage
subscripts), `lʷᵇ` (wheelbase, superscript w/b). Every renamed symbol has
a one-line comment explaining its physical meaning near its first
declaration in `params.jl` and `mtk_models.jl` — extend that pattern for
new symbols.

### Known deferred minors (not blocking, listed so you don't re-flag them)

- `test/mtk_models_test.jl`'s `accl_ref_clamp` local helper only covers
  the positive-acceleration branch (sufficient for its two call sites).
- `solve.jl`'s `State(u)` conversion inside the `ODEProblem` closure is a
  likely-redundant no-op convert.
- Minor comment-alignment whitespace drift in `params.jl`/`mtk_models.jl`
  from several rounds of identifier-width changes.
- `dynamics_rhs` destructures `𝐱` into all 7 named components even though
  only some are used per branch — intentional, documents the state layout
  at zero runtime cost.

## Phase 2 — what actually got built

Files: `src/geometry/{track,bvh,raycast,collision,step2,lazysets_export}.jl`,
`test/geometry/{track,bvh,raycast,collision,step2,lazysets_export}_test.jl`,
`test/exports_test.jl`. `src/F1Tenth.jl` updated (includes + exports).

The blueprint's original guess of "`LazySets.jl` map parsing, BVH/R-tree"
did not match how the actual work turned out — brainstorming replaced the
Python reference's occupancy-grid/EDT raycasting and GJK collision with a
vector track representation, a KD-tree, SAT, and segment intersection.
`LazySets.jl` survives only as a Phase-5-facing export function.

- `track.jl` — `Segment` (two `SVector{2,Float64}` endpoints), `Track`
  (centerline + **independent per-waypoint** `w_left`/`w_right`, not a
  single scalar width — the corridor can narrow/widen along the track).
  `build_track`/`load_track` derive inner/outer boundary polylines by
  offsetting each centerline point along its local normal, then flatten
  both into one static `Vector{Segment}`. **Assumes a closed-loop,
  counterclockwise-wound centerline** — see Gotchas below.
- `bvh.jl` — `TrackBVH` (misnomer inherited from the plan's file-naming;
  it's actually a `NearestNeighbors.jl` `KDTree` over segment midpoints,
  not a bounding-volume hierarchy — don't expect real BVH semantics),
  `build_bvh`. Only indexes *static* boundary segments; other agents'
  body edges are dynamic and few, so they're tested directly with no
  index (see `raycast.jl`'s `_other_agent_segments`).
- `raycast.jl` — `LiDARConfig` (defaults: 1080 beams, 4.7 rad FOV, 30 m
  range — Python-reference-shaped defaults, no fidelity claim),
  `CarFootprint` (defaults: length=0.58, width=0.31 — a new type, since
  Phase 1's `VehicleParams` has no body-footprint fields and Phase 1
  files couldn't be touched), `car_vertices`, `cast_ray` (sample-marched
  ray-segment intersection, KD-tree-pruned, provably conservative —
  margin = `step + max_half_length` is proven sufficient to never miss
  the true nearest hit), `scan` (full LiDAR sweep; **other agents'
  car bodies occlude beams**, a deliberate design decision, not a
  Python-fidelity artifact). Scan output is clamped to `[0, max_range]`
  after noise.
- `collision.jl` — `collides` (SAT, for convex polygons, used with
  4-vertex rectangles; touching-but-not-overlapping counts as
  colliding, by design), `check_car_collisions` (all-pairs, O(n²)),
  `check_track_collision` (car body edges vs. boundary segments, reusing
  the same KD-tree broad-phase as the raycaster; query radius is the
  car's true bounding-circle radius, `hypot(length,width)/2 +
  max_half_length` — unconditionally sufficient for any rectangle).
- `step2.jl` — `step_multi_agent!(states, actions, bvh, params, car, dt,
  cfg; rng=Random.default_rng())`: calls Phase 1's `step_dynamics`
  per-agent unmodified, then computes each agent's scan (against
  post-step states of all agents) and collision status (car-vs-track OR
  car-vs-any-other-car). Returns `(new_states, scans, collisions)` for
  the whole agent list in one call. **Does not mutate its inputs**
  despite the `!` — inherited from the spec's naming, not a Julia
  convention violation introduced here; worth renaming before any
  caller relies on the `!` implying in-place mutation.
- `lazysets_export.jl` — `to_lazysets(track) -> Vector{LazySets.VPolygon}`,
  one convex quad per centerline slice, for Phase 5's
  `ReachabilityAnalysis.jl` guards/invariants. This is the *only* file
  that touches `LazySets`; the hot raycasting/collision path never does.

### Load-bearing facts a fresh agent must not re-derive from scratch

1. **The track must be a closed loop, wound counterclockwise (CCW).**
   `build_track`'s outward-normal computation
   (`normal_outward = SVector(tangent[2], -tangent[1])`, a −90° rotation
   of the direction-of-travel tangent) only produces the geometrically
   correct outward normal for a CCW-wound centerline. For a
   clockwise-wound centerline, `inner`/`outer` (and therefore
   `w_left`/`w_right`) silently swap — no error, just a mirrored
   corridor. Documented in a comment at the computation; if you add
   support for real-world track CSVs (e.g. TUM/f1tenth format, which
   has distinct left/right widths), verify winding direction first.
2. **This departed from Python fidelity on purpose.** The Python
   reference's occupancy-grid + Euclidean-distance-transform raycasting
   and GJK collision were deliberately *not* ported — brainstorming
   concluded they don't match this project's "no Python API/algorithm
   fidelity required" stance, and a vector/KD-tree/SAT approach is
   simpler to reason about and test. Don't be surprised the geometry
   code looks nothing like `laser_models.py`/`collision_models.py` in
   `../f1tenth_gym`.
3. **Default `LiDARConfig`'s march `step` (0.1, hardcoded as a `cast_ray`
   keyword default) is not exposed on `LiDARConfig` and not tunable from
   `scan`.** Measured throughput at the shipped defaults (1080 beams):
   **~32–54 Hz single-agent** (worst case / typical case), well under a
   100 Hz sim-tick target. Not a bug — the spec chose this algorithm
   knowingly — but Phase 3/4 will hit this. The cheapest lever is adding
   a `step` field to `LiDARConfig` and threading it through `scan` to
   `cast_ray`.
4. **`CarFootprint` is a new Phase 2 type, not an extension of
   `VehicleParams`.** `VehicleParams` (Phase 1) only has axle distances
   (`lᶠ`/`lʳ`), not a physical body length/width — Phase 1 files
   couldn't be touched, so collision/raycasting geometry carries its own
   footprint type. If Phase 1 ever needs to know about car dimensions,
   this split will need revisiting.
5. **`to_lazysets`'s `VPolygon` construction silently reshapes
   self-intersecting quads rather than erroring.** `LazySets.VPolygon`'s
   default constructor runs a convex-hull pass on its input vertices
   regardless of winding order — harmless for a well-formed convex quad,
   but for a genuinely self-intersecting quad (possible at sharp track
   curvature) it silently drops the crossing vertex and returns a
   smaller, differently-shaped polygon instead of failing loudly. Not
   exercised by any current test (smooth circular test track); flagged
   as an open question in the Phase 2 spec, deferred to whoever first
   feeds this real, sharply-curved track data (likely Phase 5).

### Known deferred minors (not blocking, listed so you don't re-flag them)

- `build_bvh` has no guard against an empty `Vector{Segment}` (throws
  `ArgumentError` from `maximum` over an empty collection) — unreachable
  from `build_track` (any track produces ≥2 segments) and from every
  current caller.
- No dedicated test isolates the parallel-ray-to-segment branch in
  `_ray_segment_intersection`, or a "ray misses everything" case
  distinct from the max-range clamp — both are implicitly exercised by
  the existing square-room tests.
- `cast_ray` has no guard against `step <= 0` (would infinite-loop) —
  not reachable from any current caller, all of which use positive
  defaults.
- `_cast_ray_against_segments` recomputes `dir` from `angle`, duplicating
  work `scan`'s own loop already did for the same beam — nanoseconds
  against an ~18 ms scan, not worth fixing yet.
- `test/geometry/raycast_test.jl` and `test/geometry/step2_test.jl` both
  define an identical `_square_room_bvh` test helper, producing a
  cosmetic "Method definition overwritten" Julia warning during test
  runs (harmless — both copies are byte-identical). Candidate for a
  shared `test/geometry/testutils.jl` if either copy is ever edited.
- `_segments_intersect` returns `false` for exactly-collinear-overlapping
  segments (the `abs(rxs) < 1e-12` early-return guard) — a car edge
  flush along a wall would report no collision. Measure-zero in
  practice, not exercised by any test.
- `build_track` produces `NaN` silently for a degenerate input (fewer
  than 3 waypoints, or duplicated consecutive waypoints) rather than
  raising a clear error.
- `check_track_collision` cannot detect a car that fully tunnels past a
  wall within one `dt` (only tests edge/segment intersection, not
  interior containment) — a known simulation-step-size limitation, not
  a logic bug; documented in `step2_test.jl`'s test comment but not in
  `check_track_collision`'s own code.
- `test/exports_test.jl` checks `isdefined(F1Tenth, name)` for all
  public names, which only confirms a binding exists somewhere in the
  module — it does **not** verify the name is actually `export`ed /
  accessible unqualified after `using F1Tenth`. A future silently-dropped
  `export` statement entry would not be caught by this test. Fix is a
  one-line predicate swap to `Base.isexported(F1Tenth, name)` or
  `name in names(F1Tenth)`.
- `LazySets` is a fairly heavy dependency for the ~8 lines that use it
  (`lazysets_export.jl`), loaded for every user of Phases 1-4 even
  though only Phase 5 needs it. A Julia package extension (weakdep)
  would defer that cost; reasonable to leave until Phase 5 exists.
- `Segment` (this package's own type) and `LazySets.Segment` share a
  name — harmless today since `lazysets_export.jl` uses qualified
  `import LazySets`, but a future `using LazySets` elsewhere would
  collide.

## Where to go next

Phase 3 (Decoupled Abstract Renderer Engine) hasn't been brainstormed
yet. Start there with `superpowers:brainstorming` — it'll need its own
design conversation (what `AbstractRenderer`/`NullRenderer`/
`GLMakieRenderer`/`CairoMakieRenderer` actually need to draw given
Phase 2's `Track`/`Segment` geometry and `State`/scan data, how a
renderer consumes `step_multi_agent!`'s per-tick output without coupling
to Phase 4's not-yet-designed concurrency model).
