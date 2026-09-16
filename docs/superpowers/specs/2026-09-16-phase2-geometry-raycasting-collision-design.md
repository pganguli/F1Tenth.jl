# Phase 2 — Geometry, Raycasting & Collision Engine: Design

## Context

Phase 1 (`src/dynamics/`) built the vehicle dynamics: `State`
(`SVector{7,Float64}`: x, y, δ, v, ψ, ψ̇, β), `Action`
(`SVector{2,Float64}`), and `step_dynamics(state, action, params, dt)`,
a zero-allocation RK4 integrator. Phase 2 adds the track/map
representation, a 2D LiDAR raycaster, and collision detection, and
wires them into a multi-agent per-tick step function — without
modifying any Phase 1 file.

The original 5-phase blueprint suggested `LazySets.jl` and a
"BVH/R-tree spatial index" for this phase. Neither survived contact
with the actual reference implementation (`../f1tenth_gym`,
`gym/f110_gym/envs/laser_models.py` and `collision_models.py`): the
Python reference uses an occupancy-grid bitmap + precomputed Euclidean
distance transform for raycasting (no spatial index at all) and GJK
for vehicle-vehicle collision. This project explicitly does not chase
Python API or algorithmic fidelity — only behavioral fidelity where
called for — so this phase departs from the reference's grid/GJK
approach in favor of a vector/polygon track representation with a real
spatial index, chosen through the brainstorming session (rationale
inline below). `LazySets.jl` reappears, but only as an export target
for Phase 5 (`ReachabilityAnalysis.jl` consumes `LazySets` set types
for hybrid-system guards/invariants) — it plays no role in the hot
collision/raycasting path.

## Non-goals

- No occupancy-grid/bitmap map support (dropped in favor of vector
  track geometry — see Decision 1).
- No numeric fidelity to the Python reference's EDT-based raycasting
  or GJK collision algorithms — different algorithms, same *purpose*
  (a scan array, a collision boolean).
- No changes to any file under `src/dynamics/`.
- No Phase 3 (rendering) or Phase 4 (concurrency/channels) work. The
  combined step function below is a plain synchronous multi-agent
  step; Phase 4 will decide how it's scheduled across threads/channels.

## Decisions

### 1. Track format: centerline + variable per-point width

A `Track` is built from an ordered sequence of centerline waypoints,
each with an independent left-width and right-width (not a single
scalar track width) so the corridor can narrow/widen along its
length — e.g. a chicane or pit-lane splitter. Loaded from a CSV/table
with columns `x, y, w_left, w_right` (plus optional heading/curvature
columns reserved for future use, ignored by Phase 2).

At load time, `Track` derives two boundary polylines (inner, outer) by
offsetting each centerline waypoint along its local normal by
`w_left`/`w_right`, then flattens both polylines into a single static
`Vector{Segment}` (`Segment` = a small immutable struct of two 2D
points) used by every downstream geometry query. This flattening
happens once; nothing in the hot path re-derives segments from the
centerline.

### 2. Spatial index: static KD-tree over segment midpoints

`NearestNeighbors.jl`'s KD-tree, built once per `Track` over the
midpoint of every boundary segment. This is a new direct dependency
(`Project.toml`). Chosen over a hand-rolled BVH (more code to
maintain, no accuracy/perf benefit at this segment count) and over
`RegionTrees.jl` (quadtree; no ray-walking support, needs the same
amount of custom traversal code with a less common API). Chosen over
routing through `LazySets.jl` (oriented around convex-set operations,
not segment spatial indexing — see Decision 5).

The KD-tree only ever indexes *static* boundary segments. Other
agents' body edges are dynamic (move every tick) and few (small agent
counts), so they're tested directly with no index — see Decision 4.

### 3. Raycasting: sample-marched KD-tree queries + exact segment intersection

For a single ray (origin, direction, `max_range`): step a sample point
along the ray at increasing distance, at each sample query the KD-tree
for boundary segments within a small radius, run exact ray-segment
intersection against those candidates, and take the closest confirmed
hit. If no hit is found by `max_range`, the beam reports `max_range`.

`LiDARConfig` holds `num_beams`, `fov`, `max_range`, and a Gaussian
noise standard deviation, all tunable fields (not hardcoded), with
defaults mirroring the Python reference (1080 beams, 4.7 rad FOV, 30m
range) purely as sane starting values — no fidelity claim is made
about matching Python's scan output bit-for-bit.

`scan(state::State, track::Track, other_agents::Vector{State}, params::VehicleParams, cfg::LiDARConfig) -> Vector{Float64}`
computes one full scan for one agent: for each beam angle, intersect
against both the track's boundary segments (via the KD-tree) and every
other agent's 4 body edges (via `get_vertices`-equivalent, computed
fresh each call, no index), taking the overall closest hit. This
reproduces the Python reference's *behavior* of other cars occluding
the LiDAR, without reusing its algorithm.

### 4. Collision detection: SAT (car-vs-car), segment intersection (car-vs-track)

**Car-vs-car**: Separating Axis Theorem over each car's 4-vertex
rectangle body (from pose + length + width, same vertex construction
as the Python reference's `get_vertices`). Chosen over GJK: for a fixed
4-vertex convex body, SAT is simpler to implement and verify (project
onto each edge normal, check for a gap) with no meaningful performance
difference at this vertex count. `collides(rect_a, rect_b) -> Bool`
is pairwise; an all-pairs check (`check_collisions(states, params) ->
Vector{Bool}`, O(n²), fine for small multi-agent counts) loops it over
the full agent list.

**Car-vs-track**: test the car's 4 body edges against the track's
boundary segments, reusing the same KD-tree broad-phase and exact
segment-intersection routine the raycaster uses — one geometry
primitive serving both features, rather than a second unrelated
algorithm (the Python reference's iTTC-from-scan approach doesn't fit
now that boundaries are explicit polygons with a spatial index already
built).

### 5. LazySets export for Phase 5

`to_lazysets(track::Track) -> Vector{LazySets.VPolygon}`: decomposes
the track corridor into one convex polygon per centerline segment
"slice" (the quad formed by two consecutive waypoints' inner/outer
boundary points). This is the only place `LazySets.jl` appears in
Phase 2 — a thin conversion function, not the track's core
representation — so Phase 5's `ReachabilityAnalysis.jl` hybrid-system
work can intersect reachable flowpipes against track-corridor
constraints without Phase 2's hot path depending on `LazySets` at all.

### 6. Integration with Phase 1: a new combined step function, Phase 1 untouched

A new function (working name `step_multi_agent!`, exact name TBD at
implementation time) in Phase 2's own module:

```
step_multi_agent!(states::Vector{State}, actions::Vector{Action},
                   track::Track, params::VehicleParams, dt::Float64,
                   cfg::LiDARConfig)
    -> (new_states::Vector{State}, scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
```

For each agent: call Phase 1's `step_dynamics` unchanged to advance
its state, then (with the *updated* states of all agents) compute that
agent's LiDAR scan and its collision status (car-vs-track OR car-vs-
any-other-car). Returns per-agent states, scans, and collision flags
for the full agent list in one call.

This function owns the "what happens once per tick, across all
systems" concern; it does not decide *how* that tick is scheduled
across threads or pipeline stages — that's Phase 4's job, which may
wrap or replace this function once its Channel-based architecture
exists. Single-agent use is just this function called with
one-element vectors.

## File layout

```
src/geometry/
  track.jl            # Track type, CSV loading, boundary derivation, segment flattening
  bvh.jl              # KD-tree construction over segment midpoints (NearestNeighbors.jl)
  raycast.jl          # ray-segment intersection primitive, sample-marched scan() 
  collision.jl        # SAT car-vs-car, car-vs-track segment test, get_vertices equivalent
  step2.jl            # step_multi_agent!, wiring Phase 1's step_dynamics + this phase's geometry
  lazysets_export.jl  # to_lazysets(track) for Phase 5
test/geometry/
  track_test.jl
  bvh_test.jl
  raycast_test.jl
  collision_test.jl
  step2_test.jl
```

`Project.toml` gains `NearestNeighbors` and `LazySets` as direct
dependencies.

## Testing strategy

- **track.jl**: boundary-offset correctness on hand-computed small
  cases (straight segment, varying width, a curve where inner/outer
  offsets must not self-intersect); segment count matches waypoint
  count.
- **bvh.jl**: KD-tree query returns the correct known-nearest segment
  for hand-placed query points.
- **raycast.jl**: exact ray-segment intersection on hand-computed
  cases (hit, miss, parallel, endpoint-touching); `scan()` against a
  simple rectangular track produces known beam lengths; other-agent
  occlusion verified with a synthetic blocking car.
- **collision.jl**: SAT on axis-aligned and rotated rectangles
  (overlapping, touching, separated); car-vs-track collision at a
  known-penetrating pose vs. a known-clear pose.
- **step2.jl**: integration test — two agents, one driven into the
  wall (expect collision flag), one driven into the other (expect
  collision flag), confirms Phase 1's state evolution is unaffected by
  Phase 2 wiring (no change to `step_dynamics` outputs).

## Open questions for the implementation plan (not blocking this spec)

- Exact `Segment`/`Track` struct field names and whether they follow
  the Unicode-notation convention established in Phase 1 (likely yes
  for physical quantities like width; TBD for structural/index
  fields).
- Whether `to_lazysets` slice decomposition needs to handle
  self-intersecting quads at sharp curvature (guard or assume
  well-formed input tracks) — deferred to the plan/implementation
  since it depends on what track data is actually available for
  testing.
