# Frenet-Frame Simulator & tck Track Maps: Design

## Context

Through Phase 5 the simulator is Cartesian. `State` is
`(x, y, δ, v, ψ, ψ̇, β)`; a `Track` is a polyline centerline with per-point
half-widths, expanded into wall `Segment`s that feed a KD-tree (`TrackBVH`)
used by the LiDAR raycaster and wall-collision check. Renderers, `AsyncSim`,
`TelemetrySink` and reachability all take a `Track`.

This change moves the simulator's coordinate frame to **Frenet space** and
replaces the polyline `Track` with a **tck** (SciPy B-spline) map. Maps such
as `Austin_map.png` are converted offline by `tck.py` (`raster_to_tck`),
which fits a periodic cubic B-spline over `[x, y, w]` (centerline and
half-width) parameterized by normalized chord length `u ∈ [0, 1]`, and
saved to CSV.

What Frenet space buys (from the project owner): O(1) checks. Off-track is
`d > w_left(s)` or `d < −w_right(s)`; a lap completes when `s ≥ L`; the gap
between two cars is `s_A − s_B` (wrapped, since `s` is periodic).

## Decisions (settled during brainstorming)

1. **Full Frenet state and dynamics.** The state itself is Frenet, and the
   ODEs are integrated in curvilinear form. (Rejected: Frenet state with
   Cartesian integration; Frenet map queries only.)
2. **LiDAR ray-marches the implicit tck boundary** `|d| = w(s)` with sphere
   tracing plus bisection. No polyline, KD-tree or BVH remains. (Rejected:
   deriving a boundary polyline from the tck. Fallback if the benchmark gate
   fails, see Risks.)
3. **Reachability goes Frenet**, with curvature `κ` carried as a constant
   interval per s-window, in the same style as the existing `a`/`δ̇`
   augmentation. (Rejected: keep reachability Cartesian; drop it from scope.
   Rejected: local polynomial `κ(s)` plus remainder.)
4. **No backward compatibility.** The package is a from-scratch port with no
   API-compat obligation; signatures change freely.

## Non-goals

- No asymmetric widths: the tck carries one half-width, so
  `w_left(s) = w_right(s) = w(s)`. The `d` bounds are written with both names
  so asymmetry can be added later.
- No changes to the vehicle-body dynamics (`δ̇, v̇, ψ̈, β̇`), actuator
  constraints, or the KS/ST switch at `|v| = 0.5`.
- No raster-map loading in Julia; `tck.py` stays the offline converter.
- No rigorous (formally sound) reachability enclosure of `κ(s(t))`; see Risks.

## 1. Map: `Track` becomes a tck

### CSV format

Header row `t,cx,cy,cw`, one row per knot, exactly the arrays SciPy's
`splprep(..., per=True)` returns (`tck = (t, [cx, cy, cw], k)`). Degree
`k = 3` is fixed. SciPy pads each coefficient array to `len(t)`; the loader
uses the first `len(t) − k − 1` entries (verified by a round-trip test
against `scipy.interpolate.splev`). `tck.py` gains a small `save_tck_csv`
writer, added by the project owner or reviewed by them. Units are world
meters, and the loop is counterclockwise.

### `Track`

`Track` keeps its name so downstream `track::Track` signatures survive. It
now holds: knot vector, coefficient matrix, and the arc-length table.

- **Arc length.** `u` is chord-length, not arc length. At load, per knot
  span, integrate `|P′(u)|` with Gauss–Legendre and build a dense `(s_i, u_i)`
  table (`Δs ≈ 0.05 m`). `u(s)` is a cubic Hermite interpolation using the
  exact slope `du/ds = 1/|P′(u)|`. `L` is the total length.
- **Evaluation.** de Boor evaluation of `P(u)`, `P′(u)`, `P″(u)` with an
  allocation-free `SVector` path and a knot-span hint to avoid binary search.
- **Public queries.** `L`, `centerline(s)`, `heading(s)`, `κ(s)`, `w(s)`,
  `w′(s)` (all take unwrapped `s` and apply `mod(s, L)`), plus:
  - `to_cartesian(track, s, d) → (x, y)` = `P(s) + d·N(s)`, with
    `N = (−sin ψ_ref, cos ψ_ref)`.
  - `to_frenet(track, x, y; hint) → (s, d)`: Newton on the foot-point
    condition `(P(u) − p)·P′(u) = 0`, warm-started from `hint` (a previous
    `s`); without a hint, a coarse scan over the arc-length table seeds it.
- **Sign convention.** `+d` is left of the direction of travel.
- **Load-time validation** (loud failure, like `main.jl`'s current check):
  `max|κ|·w_max < 0.9`, `w > 0`, and the fitted curve closes.
- **Boundary sampling.** `track_boundaries(track; n)` returns left/right
  polylines `P(s) ± w(s)·N(s)` for rendering and for `to_lazysets`.
- **Deleted:** `bvh.jl`, `TrackBVH`, `build_bvh`, `build_track` (polyline),
  `NearestNeighbors` dependency. `Segment` is kept for car-vs-car checks.

## 2. State and dynamics

`State = SVector{7}` keeps its shape: `(s, d, δ, v, μ, ψ̇, β)`, with `s, d, μ`
replacing `x, y, ψ`.

- `s` is **unwrapped and increasing**, so `⌊s/L⌋` is the lap count. Wrapping
  happens only inside track queries. Gap between agents A and B:
  `mod(s_A − s_B + L/2, L) − L/2`.
- `μ = ψ − ψ_ref(s)` is the heading error, wrapped to `(−π, π]`.
- Pose kinematics (the only equations that change):

      ṡ = v·cos(μ+β) / (1 − κ(s)·d)
      ḋ = v·sin(μ+β)
      μ̇ = ψ̇ − κ(s)·ṡ

  In the KS branch the same substitution applies with its own `ψ̇` expression.
  `δ̇, v̇, ψ̈, β̇` are exactly the existing expressions.
- `step_dynamics(𝐱, action, params, track, dt)` stays classical RK4;
  `κ(s)` is evaluated at each stage's `s`.
- **Singularity guard.** The denominator `1 − κd` reaches zero at the center
  of curvature. Load-time validation keeps `|κ|w < 0.9`, so it cannot vanish
  inside the corridor. Off-track states (already flagged as collisions)
  clamp the denominator to `≥ 0.1` so a crashed car cannot produce `Inf`.
- **Reference retained.** The Cartesian `KS_RHS`/`ST_RHS`, `KS_MODE`,
  `ST_MODE` and `simulate` stay unchanged as the reference model; the Frenet
  equations are validated against them, and the Frenet RHS composes the same
  body-dynamics terms.

## 3. Sensing and collision

- **Wall collision.** Sample 8 points on the car footprint (4 corners plus 4
  edge midpoints): `to_cartesian` gives the world pose, `car_vertices` the
  corners in the world frame, and `to_frenet` (hinted with the car's `s`)
  maps each sample back to `(s_i, d_i)`. Flag collision if any
  `d_i > w_left(s_i)` or `d_i < −w_right(s_i)`. Edge midpoints catch inner walls bulging into an
  edge before a corner, up to `L²/(8R)` (about 2 cm at R = 2 m).
- **Car-car collision.** Unchanged SAT in the world frame; poses come from
  `to_cartesian`. The O(1) Frenet quantity is the `s` gap, used for
  gap analysis and as a cheap pre-filter (skip SAT when gap and `|Δd|` are
  clearly larger than the footprint diagonal).
- **LiDAR.** Per beam: convert the ray origin to world pose, then sphere
  trace. At each sample `p`, project to `(s, d)` (warm-started from the
  previous sample), compute the margin `m = w(s) − |d|`, and advance by
  `max(m / Λ, step_min)`. `Λ = 1 + max|w′| / (1 − κ_max·w_max)` is the
  Lipschitz bound of `m`, computed at load, so the step cannot jump a wall.
  When `m < 0` the boundary is bracketed; bisect to `1e-4 m`. `step_min`
  is 0.1 m, which bounds the near-parallel-ray worst case. Opponent cars are
  still intersected as segments, and `noise_std` behavior is unchanged.
- **API.** `step_multi_agent!` and `AsyncSim` take `track::Track` in place of
  `bvh`. `scan`, `cast_ray`, `check_track_collision` follow.
- **Rendering.** Renderers draw in the world frame: `to_cartesian` per state
  and `track_boundaries` for outlines. `TelemetrySink` records raw Frenet
  states, so the meaning of the `state` column changes (documented in the
  handoff doc).

## 4. Reachability (Frenet)

- **State.** 10-D augmented state `(s, d, δ, v, μ, ψ̇, β, a, δ̇, κ)`. `κ` is a
  zero-derivative dimension, like `a`/`δ̇`.
- **ODEs.** Hand-written `@taylorize` mirrors (KS and ST) in Frenet form,
  same reason as today: `@taylorize` needs source code. They divide by
  `1 − κd` and use `du[10] = 0`.
- **κ window.** `reachability_analyze` computes
  `[κ_min, κ_max]` of the tck over `[s_lo, s_hi + v⁺·T]` from the initial set
  and `tspan`, and intersects the `κ` dimension of the initial box with it.
  Docs tell users to shorten `tspan` (run per window) on winding stretches,
  since the interval widens.
- **Safety.** `check_flowpipe_safety(sol, track)` boxes each reach set in
  `(s, d)` and requires `|d| ≤ w_min` over the box's `s`-range, where `w_min`
  is the tck's minimum of `w` over that range (conservative: with
  `w_left = w_right` the bound is symmetric). `to_lazysets(track)` is kept,
  built from `track_boundaries`, for callers who want the world-frame
  corridor.
- **Validation.** The Frenet mirrors are cross-checked against the Frenet RHS
  at sample points (as the Cartesian mirrors were against `KS_RHS`/`ST_RHS`).

## Error handling

- Load-time: malformed CSV (column mismatch, `len(t) ≠ n + k + 1`, non-finite
  values), non-closing curve, `w ≤ 0`, `|κ|w ≥ 0.9` all raise `ArgumentError`
  with the offending value.
- A track whose sections come closer than a corridor width logs a warning at
  load (see Risks); it is not an error.
- `to_frenet` fails to converge (default 20 Newton iterations) → falls back
  to the table scan; if that also fails, throws.
- Renderers/`AsyncSim` misuse keeps the existing error contracts.

## Testing

- tck evaluation vs `scipy.interpolate.splev` on a committed fixture CSV
  (value and first derivative, to 1e-9).
- Analytic circle fixture (`R = 5`): `L = 2πR`, `κ = 1/R`, `to_frenet` /
  `to_cartesian` round trip to 1e-9, `s` monotone.
- Arc-length table: `|dP/ds| = |P′(u(s))|·(du/ds) = 1` to 1e-6 at random `s`.
- Frenet vs Cartesian equivalence: integrate the same action sequence in both
  frames, compare world pose after conversion (tolerance ~1e-6 over 1 s).
- Wall collision: agrees with brute-force polyline distance on random poses;
  edge-midpoint case near a tight inner wall.
- LiDAR: sphere tracing vs brute-force ray/segment on a 20 000-point
  boundary sample, all beams within 1e-3 m; no-tunneling sweep at hairpins.
- Reachability: mirrors vs Frenet RHS; a flowpipe on a straight is safe and
  one aimed at the wall is flagged.
- Existing async/render/telemetry tests updated to the new signatures.
- **Benchmark gate (first task after the map layer):** time a 1080-beam scan
  on the Austin tck. Threshold: ≤ 5 ms single-thread. If exceeded, first
  thread over beams, then fall back to the derived-polyline raycaster
  (Decision 2's rejected option) for LiDAR only.

## Demo

`main.jl` is rewritten: load a tck CSV, run pure pursuit in `(s, d)` (target
`(s + L_ahead, 0)`, curvature-limited speed from `κ(s)`, lap counting via
`⌊s/L⌋`). Requires the Austin CSV, which the project owner pre-converts;
until then a Julia-side test fixture generator exercises the same path.

## Risks

- **κ enclosure is not rigorous.** Treating `κ(s(t))` as a constant interval
  does not formally contain all time-varying trajectories, the same caveat
  as the existing `a`/`δ̇` treatment. Docs say so explicitly.
- **Sphere-tracing cost** unmeasured; guarded by the benchmark gate.
- **Fold-over tracks** (sections closer than a corridor width) can send
  `to_frenet` to the wrong branch off-track; validation checks minimum
  self-distance at load and warns.
- **CSV format** is defined here, not yet by `tck.py`; must be confirmed
  against the owner's converter before implementation.

## Implementation-time amendments

Found while prototyping and verifying against real data; the plan
(`docs/superpowers/plans/2026-09-21-frenet-tck.md`) carries the details.

1. CSV padding: periodic `splprep` returns `len(c) = len(t) − k − 1`; the
   writer pads to `len(t)`, the loader ignores the padding.
2. The Cartesian `step_dynamics` is renamed `step_dynamics_cartesian`; the
   Frenet step takes the public name `step_dynamics(𝐱, action, params, track, dt)`.
3. The car-car Frenet pre-filter is dropped (not provably safe on curves).
4. `to_frenet` never throws (table-scan retry, then last iterate).
5. Validation is pointwise `max |κ|·w < 0.9` with a located error message.
6. No knot-span hint; binary search meets the LiDAR gate (3.3 ms measured).
7. The load-time fold-over warning (sections closer than a corridor width)
   is not implemented; listed as a deferred minor.
8. Austin, as converted by the current `tck.py`, fails validation
   (`|κ|·w` up to 1.42 over ~1% of the lap). Strict rejection was chosen; the
   map must be re-fit.
