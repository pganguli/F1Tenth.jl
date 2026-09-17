# Phase 5 — Reachability Analysis & Telemetry Export: Design

## Context

Phase 1 (`src/dynamics/`) provides `KS_MODE`/`ST_MODE` — two
`ModelingToolkit` `System`s sharing one 7-dimensional state space
`(x, y, δ, v, ψ, ψ̇, β)` — and `VehicleParams`. Phase 2
(`src/geometry/`) provides `Track`/`to_lazysets(track) ->
Vector{LazySets.VPolygon}`, one convex quad per centerline slice.
Phase 4 (`src/async/`) provides `AsyncSim`, whose physics tier fans out
each tick's `(states, scans, collisions)` frame to every attached
`RendererEntry` via `put_latest!`/`put!`.

Phase 5 adds two independent capabilities on top of this:

1. **Reachability analysis** — an offline safety-verification pipeline
   that builds a two-mode hybrid automaton from `KS_MODE`/`ST_MODE`,
   computes a reachable flowpipe from an uncertain initial state and
   bounded control inputs using `ReachabilityAnalysis.jl`, and checks
   the flowpipe against the track's drivable corridor
   (`to_lazysets`-derived polygons).
2. **Telemetry export** — an `Arrow.jl`-backed sink that attaches to a
   live `AsyncSim` exactly like any other renderer and logs every
   tick's full `(states, scans, collisions)` frame to disk.

Both ride on prior-phase output types unmodified; neither requires
touching `src/dynamics/`, `src/geometry/`, `src/render/`, or
`src/async/sim.jl`.

## Non-goals

- No changes to `KS_MODE`/`ST_MODE`, `dynamics_rhs`, or any other
  Phase 1 file. Per Phase 1's load-bearing fact 1, `KS_MODE`/`ST_MODE`
  (not `dynamics_rhs`) are the hybrid-system building blocks used here.
- No online/runtime reachability monitoring wired into `AsyncSim`'s
  tick loop. Reachability analysis in this phase is an offline,
  standalone pipeline a caller runs separately, not a live safety gate.
- No changes to `sim.jl`'s tier machinery. Telemetry attaches as an
  ordinary `AbstractRenderer`, reusing `RendererEntry`/the existing
  renderer tier unchanged.
- No `HybridSystems.jl` automaton reuse across separate
  `reachability_analyze` calls — each call builds/consumes its own
  problem; no caching or incremental re-analysis.
- No numeric-fidelity claim to `../f1tenth_gym` — it has no
  reachability-analysis or telemetry-export equivalent to port from;
  this is new functionality, not a port.
- No compressed/rotated telemetry files, no streaming-to-remote-store
  sink. One local Arrow file per `TelemetrySink` instance.

## Decisions

### 1. `build_hybrid_system`: two-location automaton over `KS_MODE`/`ST_MODE`

```julia
build_hybrid_system(params::VehicleParams) -> HybridSystems.HybridSystem
```

Two discrete locations, `:ks` and `:st`, wrapping `KS_MODE` and
`ST_MODE` respectively. Invariants: `:ks` active while `v < 0.5`, `:st`
active while `v >= 0.5` (matching `dynamics_rhs`'s own `abs(v) < 0.5`
threshold, applied here as a one-sided `v < 0.5` since `HybridSystems`
guards partition on the crossed state variable, not its absolute
value — a symmetric two-guard pair, `v < 0.5` for `:ks` and `v >= 0.5`
for `:st`, is used instead of a single `abs(v)` guard). Both directions
of the `v = 0.5` transition use an identity reset map (no state
component changes value at the switch), since `:ks`/`:st` already
share one 7-dimensional space by construction.

**Control-input uncertainty is modeled by state augmentation, not a
separate `input_set` at the `HybridSystem` level.** `TMJets`-style
Taylor-model reachability, as used by `ReachabilityAnalysis.jl`, takes
one initial box and integrates it forward; there is no separate,
independently-varying disturbance/control channel it re-samples at
each step. The standard technique (used throughout this project's
reachability literature, not invented here) is to augment the state
with the uncertain inputs as extra dimensions with zero dynamics: each
mode's `@taylorize`d function operates on a 9-dimensional augmented
state `(x, y, δ, v, ψ, ψ̇, β, a, δ̇)` with `D(a) = 0`, `D(δ̇) = 0` added
to whichever 7 equations it already mirrors from `_ks_eqs`/`_st_eqs`
(which reference `a`/`δ̇` as parameters — now the last two components
of this same augmented state vector instead). This is a real,
accepted conservatism: it holds `a`/`δ̇` bounded within their initial
box for the *entire* analysis horizon rather than modeling them as
independently time-varying within that box at every instant — noted
in Open Questions.

**Amended before implementation began** (see addendum at the end of
this section for why). Originally this decision's primary mechanism
was a native `HybridSystems.jl` automaton built directly from
`KS_MODE`/`ST_MODE`'s MTK-generated dynamics (Approach A from
brainstorming), falling back to a hand-rolled per-mode driver
(Approach C) only if that proved unworkable during implementation.
Investigating the actual installed `ReachabilityAnalysis.jl`/
`HybridSystems.jl` APIs (reading their source directly, not guessing)
before writing the plan found Approach A's literal mechanism
**cannot** work at all, for two independent reasons, so the fallback
is adopted as the plan's starting point rather than as a
contingency to discover mid-implementation:

1. `ReachabilityAnalysis.jl`'s `ModelingToolkit` integration is dead
   code — `src/Initialization/init.jl`'s `__init__` has
   `#@require ModelingToolkit = ... include("init_ModelingToolkit.jl")`
   commented out, and neither `[weakdeps]` nor `[extensions]` exist in
   its `Project.toml`. There is no code path that accepts an MTK
   `System` (`KS_MODE`/`ST_MODE`) directly.
2. Nonlinear modes in `ReachabilityAnalysis.jl`'s hybrid-system API
   (confirmed against its own test suite, e.g.
   `test/models/hybrid/lotka_volterra.jl`) are plain Julia functions
   annotated with the `@taylorize` macro:
   `@taylorize function f!(du, u, p, t) ... end`, then
   `@system(x' = f!(x), dim:N, x ∈ invariant)`. `@taylorize` is a
   macro — it transforms a function's *source code* at macroexpansion
   time. `KS_RHS`/`ST_RHS` (Phase 1's `mtk_models.jl`) are
   `@eval`'d into existence at runtime, inside `__mtk_models_init__()`,
   from a `build_function`-generated `Expr` — there is no source form
   for `@taylorize` to macroexpand, at any point before or after
   `F1Tenth` loads. This isn't a version-specific quirk to retry after
   an upgrade; it's the fundamental mismatch between a compile-time
   macro and a runtime-`@eval`'d closure.

**Mechanism, as amended:** `build_hybrid_system` still produces a
genuine `HybridSystems.HybridSystem` (a `GraphAutomaton` with 2 states,
`ConstrainedIdentityMap` reset transitions in both directions,
`AutonomousSwitching()`) — matching this decision's automaton
structure — but each mode's continuous dynamics is a hand-written,
`@taylorize`d Julia function that mirrors `KS_MODE`'s/`ST_MODE`'s
equations line-for-line (transcribed from `src/dynamics/mtk_models.jl`'s
`_ks_eqs`/`_st_eqs`), rather than reusing `KS_MODE`/`ST_MODE`/`KS_RHS`/
`ST_RHS` as a value. `reachability_analyze` (Decision 2) then calls
`ReachabilityAnalysis.solve` once on the whole 2-mode `HybridSystem` —
the *automaton-level* integration originally intended by Approach A,
not Approach C's per-mode-solve-plus-manual-switch-detection driver.
Keeping both equation sets in sync by hand (rather than by
construction) is a real, accepted cost of this amendment: a future
change to `_ks_eqs`/`_st_eqs` in `mtk_models.jl` must be manually
mirrored into the reachability module's `@taylorize`d functions, with
nothing enforcing that they stay identical beyond a comment pointing
each definition at the other and a test that checks their outputs
agree at sample points (Decision 1's implementation task includes this
cross-check test). Flagged in Open Questions as a maintenance burden a
future phase could remove if `ModelingToolkit`/`Symbolics` ever gains a
supported path to emit `@taylorize`-compatible source, or if
`ReachabilityAnalysis.jl` starts accepting un-`@taylorize`d functions
for `TMJets` without a large performance cliff.

### 2. `reachability_analyze`: flowpipe computation over `Zonotope` bounds

```julia
reachability_analyze(hs, initial_set::LazySets.Zonotope,
                      input_set::LazySets.Zonotope, tspan::Tuple{Float64,Float64};
                      alg=ReachabilityAnalysis.TMJets()) -> ReachabilityAnalysis.Flowpipe
```

`initial_set` bounds the uncertain starting state (7-dimensional,
matching `State`); `input_set` bounds uncertain control input over the
analysis horizon (2-dimensional, matching `Action`). Internally,
`reachability_analyze` takes the Cartesian product of the two into one
9-dimensional `Zonotope` (via `initial_set × input_set`, `LazySets`'
Cartesian-product operator) before constructing the `InitialValueProblem`,
per Decision 1's state-augmentation technique — this 9-D combination is
never exposed to the caller, who only ever supplies/reasons about the
7-D state and 2-D input sets separately. `Zonotope`s are used (not
hyperrectangles) for tighter, correlation-aware over-approximation, per
the project owner's explicit choice — accepted even though this makes
caller-side construction (specifying a center + generator matrix rather
than plain per-dimension `[lo, hi]` bounds) less immediately intuitive
than a hyperrectangle would have been. `tspan` is the analysis time
horizon in seconds. Returns `ReachabilityAnalysis.jl`'s own `Flowpipe`
type, whose reach-sets are 9-dimensional (state + the two augmented
input dimensions) — Decision 3 projects onto just the `(x, y)`
components it needs and ignores the rest.

### 3. `check_flowpipe_safety`: flowpipe vs. track-corridor containment

```julia
check_flowpipe_safety(flowpipe, track::Track) -> (safe::Bool, violating_indices::Vector{Int})
```

For each reach-set in the flowpipe, projects it onto the `(x, y)`
subspace and checks containment within the union of `to_lazysets(track)`
corridor polygons. A reach-set index is recorded as violating if its
`(x, y)` projection is not fully contained in that union — i.e. any
part of the reachable set at that time step could be outside the
drivable corridor. `safe` is `true` iff `violating_indices` is empty.

Inherits Phase 2's load-bearing fact 5 caveat: `to_lazysets`'s
`VPolygon` construction silently reshapes self-intersecting quads
(possible at sharp track curvature) rather than erroring, so a
corridor polygon at a sharp turn could be a smaller/differently-shaped
region than the true corridor there — `check_flowpipe_safety` inherits
this imprecision unchanged, it does not re-validate `to_lazysets`'s
output.

### 4. `TelemetrySink`: an `AbstractRenderer`, not a new tier

```julia
mutable struct TelemetrySink <: AbstractRenderer
    path::String
    writer::Union{Nothing,Arrow.Writer}
    tick_index::Int
    episode_index::Int
    initialized::Bool
end

TelemetrySink(path::String) -> TelemetrySink
```

Implements the four existing `AbstractRenderer` methods:

- `init!(sink, track)` — opens `sink.path` for writing, initializes
  the Arrow writer with the fixed schema (below), sets
  `initialized = true`. Track geometry itself is not logged (a
  telemetry consumer is assumed to already have the track that
  produced a given file; only per-tick simulation output is logged).
- `render!(sink, states, scans, collisions)` — appends one row per
  agent for this tick: `episode_index`, `tick_index`, `agent_index`,
  `state::Vector{Float64}` (length 7), `scan::Vector{Float64}` (full
  beam range array, per the project owner's explicit choice to log
  complete scans rather than omit them for size), `collided::Bool`.
  Increments `tick_index` after appending. Throws `ErrorException` if
  called before `init!`/`reset!`, matching `GLMakieRendererImpl`'s
  existing lifecycle contract (Decision 1 of the Phase 3 spec) — unlike
  `NullRenderer`, a `TelemetrySink` is not a safe-by-default no-op,
  since silently dropping ticks would make the logged file incomplete
  in a way a caller could not detect from the file alone.
- `close!(sink)` — flushes and closes the Arrow writer, sets
  `sink.writer = nothing`, `initialized = false`. Safe to call twice
  (a `nothing` writer makes the second call a no-op), matching
  `GLMakieRendererImpl.close!`'s existing idempotence contract.
- `reset!(sink, track)` — does **not** rotate to a new file or reset
  `tick_index` to zero. Instead increments `episode_index` and resets
  `tick_index` to `0` for the new episode, so multiple episodes append
  to one continuous file distinguished by `episode_index` — chosen
  over per-episode file rotation because `AsyncSim.reset!` already
  calls every attached renderer's `reset!` with no way for a caller to
  supply a new path per episode, and a single file with an episode
  column is simpler for a downstream reader to load than a directory
  of per-episode files.

Attached exactly like any other renderer:
`RendererEntry(TelemetrySink(path), DropLatest())` (or `Blocking(...)`
for lossless capture) passed into `AsyncSim`'s `renderers` — **zero
changes to `sim.jl`**. `TelemetrySink` participates in `AsyncSim`'s
existing fail-fast tier contract automatically: an Arrow write error
inside `render!` propagates out of `_renderer_loop` exactly like a
`GLMakieRenderer` draw error, triggering `_record_error!`/
`_close_all_channels!` with no special-casing needed.

**Schema** (one row per `(episode_index, tick_index, agent_index)`):

| column | type | notes |
|---|---|---|
| `episode_index` | `Int` | bumped by each `reset!` |
| `tick_index` | `Int` | bumped by each `render!`, reset to 0 by `reset!` |
| `agent_index` | `Int` | 1-based, matches `states`/`scans`/`collisions` index |
| `state` | `Vector{Float64}` (length 7) | `(x, y, δ, v, ψ, ψ̇, β)` |
| `scan` | `Vector{Float64}` (length `cfg.num_beams`) | full LiDAR range array |
| `collided` | `Bool` | |

## Data flow

**Reachability** (fully offline, no `AsyncSim`):

```julia
params = VehicleParams()
hs = build_hybrid_system(params)
initial_set = Zonotope(center, generators)   # 7-D
input_set = Zonotope(u_center, u_generators)  # 2-D
flowpipe = reachability_analyze(hs, initial_set, input_set, (0.0, 2.0))
safe, violating = check_flowpipe_safety(flowpipe, track)
```

**Telemetry** (attached to a live `AsyncSim`):

```julia
sink = TelemetrySink("run1.arrow")
sim = AsyncSim(bvh, params, car, dt, cfg, initial_states, initial_actions;
               renderers=[RendererEntry(sink, DropLatest())])
start!(sim)
# ... push actions, let it run ...
stop!(sim)  # flushes sink via close! semantics reached through the tier shutdown path
```

## File layout

```
src/reachability/
  hybrid_system.jl   # build_hybrid_system
  analyze.jl         # reachability_analyze
  safety_check.jl    # check_flowpipe_safety
src/telemetry/
  sink.jl            # TelemetrySink + AbstractRenderer methods
test/reachability/
  hybrid_system_test.jl
  analyze_test.jl
  safety_check_test.jl
test/telemetry/
  sink_test.jl
```

`src/F1Tenth.jl` gains four new `include`s and exports
`build_hybrid_system, reachability_analyze, check_flowpipe_safety,
TelemetrySink`. `Project.toml` gains `ReachabilityAnalysis` and `Arrow`
as regular `[deps]` + `[compat]` entries — `HybridSystems.jl`'s
automaton API (`GraphAutomaton`, `ConstrainedIdentityMap`,
`AutonomousSwitching`, `HybridSystem`, `add_transition!`) and
`LazySets.Zonotope` are both re-exported by `ReachabilityAnalysis`
(confirmed directly against the installed package), so no separate
`HybridSystems` dependency entry is needed; `LazySets` is already a
Phase 2 dependency. `test/exports_test.jl` is extended to cover the
four new exported names.

## Open questions / deferred items for a future phase

- Whether `TelemetrySink`'s `close!` must be explicitly reachable
  through `AsyncSim.stop!`'s existing tear-down path, or whether a
  caller must separately call `close!` on the sink after `stop!` —
  `stop!` closes the renderer's *channel*, which makes
  `_renderer_loop` exit via `InvalidStateException`, but does not
  itself call the renderer's `close!`. This matches existing behavior
  for `GLMakieRenderer`/`CairoMakieRenderer` (a caller is already
  expected to call `close!` on those explicitly), so `TelemetrySink`
  follows the same existing contract rather than a new one — flagged
  here only because an unflushed Arrow writer is a more surprising
  failure mode (silent data loss on process exit) than an unclosed
  GLMakie window.
- Decision 1's hand-mirrored `@taylorize`d equations (duplicated from
  `_ks_eqs`/`_st_eqs` rather than derived from them) are a genuine,
  ongoing maintenance burden — see that decision's amendment for the
  accepted cost and the cross-check test that guards against silent
  divergence.
- Decision 2's control-input state-augmentation technique holds
  `a`/`δ̇` within their initial box for the whole analysis horizon,
  rather than modeling them as independently time-varying within that
  box at every instant — a real conservatism (over-approximation, not
  under-approximation, so still sound for safety verification) worth
  knowing about if a future caller is surprised by how wide the
  resulting flowpipe is for a long `tspan`.
- No online/runtime reachability monitor — noted in Non-goals; a
  future phase could wire `check_flowpipe_safety` (or a
  faster-to-recompute approximation of it) into `AsyncSim`'s physics
  tier as an early-warning tier, but that's out of scope here.
