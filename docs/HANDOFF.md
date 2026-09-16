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
- **Phase 2 — Geometry, Raycasting & Collision Engine.** `LazySets.jl` map
  parsing, a BVH/R-tree spatial index, a fast 2D LiDAR raycaster and
  collision detector. **Not started.**
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

Phase 1's documents are the reference example of the expected depth:
- `docs/superpowers/specs/2026-09-16-phase1-core-dynamics-design.md`
- `docs/superpowers/plans/2026-09-16-phase1-core-dynamics.md`

subagent-driven-development keeps a ledger at
`.superpowers/sdd/<plan-basename>/progress.md` inside whatever worktree
is executing the plan (git-ignored, deleted after a clean final review —
so Phase 1's ledger no longer exists; its rulings are summarized below
since they matter beyond that one execution).

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

## Where to go next

Phase 2 (Geometry, Raycasting & Collision Engine) hasn't been brainstormed
yet. Start there with `superpowers:brainstorming` — it'll need its own
design conversation (map format decision, spatial-index library choice
now that `LazySets.jl` is only a rough suggestion from the original
blueprint, how collision detection integrates with Phase 1's `State`).
