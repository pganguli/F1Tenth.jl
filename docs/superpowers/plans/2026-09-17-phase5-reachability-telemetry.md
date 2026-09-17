# Phase 5: Reachability Analysis & Telemetry Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline reachability-analysis pipeline
(`build_hybrid_system`/`reachability_analyze`/`check_flowpipe_safety`)
that verifies a `Track`'s drivable corridor stays safe under bounded
initial-state/control-input uncertainty, and an `Arrow.jl`-backed
`TelemetrySink` that attaches to a live `AsyncSim` exactly like any
other renderer and logs every tick's full `(states, scans, collisions)`
frame to disk.

**Architecture:** `src/reachability/hybrid_system.jl` builds a
two-location `ReachabilityAnalysis.HybridSystem` (`:ks`/`:st`, guards
at `v = 0.5`, identity resets) whose per-mode continuous dynamics are
hand-written `@taylorize`d functions mirroring `KS_MODE`'s/`ST_MODE`'s
equations (see amendment below for why this mirrors rather than reuses
Phase 1's MTK objects). `src/reachability/analyze.jl` wraps
`ReachabilityAnalysis.solve` over a 9-D augmented state (7 vehicle
states + 2 control inputs modeled as extra zero-derivative
dimensions). `src/reachability/safety_check.jl` checks each reach-set's
`(x, y)` projection against `to_lazysets(track)`'s corridor polygons.
`src/telemetry/sink.jl` adds `TelemetrySink <: AbstractRenderer`,
reusing `AsyncSim`'s existing renderer tier unchanged.

> **Amendment (resolved before implementation began, not discovered
> mid-task):** Brainstorming's original plan was "Approach A" — a
> native `HybridSystems.jl` automaton built directly from `KS_MODE`/
> `ST_MODE`'s MTK-generated dynamics, falling back to "Approach C" (a
> hand-rolled per-mode driver) only if that proved unworkable during
> implementation. Investigating the actual installed
> `ReachabilityAnalysis.jl`/`HybridSystems.jl` source (not guessing)
> before writing this plan found Approach A's literal mechanism cannot
> work at all: (1) `ReachabilityAnalysis.jl`'s `ModelingToolkit`
> integration is dead code (a commented-out `@require` in
> `Initialization/init.jl`, no `[weakdeps]`/`[extensions]` in its
> `Project.toml`); (2) its nonlinear-mode support requires a
> `@taylorize`-annotated **source-level** function, and `@taylorize` is
> a macro that cannot operate on `KS_RHS`/`ST_RHS` — Phase 1's
> runtime-`@eval`'d closures, with no source form to macroexpand. See
> the spec's amended Decision 1 for the full rationale. This plan
> builds the *automaton* the way Approach A intended (one call to
> `ReachabilityAnalysis.solve` over the whole 2-mode `HybridSystem`,
> not Approach C's per-mode-solve-plus-manual-switch-detection driver)
> but with hand-mirrored `@taylorize`d dynamics instead of reused MTK
> objects — a middle ground between the two brainstormed approaches,
> confirmed to actually work via a standalone smoke test before this
> plan was written.

**Tech Stack:** Julia 1.13 (repo compat floor 1.10.10),
ReachabilityAnalysis.jl (new, regular dep, re-exports the
`HybridSystems.jl`/`MathematicalSystems.jl` automaton API and
`LazySets.Zonotope` used here), Arrow.jl (new, regular dep), LazySets
(existing, Phase 2 dep), StaticArrays (existing).

**Spec:** `docs/superpowers/specs/2026-09-17-phase5-reachability-telemetry-design.md`

## Global Constraints

- No file under `src/dynamics/`, `src/geometry/`, `src/render/`, or
  `src/async/sim.jl` may be modified. All Phase 5 code lives under
  `src/reachability/` and `src/telemetry/`.
- The augmented state order for both hybrid-system modes is
  `(x, y, δ, v, ψ, ψ̇, β, a, δ̇)` — 9 dimensions, indices 1-9 — with `a`
  (index 8) and `δ̇` (index 9) added as extra zero-derivative
  dimensions per the spec's Decision 1/2 state-augmentation technique
  (`TMJets` has no separate time-varying-input channel).
- Location `1` is always `:ks` (`v <= 0.5` invariant), location `2` is
  always `:st` (`v >= 0.5` invariant), in every `GraphAutomaton`/
  `HybridSystem` built by `build_hybrid_system`.
- `TelemetrySink` implements `AbstractRenderer`'s four methods
  (`init!`/`render!`/`close!`/`reset!`) with the exact same
  positional-argument shape as every other renderer (spec Decision 4;
  Phase 3 spec Decision 1) — no changes to `src/async/sim.jl`'s tier
  machinery.
- `render!` on a `TelemetrySink` before `init!`/`reset!` must throw a
  clear `ErrorException`, matching `GLMakieRendererImpl`'s existing
  lifecycle contract. `close!` must be safe to call twice.
- ReachabilityAnalysis UUID: `1e97bd63-91d1-579d-8e8d-501d2b57c93f`,
  resolved version `0.32.1`. Arrow UUID:
  `69666777-d1a9-59fb-9406-91d4454c9d45`, resolved version `1.11.0`.
  (Looked up directly from the local Julia General registry and a
  scratch `Pkg.add` resolution, same as Phase 3's plan did for
  GLMakie/CairoMakie — use these exact values.)

---

### Task 1: `build_hybrid_system` — two-mode automaton with hand-mirrored dynamics

**Files:**
- Modify: `Project.toml` (add `ReachabilityAnalysis` under `[deps]`/`[compat]`)
- Create: `src/reachability/hybrid_system.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/reachability/hybrid_system_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `VehicleParams` (Phase 1, fields `μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m,
  Iᶻ`); `KS_RHS`/`ST_RHS` (Phase 1, only for this task's own
  cross-check test, not for production code — see amendment above for
  why they can't be reused directly).
- Produces: `build_hybrid_system(params::VehicleParams) ->
  ReachabilityAnalysis.HybridSystem`, a 2-location automaton (location
  `1` = `:ks`, location `2` = `:st`) over the 9-D augmented state
  `(x, y, δ, v, ψ, ψ̇, β, a, δ̇)`.

- [ ] **Step 1: Edit `Project.toml`**

Replace the full contents of `Project.toml` with:

```toml
name = "F1Tenth"
uuid = "ec9dc3f2-e932-4b18-a493-a4686694dcbe"
version = "1.0.0-DEV"
authors = ["Prateek Ganguli <prateek.ganguli@gmail.com> and contributors"]

[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReachabilityAnalysis = "1e97bd63-91d1-579d-8e8d-501d2b57c93f"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
CairoMakie = "0.15.6"
DelimitedFiles = "1.9.1"
DifferentialEquations = "8.1.1"
GLMakie = "0.13.6"
LazySets = "7.1.0"
ModelingToolkit = "11.43.1"
NearestNeighbors = "0.4.29"
ReachabilityAnalysis = "0.32.1"
StaticArrays = "1.9.20"
Symbolics = "7.39.2"
julia = "1.10.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.resolve(); Pkg.instantiate()'
```
Expected: `Manifest.toml` updates to include `ReachabilityAnalysis` and
its dependency tree (`HybridSystems`, `MathematicalSystems`,
`TaylorIntegration`, `TaylorModels`, `TaylorSeries`, etc. — this
downloads/precompiles a fair amount, expect a few minutes).

- [ ] **Step 2: Write the failing tests**

Create `test/reachability/hybrid_system_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using StaticArrays
using Test

@testset "build_hybrid_system: automaton structure" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    @test nmodes(hs) == 2
    @test ntransitions(hs) == 2
end

@testset "build_hybrid_system: :ks mode matches KS_RHS at a sample point" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    ks_mode = mode(hs, 1)
    f! = mapping(ks_mode)

    u = [0.0, 0.0, 0.1, 0.3, 0.2, 0.05, 0.01, 1.5, 0.2]  # v = 0.3 -> :ks regime
    du = zeros(9)
    f!(du, u, nothing, 0.0)

    p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
        params.h, params.m, params.Iᶻ, u[9], u[8])
    state = F1Tenth.State(u[1], u[2], u[3], u[4], u[5], u[6], u[7])
    ks_expected = F1Tenth.KS_RHS(state, p)

    @test du[1:5] ≈ collect(ks_expected[1:5]) atol=1e-10
    @test du[6] == 0.0
    @test du[7] == 0.0
    @test du[8] == 0.0
    @test du[9] == 0.0
end

@testset "build_hybrid_system: :st mode matches ST_RHS at a sample point" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    st_mode = mode(hs, 2)
    f! = mapping(st_mode)

    u = [0.0, 0.0, 0.1, 2.0, 0.2, 0.05, 0.01, 1.5, 0.2]  # v = 2.0 -> :st regime
    du = zeros(9)
    f!(du, u, nothing, 0.0)

    p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
        params.h, params.m, params.Iᶻ, u[9], u[8])
    state = F1Tenth.State(u[1], u[2], u[3], u[4], u[5], u[6], u[7])
    st_expected = F1Tenth.ST_RHS(state, p)

    @test du[1:7] ≈ collect(st_expected) atol=1e-10
    @test du[8] == 0.0
    @test du[9] == 0.0
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
    include("render/glmakie_test.jl")
    include("render/cairomakie_test.jl")
    include("async/latest_channel_test.jl")
    include("async/sim_test.jl")
    include("reachability/hybrid_system_test.jl")
    include("exports_test.jl")
end
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.build_hybrid_system` not defined.

- [ ] **Step 4: Implement `hybrid_system.jl`**

Create `src/reachability/hybrid_system.jl`:

```julia
using ReachabilityAnalysis

# Augmented state order shared by both modes: (x, y, δ, v, ψ, ψ̇, β, a, δ̇).
# `a`/`δ̇` (Phase 1's control inputs) are appended as extra
# zero-derivative dimensions -- TMJets has no separate time-varying
# input channel, so bounding them via the initial set is the standard
# state-augmentation technique for this kind of Taylor-model
# reachability (spec Decision 1/2). This means `a`/`δ̇` stay within
# their initial box for the whole analysis horizon rather than varying
# independently within it at every instant -- an accepted
# over-approximation, not an under-approximation, so still sound.
const _AUGMENTED_DIM = 9

# These two functions are hand-written mirrors of `_ks_eqs`/`_st_eqs`
# in src/dynamics/mtk_models.jl -- NOT calls into KS_RHS/ST_RHS/
# KS_MODE/ST_MODE. See this file's F1Tenth.jl-level comment header (and
# the spec's amended Decision 1) for why: ReachabilityAnalysis.jl's
# `@taylorize` is a macro operating on function *source code*, and
# KS_RHS/ST_RHS are `@eval`'d into existence at runtime from a
# build_function-generated Expr -- there is no source form for
# `@taylorize` to macroexpand. Keeping these in sync with
# mtk_models.jl's equations by hand is a real, accepted maintenance
# cost; this task's own tests (above) cross-check both mirrors against
# KS_RHS/ST_RHS at sample points to catch silent divergence.
function _ks_taylor_expr(fname::Symbol, params::VehicleParams)
    lf = params.lᶠ
    lr = params.lʳ
    return quote
        @taylorize function $(fname)(du, u, p, t)
            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
            du[1] = v * cos(ψ)
            du[2] = v * sin(ψ)
            du[3] = δ̇
            du[4] = a
            du[5] = v / ($lf + $lr) * tan(δ)
            du[6] = zero(v)
            du[7] = zero(v)
            du[8] = zero(v)
            du[9] = zero(v)
            return du
        end
    end
end

function _st_taylor_expr(fname::Symbol, params::VehicleParams)
    μ = params.μ
    Cbf = params.Cᵦᶠ
    Cbr = params.Cᵦʳ
    lf = params.lᶠ
    lr = params.lʳ
    h = params.h
    m = params.m
    Iz = params.Iᶻ
    g = 9.81
    return quote
        @taylorize function $(fname)(du, u, p, t)
            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
            du[1] = v * cos(β + ψ)
            du[2] = v * sin(β + ψ)
            du[3] = δ̇
            du[4] = a
            du[5] = ψ̇
            du[6] = (-$μ * $m / (v * $Iz * ($lr + $lf)) *
                        ($lf^2 * $Cbf * ($g * $lr - a * $h) +
                         $lr^2 * $Cbr * ($g * $lf + a * $h)) * ψ̇ +
                     $μ * $m / ($Iz * ($lr + $lf)) *
                        ($lr * $Cbr * ($g * $lf + a * $h) -
                         $lf * $Cbf * ($g * $lr - a * $h)) * β +
                     $μ * $m / ($Iz * ($lr + $lf)) *
                        $lf * $Cbf * ($g * $lr - a * $h) * δ)
            du[7] = (($μ / (v^2 * ($lr + $lf)) *
                         ($Cbr * ($g * $lf + a * $h) * $lr -
                          $Cbf * ($g * $lr - a * $h) * $lf) - 1) * ψ̇ -
                     $μ / (v * ($lr + $lf)) *
                         ($Cbr * ($g * $lf + a * $h) +
                          $Cbf * ($g * $lr - a * $h)) * β +
                     $μ / (v * ($lr + $lf)) *
                         ($Cbf * ($g * $lr - a * $h)) * δ)
            du[8] = zero(v)
            du[9] = zero(v)
            return du
        end
    end
end

function _build_mode(fname::Symbol, taylor_expr::Expr, invariant, dim::Int)
    Core.eval(@__MODULE__, taylor_expr)
    sys_expr = :(@system(u' = $(fname)(u), dim: $dim, u ∈ $invariant))
    return Core.eval(@__MODULE__, sys_expr)
end

function build_hybrid_system(params::VehicleParams)
    ks_name = gensym(:phase5_ks)
    st_name = gensym(:phase5_st)

    # v is index 4 of the 9-D augmented state.
    v_ks_invariant = HalfSpace([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5)   # v <= 0.5
    v_st_invariant = HalfSpace([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.5) # v >= 0.5

    ks_mode = _build_mode(ks_name, _ks_taylor_expr(ks_name, params), v_ks_invariant, _AUGMENTED_DIM)
    st_mode = _build_mode(st_name, _st_taylor_expr(st_name, params), v_st_invariant, _AUGMENTED_DIM)

    automaton = GraphAutomaton(2)
    add_transition!(automaton, 1, 2, 1)  # :ks -> :st, v crosses 0.5 upward
    add_transition!(automaton, 2, 1, 2)  # :st -> :ks, v crosses 0.5 downward

    reset_ks_to_st = ConstrainedIdentityMap(_AUGMENTED_DIM, v_st_invariant)
    reset_st_to_ks = ConstrainedIdentityMap(_AUGMENTED_DIM, v_ks_invariant)

    return HybridSystem(automaton, [ks_mode, st_mode],
        [reset_ks_to_st, reset_st_to_ks],
        [AutonomousSwitching(), AutonomousSwitching()])
end
```

Modify `src/F1Tenth.jl` to add the new `include` (after the async
includes, before any export changes — export additions happen in Task
5, matching every prior phase's plan pattern):

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
include("render/glmakie.jl")
include("render/cairomakie.jl")
include("async/latest_channel.jl")
include("async/sim.jl")
include("reachability/hybrid_system.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE
export Track, Segment, build_track, load_track
export TrackBVH, build_bvh
export LiDARConfig, CarFootprint, car_vertices, cast_ray, scan
export collides, check_car_collisions, check_track_collision
export step_multi_agent!
export to_lazysets
export AbstractRenderer, NullRenderer, GLMakieRenderer, CairoMakieRenderer
export init!, render!, close!, reset!
export LatestChannel, put_latest!, take_latest!, wait_latest!
export AsyncSim, start!, stop!
export RenderBackpressure, DropLatest, Blocking, RendererEntry

function __init__()
    __mtk_models_init__()
end

end
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the `:st` mode's cross-check test fails on a sign or
term, re-check the transcription against `mtk_models.jl:34-58`'s
`_st_eqs` character-by-character before changing anything else — this
is exactly the divergence risk the amendment above calls out.

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/reachability/hybrid_system.jl \
    test/reachability/hybrid_system_test.jl test/runtests.jl
git commit -m "Add build_hybrid_system: two-mode automaton over hand-mirrored KS/ST dynamics"
```

---

### Task 2: `reachability_analyze`

**Files:**
- Create: `src/reachability/analyze.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/reachability/analyze_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `build_hybrid_system` (Task 1); `ReachabilityAnalysis.Zonotope`,
  `HalfSpace`, `TMJets`, `InitialValueProblem`, `solve` (re-exported by
  `ReachabilityAnalysis`); `LazySets.×` (Cartesian product), `low`,
  `high`, `convert` (re-exported by `ReachabilityAnalysis`, already a
  Phase 2 transitive dependency via `LazySets`).
- Produces: `reachability_analyze(hs, initial_set::Zonotope,
  input_set::Zonotope, tspan::Tuple{Float64,Float64};
  alg=TMJets()) -> sol` where `sol` is `ReachabilityAnalysis.jl`'s
  solution object (`sol.F :: Vector{<:Flowpipe}`, one flowpipe segment
  per mode the trajectory occupies) — consumed by Task 3 via `sol.F`.

- [ ] **Step 1: Write the failing tests**

Create `test/reachability/analyze_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using LinearAlgebra
using Test

@testset "reachability_analyze: :st initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
    @test length(sol.F[1]) > 0
end

@testset "reachability_analyze: :ks initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], 0.001 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.001 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
end

@testset "reachability_analyze: initial_set straddling v=0.5 throws" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    initial_set = Zonotope([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 0.3 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    @test_throws ArgumentError F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1))
end
```

Modify `test/runtests.jl` (add after `hybrid_system_test.jl`):

```julia
    include("reachability/hybrid_system_test.jl")
    include("reachability/analyze_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.reachability_analyze` not defined.

- [ ] **Step 3: Implement `analyze.jl`**

Create `src/reachability/analyze.jl`:

```julia
using ReachabilityAnalysis

function reachability_analyze(hs, initial_set::Zonotope, input_set::Zonotope,
    tspan::Tuple{Float64,Float64}; alg=TMJets())
    v_lo = low(initial_set)[4]
    v_hi = high(initial_set)[4]
    if v_hi <= 0.5
        location = 1  # :ks
    elseif v_lo >= 0.5
        location = 2  # :st
    else
        throw(ArgumentError(
            "reachability_analyze: initial_set straddles the v = 0.5 KS/ST mode " *
            "boundary (v ∈ [$v_lo, $v_hi]); supply an initial_set fully within one mode"))
    end

    augmented = convert(Zonotope, initial_set × input_set)
    prob = InitialValueProblem(hs, [(location, augmented)])
    return solve(prob; tspan=tspan, alg=alg, max_jumps=4)
end
```

Modify `src/F1Tenth.jl` to add
`include("reachability/analyze.jl")` immediately after
`include("reachability/hybrid_system.jl")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If `solve` errors on numerical grounds (Taylor-model
divergence, step-size issues), first try shrinking `tspan`/increasing
`abstol` before suspecting the mode functions themselves — Task 1's
cross-check tests already confirm those match `KS_RHS`/`ST_RHS`.

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/reachability/analyze.jl test/reachability/analyze_test.jl test/runtests.jl
git commit -m "Add reachability_analyze: flowpipe computation over Zonotope bounds"
```

---

### Task 3: `check_flowpipe_safety`

**Files:**
- Create: `src/reachability/safety_check.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/reachability/safety_check_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `reachability_analyze`'s return value (`sol`, Task 2);
  `to_lazysets` (Phase 2, `Track -> Vector{LazySets.VPolygon}`);
  `LazySets.UnionSetArray`, `overapproximate`, `Hyperrectangle`,
  `issubset`, `set` (re-exported by `ReachabilityAnalysis`).
- Produces: `check_flowpipe_safety(sol, track::Track) -> (safe::Bool,
  violating_indices::Vector{Int})`.

- [ ] **Step 1: Write the failing tests**

Create `test/reachability/safety_check_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using StaticArrays
using LinearAlgebra
using Test

function _straight_test_track()
    n = 20
    centerline = [SVector(Float64(i), 0.0) for i in 0:(n - 1)]
    return F1Tenth.build_track(centerline, fill(1.5, n), fill(1.5, n))
end

@testset "check_flowpipe_safety: tight uncertainty inside a wide corridor is safe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = _straight_test_track()

    initial_set = Zonotope([5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test safe
    @test isempty(violating)
end

@testset "check_flowpipe_safety: huge uncertainty exceeding the corridor is unsafe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = _straight_test_track()

    initial_set = Zonotope([5.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        Diagonal([5.0, 5.0, 0.01, 0.01, 0.01, 0.01, 0.01]))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05);
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test !safe
    @test !isempty(violating)
end
```

Modify `test/runtests.jl` (add after `analyze_test.jl`):

```julia
    include("reachability/analyze_test.jl")
    include("reachability/safety_check_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.check_flowpipe_safety` not defined.

- [ ] **Step 3: Implement `safety_check.jl`**

Create `src/reachability/safety_check.jl`:

```julia
using ReachabilityAnalysis

function check_flowpipe_safety(sol, track::Track)
    corridor = UnionSetArray(to_lazysets(track))
    violating = Int[]
    idx = 0
    for fp in sol.F
        for rs in fp
            idx += 1
            box = overapproximate(set(rs), Hyperrectangle)
            xy_box = Hyperrectangle(low=low(box)[1:2], high=high(box)[1:2])
            issubset(xy_box, corridor) || push!(violating, idx)
        end
    end
    return (isempty(violating), violating)
end
```

Modify `src/F1Tenth.jl` to add
`include("reachability/safety_check.jl")` immediately after
`include("reachability/analyze.jl")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the "unsafe" test doesn't actually detect a
violation, increase that test's initial-set uncertainty further (e.g.
`Diagonal([20.0, 20.0, ...])`) rather than changing
`check_flowpipe_safety`'s logic — the corridor here is only 3 m wide
(`w_left = w_right = 1.5`).

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/reachability/safety_check.jl \
    test/reachability/safety_check_test.jl test/runtests.jl
git commit -m "Add check_flowpipe_safety: flowpipe vs. track-corridor containment"
```

---

### Task 4: `TelemetrySink`

**Files:**
- Modify: `Project.toml` (add `Arrow` under `[deps]`/`[compat]`)
- Create: `src/telemetry/sink.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/telemetry/sink_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `AbstractRenderer`, `init!`, `render!`, `close!`, `reset!`
  (Phase 3); `Track`, `State` (Phase 1/2); `Arrow.Writer`,
  `Arrow.write`, `Arrow.Table` (new dependency).
- Produces: `TelemetrySink(path::String) -> TelemetrySink`, a
  `mutable struct TelemetrySink <: AbstractRenderer` with fields
  `path::String`, `writer::Union{Nothing,Arrow.Writer}`,
  `tick_index::Int`, `episode_index::Int`, `initialized::Bool`, plus
  `init!`/`render!`/`close!`/`reset!` methods for it.

- [ ] **Step 1: Edit `Project.toml`**

Replace the full contents of `Project.toml` with:

```toml
name = "F1Tenth"
uuid = "ec9dc3f2-e932-4b18-a493-a4686694dcbe"
version = "1.0.0-DEV"
authors = ["Prateek Ganguli <prateek.ganguli@gmail.com> and contributors"]

[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReachabilityAnalysis = "1e97bd63-91d1-579d-8e8d-501d2b57c93f"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
Arrow = "1.11.0"
CairoMakie = "0.15.6"
DelimitedFiles = "1.9.1"
DifferentialEquations = "8.1.1"
GLMakie = "0.13.6"
LazySets = "7.1.0"
ModelingToolkit = "11.43.1"
NearestNeighbors = "0.4.29"
ReachabilityAnalysis = "0.32.1"
StaticArrays = "1.9.20"
Symbolics = "7.39.2"
julia = "1.10.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.resolve(); Pkg.instantiate()'
```
Expected: `Manifest.toml` updates to include `Arrow` and its dependency
tree.

- [ ] **Step 2: Write the failing tests**

Create `test/telemetry/sink_test.jl`:

```julia
using F1Tenth
using Arrow
using StaticArrays
using Test

function _small_test_track()
    n = 8
    R = 5.0
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
end

@testset "TelemetrySink: logs ticks across episodes and round-trips via Arrow" begin
    track = _small_test_track()
    mktempdir() do dir
        path = joinpath(dir, "telemetry.arrow")
        sink = F1Tenth.TelemetrySink(path)

        F1Tenth.init!(sink, track)
        states1 = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        scans1 = [[1.0, 2.0, 3.0]]
        F1Tenth.render!(sink, states1, scans1, [false])
        states2 = [F1Tenth.State(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(sink, states2, scans1, [true])

        F1Tenth.reset!(sink, track)
        states3 = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(sink, states3, scans1, [false])

        F1Tenth.close!(sink)

        tbl = Arrow.Table(path)
        @test length(tbl.tick_index) == 3
        @test collect(tbl.episode_index) == [0, 0, 1]
        @test collect(tbl.tick_index) == [0, 1, 0]
        @test collect(tbl.agent_index) == [1, 1, 1]
        @test collect(tbl.collided) == [false, true, false]
        @test Vector(tbl.state[2]) ≈ [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        @test Vector(tbl.scan[1]) ≈ [1.0, 2.0, 3.0]
    end
end

@testset "TelemetrySink: render! before init! throws" begin
    sink = F1Tenth.TelemetrySink(tempname())
    @test_throws ErrorException F1Tenth.render!(sink, F1Tenth.State[], Vector{Float64}[], Bool[])
end

@testset "TelemetrySink: close! is idempotent" begin
    track = _small_test_track()
    mktempdir() do dir
        sink = F1Tenth.TelemetrySink(joinpath(dir, "telemetry2.arrow"))
        F1Tenth.init!(sink, track)
        F1Tenth.close!(sink)
        F1Tenth.close!(sink)
        @test true
    end
end
```

Modify `test/runtests.jl` (add after `safety_check_test.jl`):

```julia
    include("reachability/safety_check_test.jl")
    include("telemetry/sink_test.jl")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.TelemetrySink` not defined.

- [ ] **Step 4: Implement `sink.jl`**

Create `src/telemetry/sink.jl`:

```julia
using Arrow

mutable struct TelemetrySink <: AbstractRenderer
    path::String
    writer::Union{Nothing,Arrow.Writer}
    tick_index::Int
    episode_index::Int
    initialized::Bool
end

TelemetrySink(path::String) = TelemetrySink(path, nothing, 0, 0, false)

function init!(sink::TelemetrySink, track::Track)
    sink.writer = open(Arrow.Writer, sink.path)
    sink.tick_index = 0
    sink.episode_index = 0
    sink.initialized = true
    return sink
end

function reset!(sink::TelemetrySink, track::Track)
    sink.initialized || error("TelemetrySink: reset! called before init!")
    sink.episode_index += 1
    sink.tick_index = 0
    return sink
end

function render!(sink::TelemetrySink, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    sink.initialized || error("TelemetrySink: render! called before init!/reset!")
    n = length(states)
    batch = (
        episode_index=fill(sink.episode_index, n),
        tick_index=fill(sink.tick_index, n),
        agent_index=collect(1:n),
        state=[Vector(s) for s in states],
        scan=scans,
        collided=collisions,
    )
    Arrow.write(sink.writer, batch)
    sink.tick_index += 1
    return nothing
end

function close!(sink::TelemetrySink)
    sink.writer === nothing && return nothing
    close(sink.writer)
    sink.writer = nothing
    sink.initialized = false
    return nothing
end
```

Modify `src/F1Tenth.jl` to add `include("telemetry/sink.jl")`
immediately after `include("reachability/safety_check.jl")`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/telemetry/sink.jl \
    test/telemetry/sink_test.jl test/runtests.jl
git commit -m "Add TelemetrySink: Arrow-backed telemetry export as an AbstractRenderer"
```

---

### Task 5: Module exports and full-suite verification

**Files:**
- Modify: `src/F1Tenth.jl`
- Modify: `test/exports_test.jl`

**Interfaces:**
- Consumes: everything produced in Tasks 1-4.
- Produces: public names exported from `F1Tenth` in addition to every
  prior phase's exports: `build_hybrid_system`, `reachability_analyze`,
  `check_flowpipe_safety`, `TelemetrySink`.

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
include("render/glmakie.jl")
include("render/cairomakie.jl")
include("async/latest_channel.jl")
include("async/sim.jl")
include("reachability/hybrid_system.jl")
include("reachability/analyze.jl")
include("reachability/safety_check.jl")
include("telemetry/sink.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE
export Track, Segment, build_track, load_track
export TrackBVH, build_bvh
export LiDARConfig, CarFootprint, car_vertices, cast_ray, scan
export collides, check_car_collisions, check_track_collision
export step_multi_agent!
export to_lazysets
export AbstractRenderer, NullRenderer, GLMakieRenderer, CairoMakieRenderer
export init!, render!, close!, reset!
export LatestChannel, put_latest!, take_latest!, wait_latest!
export AsyncSim, start!, stop!
export RenderBackpressure, DropLatest, Blocking, RendererEntry
export build_hybrid_system, reachability_analyze, check_flowpipe_safety
export TelemetrySink

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
        :LatestChannel, :put_latest!, :take_latest!, :wait_latest!,
        :AsyncSim, :start!, :stop!,
        :RenderBackpressure, :DropLatest, :Blocking, :RendererEntry,
        :build_hybrid_system, :reachability_analyze, :check_flowpipe_safety,
        :TelemetrySink,
    ]
    for name in exported_names
        @test isdefined(F1Tenth, name)
    end
end
```

- [ ] **Step 3: Run the full test suite**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS (every testset from Tasks 1-4, plus every Phase 1-4
testset still passing unchanged).

- [ ] **Step 4: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl test/exports_test.jl
git commit -m "Export Phase 5 reachability/telemetry API from F1Tenth module"
```
