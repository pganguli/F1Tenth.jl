# Phase 1: Core Dynamics & Symbolic System — Design Spec

## Purpose

Port the single-track vehicle dynamics from Python `f1tenth_gym`
(`gym/f110_gym/envs/dynamic_models.py`, `base_classes.py`) to Julia, as the
foundation for all later phases. No Python API compatibility is required or
attempted — this is a from-scratch Julia design that reproduces the same
physics and matches the same numeric ground truth.

Two consumers drive the design:
- A real-time, allocation-free fixed-step path (`step_dynamics`), for the
  interactive GLMakie loop in later phases.
- An accuracy-focused, adaptive-step path (`simulate`), for offline
  CairoMakie rendering / Arrow-logged trajectory analysis in later phases.

Both paths must use the *same* equations, generated once from
`ModelingToolkit.jl` (MTK) symbolic models — never two independently
hand-maintained copies of the physics.

## Reference behavior (ground truth)

The Python reference always carries a 7-element state
`[x, y, δ, v, ψ, ψ̇, β]` (x, y position; steering angle; velocity; yaw;
yaw rate; slip angle) and never reduces state dimensionality. The "kinematic
vs dynamic" switch documented in `vehicle_dynamics_st` is *internal*: when
`|v| < 0.5`, it computes the kinematic bicycle equations for the first 5
components and sets `ψ̇_dot = β_dot = 0`; otherwise it computes the full
dynamic single-track equations. `base_classes.py`'s RK4 integrator calls this
one switching function at every stage — it never calls a separate 5-state
function at the top level.

This means: one 7-dimensional state space, two flow modes on that same
space, switched per RHS evaluation by `|v| < 0.5`.

Control input is always `(steering_velocity, acceleration)` — the low-level
actuation signal the ODE actually consumes. The higher-level `pid()` speed/
steering-angle-to-actuation convenience function in Python is **not**
ported in Phase 1 (YAGNI; add later only if a specific planner needs it).

Physical constraints (`steering_constraint`, `accl_constraints`) clamp the
raw control input before it reaches the dynamics, exactly as Python does,
and are ported verbatim.

Default vehicle parameters (from `f110_env.py`), to become `VehicleParams`'
default constructor:

```
mu = 1.0489, C_Sf = 4.718, C_Sr = 5.4562, lf = 0.15875, lr = 0.17145,
h = 0.074, m = 3.74, I = 0.04712,
s_min = -0.4189, s_max = 0.4189, sv_min = -3.2, sv_max = 3.2,
v_switch = 7.319, a_max = 9.51, v_min = -5.0, v_max = 20.0
```

(`width`/`length` are vehicle geometry for Phase 2's collision model, not
Phase 1's `VehicleParams`.)

## Architecture

Two MTK `ODESystem`s share one 7-unknown state space:

- `ks_mode`: kinematic bicycle equations on `[x, y, δ, v, ψ]`, with
  `ψ̇_dot = 0` and `β_dot = 0` as explicit equations, so the system has all
  7 unknowns.
- `st_mode`: the full dynamic single-track equations on all 7 unknowns.

Both take the same 2 control inputs (steering velocity, acceleration) and
the same `VehicleParams` fields as MTK parameters. Each is built once via
`@mtkbuild`/`structural_simplify`, and a numerical RHS function is generated
from each via MTK's `ODEFunction` (not `build_function` by hand — use MTK's
own function-generation path so there is exactly one source of truth
between the symbolic model and the generated code). The generated
`ODEFunction`s are built once at module load (or on first use, memoized) and
reused — never rebuilt per call.

A thin Julia function `select_mode(state, params) -> (:ks, :st)` implements
the `|v| < 0.5` branch and picks which generated `ODEFunction` to evaluate.
This branching lives in Julia, not inside either MTK system, so that Phase 5
can later assemble `ks_mode`/`st_mode` directly into a `HybridSystems.jl`
`HybridSystem` (two modes sharing one state space, a guard at `|v| = 0.5`,
identity reset map) for `ReachabilityAnalysis.jl` — no state conversion or
resets needed at the transition since both modes already share the same
7-dim space.

## Components

All new code lives under `src/dynamics/`, included from `src/F1Tenth.jl`.

- **`src/dynamics/params.jl`**
  - `struct VehicleParams` — immutable, all `Float64` fields listed above,
    with a keyword constructor defaulting to the values above.
  - `steering_constraint(steering_angle, steering_velocity, params)` and
    `accl_constraints(vel, accl, params)` — ported directly from Python,
    taking `VehicleParams` instead of individual scalars.

- **`src/dynamics/mtk_models.jl`**
  - Builds `ks_mode` and `st_mode` as `ModelingToolkit.ODESystem`s using
    `@variables`/`@parameters`/`@mtkbuild`, following MTK's standard
    idiomatic construction (not manual `Equation` vectors where a macro
    covers it).
  - Exposes memoized numerical RHS closures (`ks_rhs`, `st_rhs`) built via
    MTK's `ODEFunction`, each callable as `f(state::SVector{7}, u::SVector{2},
    params::VehicleParams) -> SVector{7}` after being wrapped so the
    generated in-place/out-of-place function works with `StaticArrays`
    (MTK's `ODEFunction` supports out-of-place `SVector` calls directly when
    built with `jac=false` and no `sparse`; use that form rather than a
    hand-rolled wrapper if MTK provides it).

- **`src/dynamics/step.jl`**
  - `State = SVector{7,Float64}`, `Action = SVector{2,Float64}` type
    aliases.
  - `select_mode(state, params) -> Val{:ks}/Val{:st}` (or a `Bool`) —
    the `|v| < 0.5` branch, used to pick `ks_rhs`/`st_rhs` per call.
  - `dynamics_rhs(state, action, params) -> State` — applies
    `steering_constraint`/`accl_constraints` to `action`, calls
    `select_mode`, dispatches to `ks_rhs`/`st_rhs`. This is the single
    per-stage evaluation point shared by both the fixed-step and adaptive
    paths.
  - `step_dynamics(state::State, action::Action, params::VehicleParams,
    dt::Float64) -> State` — hand-written RK4 (no
    `OrdinaryDiffEq`/`DifferentialEquations.jl` dependency here — those
    integrators allocate an integrator object and are not meant for a
    zero-allocation per-step hot loop) calling `dynamics_rhs` 4 times,
    entirely in terms of `SVector` arithmetic. Wraps yaw (`state[5]`) into
    `[0, 2π)` after the step, matching `update_pose`.

- **`src/dynamics/solve.jl`**
  - `simulate(state0::State, control_fn, params::VehicleParams, tspan) ->
    ODESolution` — builds an `ODEProblem` around `dynamics_rhs` (wrapped to
    match the `(u, p, t) -> du` signature `DifferentialEquations.jl`
    expects, with `control_fn(t)` supplying the time-varying action) and
    calls `solve(...)` with an adaptive solver (`Tsit5()` default,
    overridable via keyword). This is the accuracy/offline path; it is
    expected to be a thin wrapper — `DifferentialEquations.jl` does the
    actual integration work, not hand-written code.

No new abstractions beyond these four files are introduced in Phase 1.

## Data flow

```
VehicleParams ──┐
                ├─▶ dynamics_rhs(state, action, params) ──▶ step_dynamics  (hot loop, Phase 4)
control input ──┘                    │
                                      └────────────────────▶ simulate       (offline, Phases 3/5)
```

Both call paths bottom out in the same `dynamics_rhs`, which bottoms out in
the same MTK-generated `ks_rhs`/`st_rhs`.

## Error handling

No runtime validation of state/parameter shapes — internal callers are
trusted, matching the project's stated engineering guidelines. The only
guards are the physical constraint clamps (`steering_constraint`,
`accl_constraints`), ported verbatim from Python, plus the yaw-angle wrap
after each `step_dynamics` call.

## Testing

`test/dynamics_test.jl`, ported directly from
`gym/f110_gym/unittest/dynamics_test.py` (values, not just structure):

- `test_derivatives` — exact `f_ks_gt`/`f_st_gt` 5-/7-element ground-truth
  derivative vectors for given `x_ks`/`x_st`/`u` inputs, asserting
  `dynamics_rhs` (or the underlying `ks_rhs`/`st_rhs` directly) matches to
  the same tolerance Python used (`assertAlmostEqual`, effectively 1e-7).
- Four zero-init singularity cases (`test_zeroinit_roll`,
  `test_zeroinit_dec`, `test_zeroinit_acc`, `test_zeroinit_rollleft`) — same
  initial states, same control inputs, same 1-second/1e-4-step rollouts
  (via `step_dynamics` looped, or `simulate`), asserting against the same
  ground-truth end-state literals within the same `1e-2` tolerance Python
  used.
- A 10-second `step_dynamics` RK4 rollout at `dt = 0.01` (Python's default
  `time_step`) from a fixed initial state and a fixed control sequence,
  asserting against reference trajectory values (computed once by running
  the actual Python `f1tenth_gym` code and hard-coded as literals) within
  `1e-5` — satisfying the phase's stated acceptance criterion.
- An allocation test: after a warmup call, `@allocated step_dynamics(state,
  action, params, dt) == 0`.

## Out of scope for Phase 1

- `pid()` / speed-steering-angle control convenience layer.
- Vehicle geometry (`width`/`length`), collision, map/track handling
  (Phase 2).
- Any rendering, concurrency, or reachability wiring (Phases 3–5) — Phase 1
  only needs to leave `ks_mode`/`st_mode` as independently reusable
  `ODESystem`s so Phase 5 can build a `HybridSystem` from them later.
