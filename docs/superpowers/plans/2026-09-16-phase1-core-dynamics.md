# Phase 1: Core Dynamics & Symbolic System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the single-track kinematic (KS) and dynamic (ST) bicycle
models from Python `f1tenth_gym` into `F1Tenth.jl`, exposing both an
adaptive-accuracy solve path and a zero-allocation fixed-step path built
from one shared set of `ModelingToolkit.jl`-generated equations.

**Architecture:** Two `ModelingToolkit.ODESystem`s (`ks_mode`, `st_mode`)
share one 7-element state space `[x, y, δ, v, ψ, ψ̇, β]`. Each is compiled
once (at module load) into an allocation-free out-of-place numerical
function via `Symbolics.build_function`. A plain Julia dispatcher
(`dynamics_rhs`) applies the physical control-input clamps, picks KS vs ST
by `|v| < 0.5`, and calls the matching generated function. `step_dynamics`
is a hand-written `SVector`-based RK4 built on `dynamics_rhs`, for the
real-time hot loop. `simulate` wraps the same `dynamics_rhs` in a
`DifferentialEquations.jl` `ODEProblem`/`solve` call, for offline/accuracy
use.

**Tech Stack:** Julia 1.13 (repo compat floor 1.10.10), ModelingToolkit.jl,
Symbolics.jl, StaticArrays.jl, DifferentialEquations.jl, Test (stdlib).

**Spec:** `docs/superpowers/specs/2026-09-16-phase1-core-dynamics-design.md`

## Global Constraints

- No Python API compatibility is required or attempted (per project-wide
  instruction) — only numeric/behavioral fidelity.
- One source of truth for the physics: the MTK symbolic equations. Never
  hand-duplicate the ODE math in a second place.
- `step_dynamics` must be zero-allocation after warmup (`@allocated == 0`).
- `ks_mode`/`st_mode` must remain independently usable `ODESystem`s (not
  inlined/erased) so a later phase can assemble a `HybridSystems.jl`
  `HybridSystem` from them without rework.
- Default `VehicleParams` values (from Python `f110_env.py`): `μ=1.0489,
  Cᵦᶠ=4.718, Cᵦʳ=5.4562, lᶠ=0.15875, lʳ=0.17145, h=0.074, m=3.74,
  Iᶻ=0.04712, δ⁻=-0.4189, δ⁺=0.4189, δ̇⁻=-3.2, δ̇⁺=3.2,
  vₛ=7.319, a⁺=9.51, v⁻=-5.0, v⁺=20.0`.

---

### Task 1: Vehicle parameters and control-input constraints

**Files:**
- Create: `src/dynamics/params.jl`
- Modify: `Project.toml` (add `StaticArrays` dependency; others added in
  later tasks)
- Test: `test/params_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Produces: `struct VehicleParams` with fields `μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h,
  m, Iᶻ, δ⁻, δ⁺, δ̇⁻, δ̇⁺, vₛ, a⁺, v⁻, v⁺ ::
  Float64`, and a keyword constructor `VehicleParams(; μ=1.0489, ...)`
  with the defaults listed in Global Constraints.
- Produces: `steering_constraint(steering_angle::Float64,
  steering_velocity::Float64, params::VehicleParams) -> Float64`.
- Produces: `accl_constraints(vel::Float64, accl::Float64,
  params::VehicleParams) -> Float64`.

- [ ] **Step 1: Add StaticArrays dependency**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add("StaticArrays")'
```
Expected: `Project.toml` gains a `[deps]` entry for `StaticArrays`, and
`Manifest.toml` is updated.

- [ ] **Step 2: Write the failing test for `VehicleParams` defaults**

Create `test/params_test.jl`:

```julia
using F1Tenth
using Test

@testset "VehicleParams" begin
    p = F1Tenth.VehicleParams()
    @test p.μ == 1.0489
    @test p.Cᵦᶠ == 4.718
    @test p.Cᵦʳ == 5.4562
    @test p.lᶠ == 0.15875
    @test p.lʳ == 0.17145
    @test p.h == 0.074
    @test p.m == 3.74
    @test p.Iᶻ == 0.04712
    @test p.δ⁻ == -0.4189
    @test p.δ⁺ == 0.4189
    @test p.δ̇⁻ == -3.2
    @test p.δ̇⁺ == 3.2
    @test p.vₛ == 7.319
    @test p.a⁺ == 9.51
    @test p.v⁻ == -5.0
    @test p.v⁺ == 20.0
end

@testset "steering_constraint" begin
    p = F1Tenth.VehicleParams()
    # within bounds: passes through unchanged
    @test F1Tenth.steering_constraint(0.0, 1.0, p) == 1.0
    # at upper angle limit, positive velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.δ⁺, 1.0, p) == 0.0
    # at lower angle limit, negative velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.δ⁻, -1.0, p) == 0.0
    # requested velocity below δ̇⁻ -> clamped
    @test F1Tenth.steering_constraint(0.0, -10.0, p) == p.δ̇⁻
    # requested velocity above δ̇⁺ -> clamped
    @test F1Tenth.steering_constraint(0.0, 10.0, p) == p.δ̇⁺
end

@testset "accl_constraints" begin
    p = F1Tenth.VehicleParams()
    # within bounds, below vₛ: passes through unchanged
    @test F1Tenth.accl_constraints(1.0, 2.0, p) == 2.0
    # at v⁺, positive accel requested -> zeroed
    @test F1Tenth.accl_constraints(p.v⁺, 1.0, p) == 0.0
    # at v⁻, negative accel requested -> zeroed
    @test F1Tenth.accl_constraints(p.v⁻, -1.0, p) == 0.0
    # requested accel below -a⁺ -> clamped
    @test F1Tenth.accl_constraints(1.0, -100.0, p) == -p.a⁺
    # above vₛ, positive-limit scales down: pos_limit = a⁺*vₛ/vel
    vel = p.vₛ * 2
    pos_limit = p.a⁺ * p.vₛ / vel
    @test F1Tenth.accl_constraints(vel, 100.0, p) == pos_limit
end
```

Modify `test/runtests.jl` to:

```julia
using F1Tenth
using Test

@testset "F1Tenth.jl" begin
    include("params_test.jl")
end
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `VehicleParams`/`steering_constraint`/`accl_constraints`
not defined.

- [ ] **Step 4: Implement `params.jl`**

Create `src/dynamics/params.jl`:

```julia
struct VehicleParams
    μ::Float64
    Cᵦᶠ::Float64
    Cᵦʳ::Float64
    lᶠ::Float64
    lʳ::Float64
    h::Float64
    m::Float64
    Iᶻ::Float64
    δ⁻::Float64
    δ⁺::Float64
    δ̇⁻::Float64
    δ̇⁺::Float64
    vₛ::Float64
    a⁺::Float64
    v⁻::Float64
    v⁺::Float64
end

function VehicleParams(;
    μ::Float64=1.0489,
    Cᵦᶠ::Float64=4.718,
    Cᵦʳ::Float64=5.4562,
    lᶠ::Float64=0.15875,
    lʳ::Float64=0.17145,
    h::Float64=0.074,
    m::Float64=3.74,
    Iᶻ::Float64=0.04712,
    δ⁻::Float64=-0.4189,
    δ⁺::Float64=0.4189,
    δ̇⁻::Float64=-3.2,
    δ̇⁺::Float64=3.2,
    vₛ::Float64=7.319,
    a⁺::Float64=9.51,
    v⁻::Float64=-5.0,
    v⁺::Float64=20.0,
)
    return VehicleParams(μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ, δ⁻, δ⁺,
                          δ̇⁻, δ̇⁺, vₛ, a⁺, v⁻, v⁺)
end

function steering_constraint(steering_angle::Float64, steering_velocity::Float64,
                              params::VehicleParams)
    if (steering_angle <= params.δ⁻ && steering_velocity <= 0) ||
       (steering_angle >= params.δ⁺ && steering_velocity >= 0)
        return 0.0
    elseif steering_velocity <= params.δ̇⁻
        return params.δ̇⁻
    elseif steering_velocity >= params.δ̇⁺
        return params.δ̇⁺
    else
        return steering_velocity
    end
end

function accl_constraints(vel::Float64, accl::Float64, params::VehicleParams)
    pos_limit = vel > params.vₛ ? params.a⁺ * params.vₛ / vel : params.a⁺
    if (vel <= params.v⁻ && accl <= 0) || (vel >= params.v⁺ && accl >= 0)
        return 0.0
    elseif accl <= -params.a⁺
        return -params.a⁺
    elseif accl >= pos_limit
        return pos_limit
    else
        return accl
    end
end
```

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/dynamics/params.jl test/params_test.jl test/runtests.jl
git commit -m "Add VehicleParams and control-input constraint functions"
```

---

### Task 2: MTK KS/ST symbolic models and generated RHS functions

**Files:**
- Create: `src/dynamics/mtk_models.jl`
- Modify: `Project.toml` (add `ModelingToolkit`, `Symbolics` dependencies)
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/mtk_models_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this task's generated functions
  take a plain parameter tuple, not `VehicleParams`, to keep the MTK layer
  independent of the params struct's representation).
- Produces: `const KS_RHS::Function` and `const ST_RHS::Function`, each
  callable as `f(state::SVector{7,Float64}, p::NTuple{10,Float64}) ->
  SVector{7,Float64}` where `p = (μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ, sv,
  accl)` — the 8 physics parameters (in that order) followed by the
  already-clamped control input `(steering_velocity, acceleration)`.
- Produces: `const KS_MODE::ModelingToolkit.ODESystem` and
  `const ST_MODE::ModelingToolkit.ODESystem`, kept as top-level bindings
  (not local to a function) so a later phase can import them directly for
  hybrid-system assembly.

- [ ] **Step 1: Add ModelingToolkit and Symbolics dependencies**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add(["ModelingToolkit", "Symbolics"])'
```
Expected: `Project.toml`/`Manifest.toml` updated.

- [ ] **Step 2: Write the failing test for RHS ground-truth derivatives**

These values are ported directly from
`f1tenth_gym/gym/f110_gym/unittest/dynamics_test.py`'s `test_derivatives`,
using that test's own vehicle parameters (the CommonRoad reference
vehicle, distinct from `VehicleParams`'s F1TENTH-scale defaults).

Create `test/mtk_models_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Test

@testset "KS_RHS / ST_RHS ground truth derivatives" begin
    # CommonRoad reference vehicle parameters used by the Python test suite
    μ = 1.0489
    Cᵦᶠ = 21.92 / 1.0489
    Cᵦʳ = 21.92 / 1.0489
    lᶠ = 0.3048 * 3.793293
    lʳ = 0.3048 * 4.667707
    h = 0.3048 * 2.01355
    m = 4.4482216152605 / 0.3048 * 74.91452
    Iᶻ = 4.4482216152605 * 0.3048 * 1321.416

    g = 9.81
    sv = 0.15
    accl = 0.63 * g

    p = (μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ, sv, accl)

    # x_ks from Python test, padded with psi_dot=0, beta=0 to fill the
    # shared 7-element state space
    x_ks = SVector{7,Float64}(3.9579422297936526, 0.0391650102771405,
                               0.0378491427211811, 16.3546957860883566,
                               0.0294717351052816, 0.0, 0.0)
    f_ks_gt = SVector{7,Float64}(16.3475935934250209, 0.4819314886013121,
                                  0.1500000000000000, 5.1464424102339752,
                                  0.2401426578627629, 0.0, 0.0)

    x_st = SVector{7,Float64}(2.0233348142065677, 0.0041907137716636,
                               0.0197545248559617, 15.7216236334290116,
                               0.0025857914776859, 0.0529001056654038,
                               0.0033012170610298)
    f_st_gt = SVector{7,Float64}(15.7213512030862397, 0.0925527979719355,
                                  0.1500000000000000, 5.3536773276413925,
                                  0.0529001056654038, 0.6435589397748606,
                                  0.0313297971641291)

    @test isapprox(F1Tenth.KS_RHS(x_ks, p), f_ks_gt; atol=1e-6)
    @test isapprox(F1Tenth.ST_RHS(x_st, p), f_st_gt; atol=1e-6)
end

@testset "KS_RHS / ST_RHS are allocation-free" begin
    μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ = 1.0489, 4.718, 5.4562, 0.15875, 0.17145, 0.074, 3.74, 0.04712
    p = (μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ, 0.1, 1.0)
    state = SVector{7,Float64}(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    F1Tenth.KS_RHS(state, p) # warmup
    F1Tenth.ST_RHS(state, p) # warmup
    @test (@allocated F1Tenth.KS_RHS(state, p)) == 0
    @test (@allocated F1Tenth.ST_RHS(state, p)) == 0
end
```

Modify `test/runtests.jl`:

```julia
using F1Tenth
using Test

@testset "F1Tenth.jl" begin
    include("params_test.jl")
    include("mtk_models_test.jl")
end
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `KS_RHS`/`ST_RHS` not defined.

- [ ] **Step 4: Implement `mtk_models.jl`**

Create `src/dynamics/mtk_models.jl`:

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics: build_function

@parameters mtk_mu mtk_C_Sf mtk_C_Sr mtk_lf mtk_lr mtk_h mtk_m mtk_Iz mtk_sv mtk_accl
@variables mtk_x(t) mtk_y(t) mtk_delta(t) mtk_v(t) mtk_psi(t) mtk_psi_dot(t) mtk_beta(t)

const _MTK_G = 9.81

const _ks_eqs = [
    D(mtk_x) ~ mtk_v * cos(mtk_psi),
    D(mtk_y) ~ mtk_v * sin(mtk_psi),
    D(mtk_delta) ~ mtk_sv,
    D(mtk_v) ~ mtk_accl,
    D(mtk_psi) ~ mtk_v / (mtk_lf + mtk_lr) * tan(mtk_delta),
    D(mtk_psi_dot) ~ 0.0,
    D(mtk_beta) ~ 0.0,
]

const _st_eqs = [
    D(mtk_x) ~ mtk_v * cos(mtk_beta + mtk_psi),
    D(mtk_y) ~ mtk_v * sin(mtk_beta + mtk_psi),
    D(mtk_delta) ~ mtk_sv,
    D(mtk_v) ~ mtk_accl,
    D(mtk_psi) ~ mtk_psi_dot,
    D(mtk_psi_dot) ~
        -mtk_mu * mtk_m / (mtk_v * mtk_Iz * (mtk_lr + mtk_lf)) *
            (mtk_lf^2 * mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h) +
             mtk_lr^2 * mtk_C_Sr * (_MTK_G * mtk_lf + mtk_accl * mtk_h)) * mtk_psi_dot +
        mtk_mu * mtk_m / (mtk_Iz * (mtk_lr + mtk_lf)) *
            (mtk_lr * mtk_C_Sr * (_MTK_G * mtk_lf + mtk_accl * mtk_h) -
             mtk_lf * mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h)) * mtk_beta +
        mtk_mu * mtk_m / (mtk_Iz * (mtk_lr + mtk_lf)) *
            mtk_lf * mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h) * mtk_delta,
    D(mtk_beta) ~
        (mtk_mu / (mtk_v^2 * (mtk_lr + mtk_lf)) *
            (mtk_C_Sr * (_MTK_G * mtk_lf + mtk_accl * mtk_h) * mtk_lr -
             mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h) * mtk_lf) - 1) * mtk_psi_dot -
        mtk_mu / (mtk_v * (mtk_lr + mtk_lf)) *
            (mtk_C_Sr * (_MTK_G * mtk_lf + mtk_accl * mtk_h) +
             mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h)) * mtk_beta +
        mtk_mu / (mtk_v * (mtk_lr + mtk_lf)) *
            (mtk_C_Sf * (_MTK_G * mtk_lr - mtk_accl * mtk_h)) * mtk_delta,
]

const KS_MODE = ODESystem(_ks_eqs, t; name=:ks_mode)
const ST_MODE = ODESystem(_st_eqs, t; name=:st_mode)

const _state_syms = [mtk_x, mtk_y, mtk_delta, mtk_v, mtk_psi, mtk_psi_dot, mtk_beta]
const _param_syms = [mtk_mu, mtk_C_Sf, mtk_C_Sr, mtk_lf, mtk_lr, mtk_h, mtk_m, mtk_Iz, mtk_sv, mtk_accl]

function _generate_rhs(eqs, state_syms, param_syms)
    rhss = [eq.rhs for eq in eqs]
    f_oop, _ = build_function(rhss, state_syms, param_syms; expression=Val{false})
    return f_oop
end

const KS_RHS = _generate_rhs(_ks_eqs, _state_syms, _param_syms)
const ST_RHS = _generate_rhs(_st_eqs, _state_syms, _param_syms)
```

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the allocation test fails because `build_function`'s
generated closure returns a `Vector` instead of an `SVector` when called
with an `SVector` state, this means this Julia/Symbolics version does not
auto-detect the static array output; in that case wrap `KS_RHS`/`ST_RHS`
to force zero-allocation, e.g. by rebuilding with
`build_function(rhss, state_syms, param_syms; expression=Val{false},
static_type=SVector{7,Float64})` if that keyword is available in the
resolved Symbolics.jl version, or by writing the 7 RHS scalar expressions
into a manually unrolled `SVector{7,Float64}(rhss[1], rhss[2], ..., rhss[7])`
constructor generated via `eval` of a hand-assembled `Expr` — check
`Symbolics.build_function`'s docstring (`?build_function` in the Julia
REPL) for the resolved version's exact keyword before choosing an
approach, since this varies across Symbolics.jl versions.

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/dynamics/mtk_models.jl test/mtk_models_test.jl test/runtests.jl
git commit -m "Add MTK KS/ST symbolic models and generated RHS functions"
```

---

### Task 3: Mode selection, `dynamics_rhs`, and fixed-step `step_dynamics`

**Files:**
- Create: `src/dynamics/step.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/step_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.VehicleParams`, `F1Tenth.steering_constraint`,
  `F1Tenth.accl_constraints` (Task 1); `F1Tenth.KS_RHS`, `F1Tenth.ST_RHS`
  (Task 2, each `f(state::SVector{7,Float64}, p::NTuple{10,Float64}) ->
  SVector{7,Float64}`).
- Produces: `const State = SVector{7,Float64}`, `const Action =
  SVector{2,Float64}` type aliases.
- Produces: `dynamics_rhs(state::State, action::Action,
  params::VehicleParams) -> State`.
- Produces: `step_dynamics(state::State, action::Action,
  params::VehicleParams, dt::Float64) -> State`.

- [ ] **Step 1: Write the failing tests**

These are ported directly from `dynamics_test.py`'s
`test_zeroinit_roll`/`test_zeroinit_dec`/`test_zeroinit_acc`/
`test_zeroinit_rollleft`, using that test's CommonRoad vehicle parameters.
Python's tests integrate for 1 simulated second using `scipy.odeint`
(adaptive); this port instead loops the fixed-step `step_dynamics` at
`dt=1e-3` for 1000 steps, which was verified (during spec/plan
preparation, by running the actual Python reference at matching
resolution) to land within the same `1e-2` tolerance Python's tests use.
Both the KS-only ground truth (looping a KS-only RHS) and the ST/switching
ground truth (looping the full `dynamics_rhs`) are checked, since Python's
suite validates both `vehicle_dynamics_ks` alone and
`vehicle_dynamics_st`'s internal KS/ST switch separately.

Create `test/step_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Test

function _commonroad_params()
    return F1Tenth.VehicleParams(;
        μ=1.0489,
        Cᵦᶠ=21.92 / 1.0489,
        Cᵦʳ=21.92 / 1.0489,
        lᶠ=0.3048 * 3.793293,
        lʳ=0.3048 * 4.667707,
        h=0.3048 * 2.01355,
        m=4.4482216152605 / 0.3048 * 74.91452,
        Iᶻ=4.4482216152605 * 0.3048 * 1321.416,
        δ⁻=-1.066,
        δ⁺=1.066,
        δ̇⁻=-0.4,
        δ̇⁺=0.4,
        v⁻=-13.6,
        v⁺=50.8,
        vₛ=7.319,
        a⁺=11.5,
    )
end

# Loops the KS RHS only, bypassing the |v|<0.5 mode switch, matching
# Python's separate func_KS-only rollout used for the *_ks_gt literals.
function _rollout_ks_only(state0::F1Tenth.State, action::F1Tenth.Action,
                           params::F1Tenth.VehicleParams, dt::Float64, steps::Int)
    state = state0
    for _ in 1:steps
        sv = F1Tenth.steering_constraint(state[3], action[1], params)
        accl = F1Tenth.accl_constraints(state[4], action[2], params)
        p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
             params.h, params.m, params.Iᶻ, sv, accl)
        k₁ = F1Tenth.KS_RHS(state, p)
        k₂ = F1Tenth.KS_RHS(state + dt * (k₁ / 2), p)
        k₃ = F1Tenth.KS_RHS(state + dt * (k₂ / 2), p)
        k₄ = F1Tenth.KS_RHS(state + dt * k₃, p)
        state = state + dt * (1 / 6) * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    end
    return state
end

const G = 9.81
const ZERO7 = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@testset "zeroinit_roll" begin
    p = _commonroad_params()
    u = F1Tenth.Action(0.0, 0.0)
    ks_final = _rollout_ks_only(ZERO7, u, p, 1e-3, 1000)
    st_final = ZERO7
    for _ in 1:1000
        st_final = F1Tenth.step_dynamics(st_final, u, p, 1e-3)
    end
    @test isapprox(ks_final[1:5], ZERO7[1:5]; atol=1e-2)
    @test isapprox(st_final, ZERO7; atol=1e-2)
end

@testset "zeroinit_dec" begin
    p = _commonroad_params()
    u = F1Tenth.Action(0.0, -0.7 * G)
    ks_gt = SVector{5,Float64}(-3.4335000000000013, 0.0, 0.0, -6.8670000000000018, 0.0)
    st_gt = F1Tenth.State(-3.4335000000000013, 0.0, 0.0, -6.8670000000000018, 0.0, 0.0, 0.0)
    ks_final = _rollout_ks_only(ZERO7, u, p, 1e-3, 1000)
    st_final = ZERO7
    for _ in 1:1000
        st_final = F1Tenth.step_dynamics(st_final, u, p, 1e-3)
    end
    @test isapprox(ks_final[1:5], ks_gt; atol=1e-2)
    @test isapprox(st_final, st_gt; atol=1e-2)
end

@testset "zeroinit_acc" begin
    p = _commonroad_params()
    u = F1Tenth.Action(0.15, 0.63 * G)
    ks_gt = SVector{5,Float64}(3.0845676868494927, 0.1484249221523042,
                                0.1500000000000000, 6.1803000000000017,
                                0.1203664469224163)
    st_gt = F1Tenth.State(3.0731976046859715, 0.2869835398304389,
                           0.1500000000000000, 6.1802999999999999,
                           0.1097747074946325, 0.3248268063223301,
                           0.0697547542798040)
    ks_final = _rollout_ks_only(ZERO7, u, p, 1e-3, 1000)
    st_final = ZERO7
    for _ in 1:1000
        st_final = F1Tenth.step_dynamics(st_final, u, p, 1e-3)
    end
    @test isapprox(ks_final[1:5], ks_gt; atol=1e-2)
    @test isapprox(st_final, st_gt; atol=1e-2)
end

@testset "zeroinit_rollleft" begin
    p = _commonroad_params()
    u = F1Tenth.Action(0.15, 0.0)
    ks_gt = SVector{5,Float64}(0.0, 0.0, 0.15, 0.0, 0.0)
    st_gt = F1Tenth.State(0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0)
    ks_final = _rollout_ks_only(ZERO7, u, p, 1e-3, 1000)
    st_final = ZERO7
    for _ in 1:1000
        st_final = F1Tenth.step_dynamics(st_final, u, p, 1e-3)
    end
    @test isapprox(ks_final[1:5], ks_gt; atol=1e-2)
    @test isapprox(st_final, st_gt; atol=1e-2)
end

@testset "10-second rollout matches Python RK4 reference" begin
    # Default F1TENTH-scale VehicleParams, constant control input,
    # dt=0.01 (Python's default time_step), 1000 steps. These checkpoints
    # were generated by running the actual Python f1tenth_gym RK4
    # integration (base_classes.py's update_pose algorithm) at the same
    # dt with the same constant control input, starting from rest.
    p = F1Tenth.VehicleParams()
    u = F1Tenth.Action(0.1, 2.0)
    dt = 0.01
    state = ZERO7
    checkpoints = Dict(
        100 => F1Tenth.State(0.9930439005626006, 0.09991503614876832,
                              0.10000000000000006, 2.0000000000000013,
                              0.18664458347493487, 0.5459894203054845,
                              0.030548162189671327),
        500 => F1Tenth.State(-0.4470799399978235, 0.989101019047281,
                              0.41933333333333367, 9.999999999999876,
                              6.089250561070209, 4.446247165726759,
                              -0.6276888826193096),
        1000 => F1Tenth.State(-5.652813793698802, 2.744955875560934,
                               0.41933333333333367, 19.999999999999662,
                               5.569501841414595, 2.962599570486635,
                               -0.9299998114470461),
    )
    for i in 1:1000
        state = F1Tenth.step_dynamics(state, u, p, dt)
        if haskey(checkpoints, i)
            @test isapprox(state, checkpoints[i]; atol=1e-5)
        end
    end
end

@testset "step_dynamics is allocation-free" begin
    p = F1Tenth.VehicleParams()
    u = F1Tenth.Action(0.1, 1.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    F1Tenth.step_dynamics(state, u, p, 0.01) # warmup
    @test (@allocated F1Tenth.step_dynamics(state, u, p, 0.01)) == 0
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
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `State`/`Action`/`step_dynamics` not defined.

- [ ] **Step 3: Implement `step.jl`**

Create `src/dynamics/step.jl`:

```julia
using StaticArrays

const State = SVector{7,Float64}
const Action = SVector{2,Float64}

@inline function dynamics_rhs(state::State, action::Action, params::VehicleParams)
    sv = steering_constraint(state[3], action[1], params)
    accl = accl_constraints(state[4], action[2], params)
    p = (params.μ, params.Cᵦᶠ, params.Cᵦʳ, params.lᶠ, params.lʳ,
         params.h, params.m, params.Iᶻ, sv, accl)
    return abs(state[4]) < 0.5 ? KS_RHS(state, p) : ST_RHS(state, p)
end

function step_dynamics(state::State, action::Action, params::VehicleParams, dt::Float64)
    k₁ = dynamics_rhs(state, action, params)
    k₂ = dynamics_rhs(state + dt * (k₁ / 2), action, params)
    k₃ = dynamics_rhs(state + dt * (k₂ / 2), action, params)
    k₄ = dynamics_rhs(state + dt * k₃, action, params)
    new_state = state + dt * (1 / 6) * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

    yaw = new_state[5]
    if yaw > 2 * pi
        yaw -= 2 * pi
    elseif yaw < 0
        yaw += 2 * pi
    end

    return State(new_state[1], new_state[2], new_state[3], new_state[4],
                 yaw, new_state[6], new_state[7])
end
```

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")

end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If the allocation test fails, check whether
`dynamics_rhs`'s ternary is causing a type instability between the
`KS_RHS`/`ST_RHS` branches (both must return the same concrete
`SVector{7,Float64}` type — verify with `@code_warntype dynamics_rhs(...)`
in the REPL) before making further changes.

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/dynamics/step.jl test/step_test.jl test/runtests.jl
git commit -m "Add mode-switching dynamics_rhs and fixed-step step_dynamics"
```

---

### Task 4: Adaptive-accuracy `simulate` via DifferentialEquations.jl

**Files:**
- Create: `src/dynamics/solve.jl`
- Modify: `Project.toml` (add `DifferentialEquations` dependency)
- Modify: `src/F1Tenth.jl` (include the new file)
- Test: `test/solve_test.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: `F1Tenth.State`, `F1Tenth.Action`, `F1Tenth.VehicleParams`,
  `F1Tenth.dynamics_rhs` (Task 3).
- Produces: `simulate(state0::State, control_fn, params::VehicleParams,
  tspan::Tuple{Float64,Float64}; solver=Tsit5()) ->
  SciMLBase.ODESolution`, where `control_fn(t::Float64) -> Action` supplies
  the control input at time `t`.

- [ ] **Step 1: Add DifferentialEquations dependency**

Run:
```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
julia --project=. -e 'import Pkg; Pkg.add("DifferentialEquations")'
```
Expected: `Project.toml`/`Manifest.toml` updated.

- [ ] **Step 2: Write the failing test**

This exercises `simulate` on the same mild, non-saturating 1-second
"zeroinit_acc" scenario from Task 3 (CommonRoad params), which was
verified (during spec/plan preparation, via `scipy.solve_ivp` at tight
tolerance against the same ground truth) to land within `1e-2`. A
saturating, longer scenario is deliberately not used here: the ODE's
control-input clamps make the vector field non-smooth at the clamp
boundary, and an adaptive solver's step placement around that
non-smoothness diverges from a fixed-step RK4's — that divergence is
expected numerical behavior, not a bug, so it is not used as a pass/fail
signal.

Create `test/solve_test.jl`:

```julia
using F1Tenth
using Test

@testset "simulate matches ground truth on a mild scenario" begin
    p = F1Tenth.VehicleParams(;
        μ=1.0489,
        Cᵦᶠ=21.92 / 1.0489,
        Cᵦʳ=21.92 / 1.0489,
        lᶠ=0.3048 * 3.793293,
        lʳ=0.3048 * 4.667707,
        h=0.3048 * 2.01355,
        m=4.4482216152605 / 0.3048 * 74.91452,
        Iᶻ=4.4482216152605 * 0.3048 * 1321.416,
        δ⁻=-1.066,
        δ⁺=1.066,
        δ̇⁻=-0.4,
        δ̇⁺=0.4,
        v⁻=-13.6,
        v⁺=50.8,
        vₛ=7.319,
        a⁺=11.5,
    )
    u = F1Tenth.Action(0.15, 0.63 * 9.81)
    state0 = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gt = F1Tenth.State(3.0731976046859715, 0.2869835398304389,
                        0.1500000000000000, 6.1802999999999999,
                        0.1097747074946325, 0.3248268063223301,
                        0.0697547542798040)

    sol = F1Tenth.simulate(state0, _ -> u, p, (0.0, 1.0))
    final = F1Tenth.State(sol.u[end])

    @test isapprox(final, gt; atol=1e-2)
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
end
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `simulate` not defined.

- [ ] **Step 4: Implement `solve.jl`**

Create `src/dynamics/solve.jl`:

```julia
using DifferentialEquations

function simulate(state0::State, control_fn, params::VehicleParams,
                   tspan::Tuple{Float64,Float64}; solver=Tsit5())
    f = (u, _p, t) -> dynamics_rhs(State(u), control_fn(t), params)
    prob = ODEProblem(f, state0, tspan)
    return solve(prob, solver)
end
```

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add Project.toml Manifest.toml src/F1Tenth.jl src/dynamics/solve.jl test/solve_test.jl test/runtests.jl
git commit -m "Add DifferentialEquations.jl-based adaptive simulate path"
```

---

### Task 5: Module exports and full-suite verification

**Files:**
- Modify: `src/F1Tenth.jl`

**Interfaces:**
- Consumes: everything produced in Tasks 1-4.
- Produces: public names exported from `F1Tenth` so downstream phases
  don't need `F1Tenth.`-prefixed access: `VehicleParams`, `State`,
  `Action`, `step_dynamics`, `simulate`, `KS_MODE`, `ST_MODE`.

- [ ] **Step 1: Add exports**

Modify `src/F1Tenth.jl`:

```julia
module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE

end
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS (all testsets from Tasks 1-4).

- [ ] **Step 3: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl
git commit -m "Export Phase 1 public API from F1Tenth module"
```
