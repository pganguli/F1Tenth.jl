# Phase 4: Async Concurrency & Control Channels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-threaded async sim loop, `AsyncSim`, that wraps
Phase 2's `step_multi_agent!` and Phase 3's `render!` in an owned
background pipeline: per-agent input `Channel`s for control actions
in, an output `Channel` for observations out, and independent
renderer `Task`s — so a caller no longer has to hand-write a
synchronous `while` loop calling both in lockstep.

**Architecture:** One reusable primitive, `LatestChannel{T}`, backs
both per-agent input (physics keeps a cached last-action-per-agent,
zero-order-hold) and default render-frame delivery (drop oldest, keep
latest). `AsyncSim` owns a physics `Threads.@spawn` `Task` and one
renderer `Task` per attached `RendererEntry`; each entry picks
`DropLatest` (never blocks physics) or `Blocking` (physics stalls if
that renderer falls behind — for `CairoMakieRenderer` per-tick
flip-book recording). Tick pacing is one code path with two modes:
`tick_hz=nothing` runs flat-out (training), a numeric `tick_hz`
sleeps to a wall-clock deadline each tick (live/interactive). Any
tier's exception is fail-fast: it closes every channel (waking the
other tiers, which treat that as a clean-exit signal) and is
re-surfaced by `stop!` to the caller.

**Tech Stack:** Julia 1.13 (repo compat floor 1.10.10), `Base.Threads`
(stdlib, `Threads.@spawn`/`Atomic`/`nthreads`), `Random` (existing
dep). No new `Project.toml` dependencies — everything Phase 4 needs is
either already a dependency or part of Base/stdlib.

> **Amendment 4 (post-final-review, after all 4 tasks passed their own
> reviews):** The final whole-branch review found one Critical and
> several Important cross-task issues that no single task's diff could
> reveal, since each depends on how a *later* task's code actually
> uses an *earlier* task's primitive:
>
> - **Critical — `put_latest!` still races `wait_latest!`.** Task 1's
>   fix closed the `put_latest!`/`take_latest!` race with a custom
>   `lc.lock`, reasoned to be safe for `wait_latest!` on the grounds
>   that a `put!` on an empty `Channel` with a pending `take!` waiter
>   is a non-blocking handoff. That reasoning only covers
>   `put_latest!`'s `put!` call, not its internal `take!` (the drain of
>   a stale value): if `wait_latest!`'s bare `take!` claims the pending
>   value in the window between `put_latest!`'s `isready` check and its
>   own `take!`, `put_latest!`'s `take!` blocks forever on a now-empty
>   channel — while holding `lc.lock`. Task 3 introduced exactly this
>   pairing on every `DropLatest` renderer channel (physics tier's
>   `put_latest!` vs. that renderer's `wait_latest!`), which didn't
>   exist when Task 1 was reviewed. Reproduced empirically against the
>   shipped code. Fix: drop the custom `lc.lock` field entirely and use
>   `Channel`'s own built-in lock (`lock(lc.ch) do ... end`, a
>   `ReentrantLock` Julia already associates with every `Channel`) for
>   `put_latest!`'s and `take_latest!`'s compound check-then-act — a
>   concurrent `wait_latest!`'s `take!` call acquires that same lock
>   internally, so it's now properly excluded too.
> - **Important — `start!`/`reset!` only guard on `running[]`, not on
>   whether the previous tiers have actually finished unwinding.** A
>   failing tier sets `running[] = false` before it (and any other
>   tier) has returned from its `catch` block. `start!` (or `reset!`)
>   called in that window passes the guard while an old task is still
>   live, discarding a real recorded error and risking two physics
>   tasks touching `sim.states` at once. Fix: guard on task quiescence
>   (`sim.physics_task === nothing && isempty(sim.renderer_tasks)`)
>   instead of `running[]` — `stop!` is the only thing that nulls those
>   fields, so this is a strictly stronger check.
> - **Important — actions pushed to `sim.input` before the first
>   `start!` are silently discarded.** `start!` unconditionally rebuilds
>   every channel, including ones nothing has closed yet. Fix: rebuild
>   a channel only if it's actually closed (`isopen(...) ? existing :
>   fresh`), for input, output, and every renderer entry.
> - **Important — `AsyncSim`'s usage contract lives only in the plan
>   and spec, not in the code.** Fix: add a comment block above
>   `mutable struct AsyncSim` in `src/async/sim.jl` stating: `start!`/
>   `stop!`/`reset!` must be called from a single owning thread/task;
>   `start!` invalidates every previously-obtained channel reference,
>   so a caller must push to `sim.input[i]` only *after* `start!`
>   returns (matching the Task-3 amendment's corrected wording above);
>   `stop!` rethrows whatever exception, if any, a tier recorded; and
>   an external `put_latest!`/`take_latest!` call can raise
>   `InvalidStateException` once a tier has failed or after `stop!`.
> - **Important — two test skip guards check `Threads.nthreads() >= 2`
>   but the testsets they gate all attach a renderer, so they actually
>   need `>= 3`** (`_required_threads` includes the renderer). Fix:
>   correct both guards to `>= 3`.
> - **Flake fix, same underlying non-atomicity as the Critical above,
>   benign guise:** Task 2's "tick_hz real-time mode runs without
>   error" test reads `sim.output` *after* `stop!`, so if `stop!`
>   closes channels inside `put_latest!`'s drain-then-put window, the
>   last frame before shutdown can be lost and the read comes back
>   `missing` — this reproduces the one flaky failure noted during
>   Task 3/4. Fix: capture the frame *before* `stop!`, using the same
>   poll-with-timeout loop the lifecycle test in Task 2 already uses.
>
> All six are implementation fixes within already-reviewed files; none
> change the public behavior the spec describes. Minor findings from
> the final review (a `tick_hz` catch-up-burst edge case, a confusing
> thread-count error message, a narrow `stop!` catch-all, an untested
> zero-agent/fail-on-first-call path, and a spec-wording note about
> `reset!`'s zero-action default) are deliberately deferred as
> follow-up work, not fixed here — see the final review's Deferred
> Findings Triage in the SDD ledger for the reasoning on each.

**Spec:** `docs/superpowers/specs/2026-09-16-phase4-async-concurrency-design.md`

## Global Constraints

- No file under `src/dynamics/`, `src/geometry/`, or `src/render/` may
  be modified. All Phase 4 code lives under `src/async/` (spec
  Non-goals).
- `step_multi_agent!` and `render!` are called unmodified — Phase 4
  only orchestrates calls into them, never changes their signatures or
  behavior (spec Non-goals).
- There is no dedicated input-processing thread. External code writes
  actions directly into per-agent `LatestChannel`s; the only two owned
  thread kinds are physics (one) and render (one per `RendererEntry`)
  (spec Decision 2).
- `LatestChannel{T}` is the only channel primitive Phase 4 defines;
  both zero-order-hold input and drop-latest render backpressure are
  built from it (spec Decision 1).
- Physics reuses each agent's last-received action when nothing new
  has arrived by tick time — a zero-order hold, never blocks on one
  slow/silent agent (spec Decision 3).
- Render backpressure is chosen per `RendererEntry` at construction:
  `DropLatest` (default, physics never blocks) or `Blocking` (for
  per-tick `CairoMakieRenderer` flip-book recording, where every tick
  must be captured) (spec Decision 4).
- The output side is an async `LatestChannel`, not a blocking
  `step!(sim, actions) -> observations` call (spec Decision 5).
- `start!` asserts `nthreads() >= 2 + length(sim.renderers)` before
  spawning anything, erroring with the exact required count rather
  than silently degrading (spec Decision 7).
- Any tier's exception is fail-fast across the whole `AsyncSim` — no
  "renderer errors are best-effort" carve-out (spec Decision 8).
- `reset!(sim::AsyncSim, track::Track, initial_states::Vector{State})`
  is the exact signature spec Decision 7 fixes; it requires the sim
  already stopped, and it must not silently reuse whatever the
  previous episode's last actions were. This plan resolves that by
  zeroing `last_actions` to `zero(Action)` on `reset!` — a caller
  wanting different reset actions pushes them onto the input channels
  *after* the next `start!` returns (not before: `start!` rebuilds
  every channel from scratch — see Task 3's restart-fix amendment — so
  a channel reference obtained before that call is stale). This is the
  same zero-order-hold, best-effort timing as pushing actions during
  normal operation (Decision 3): whichever action has arrived by the
  time a tick reads it is what's used, not necessarily tick 1
  specifically. (Small gap-filling decision, same kind as Phase 3's
  `CairoMakieRenderer` `summary_path` — the spec fixed the behavior's
  intent and signature but not this one implementation detail; the
  "before the next `start!`" phrasing here was corrected after Task
  3's review caught that it described a physically unachievable
  window — a channel `start!` hasn't built yet can't be pushed to.)

---

### Task 1: `LatestChannel{T}` primitive

**Files:**
- Create: `src/async/latest_channel.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Create: `test/async/latest_channel_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Produces: `struct LatestChannel{T}` wrapping a capacity-1
  `Channel{T}`.
- Produces: `put_latest!(lc::LatestChannel{T}, v::T) where {T} ->
  Nothing` — never blocks the producer, overwrites any pending value.
- Produces: `take_latest!(lc::LatestChannel{T}) where {T} -> Union{T,
  Missing}` — never blocks, returns `missing` on empty.
- Produces: `wait_latest!(lc::LatestChannel{T}) where {T} -> T` —
  blocks until a value is available (used by the render tier in Task
  3 instead of busy-polling `take_latest!`).

- [ ] **Step 1: Write the failing tests**

Create `test/async/latest_channel_test.jl`:

```julia
using F1Tenth
using Test

@testset "LatestChannel: put_latest! never blocks and overwrites a pending value" begin
    lc = F1Tenth.LatestChannel{Int}()
    F1Tenth.put_latest!(lc, 1)
    F1Tenth.put_latest!(lc, 2)
    @test F1Tenth.take_latest!(lc) === 2
end

@testset "LatestChannel: take_latest! returns missing on empty and drains exactly once" begin
    lc = F1Tenth.LatestChannel{Int}()
    @test F1Tenth.take_latest!(lc) === missing
    F1Tenth.put_latest!(lc, 5)
    @test F1Tenth.take_latest!(lc) === 5
    @test F1Tenth.take_latest!(lc) === missing
end

@testset "LatestChannel: wait_latest! blocks until a value is available" begin
    lc = F1Tenth.LatestChannel{Int}()
    t = Threads.@spawn F1Tenth.wait_latest!(lc)
    sleep(0.05)
    @test !istaskdone(t)
    F1Tenth.put_latest!(lc, 7)
    @test fetch(t) == 7
end
```

Modify `test/runtests.jl` (add after the existing `exports_test.jl`
include, since `exports_test.jl` will itself grow across Tasks 2-4 —
keeping it last means every earlier include has already run):

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
    include("exports_test.jl")
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.LatestChannel` not defined.

- [ ] **Step 3: Implement `latest_channel.jl`**

> **Amendment (post-Task-1 review):** The first version of this block
> composed `isready(lc.ch)` and `take!(lc.ch)`/`put!(lc.ch, v)` as two
> separate, non-atomic steps in both `put_latest!` and `take_latest!`.
> Task 1's review caught a real race: under genuine concurrent use
> (the physics tier's `take_latest!` racing an external caller's
> `put_latest!` on the same input channel, or symmetrically on the
> output channel), one side's `isready` can see a stale value that the
> other side drains first, so the loser's own `take!`/`put!` call
> blocks — violating the "never blocks the producer/consumer"
> guarantee this primitive exists to provide. `wait_latest!` is not
> part of this race (a `put!` on an empty `Channel` with a pending
> `take!` waiter is a direct, non-blocking handoff), and per this
> package's actual call sites `wait_latest!` and `take_latest!` are
> never both used against the same `LatestChannel` instance (a
> renderer's `DropLatest` channel only ever receives `wait_latest!`
> calls from its own render task; input/output channels only ever
> receive `take_latest!` calls from their sole consumer) — so guarding
> only `put_latest!`/`take_latest!` with a shared lock closes the race
> without touching `wait_latest!`. Fixed version below; ship this one.

Create `src/async/latest_channel.jl`:

```julia
# Single-producer, single-consumer "latest value" slot: put_latest!
# drops any value the consumer hasn't taken yet, so a fast producer
# never blocks and a slow consumer only ever sees the most recent
# value. Not safe for multiple concurrent producers on the same
# LatestChannel -- every use site in this package guarantees exactly
# one producer per instance (physics is the sole producer for output
# and render channels; external caller code is the sole producer for
# each agent's input channel).
struct LatestChannel{T}
    ch::Channel{T}
    lock::ReentrantLock
end

LatestChannel{T}() where {T} = LatestChannel{T}(Channel{T}(1), ReentrantLock())

# put_latest! and take_latest! share `lock` because each is a
# check-then-act (isready, then take!/put!) that is not atomic on its
# own -- without the lock, a concurrent put_latest!/take_latest! pair
# can race such that one side's isready check is stale by the time it
# acts, and that side's take!/put! call blocks. See the amendment note
# above this block.
function put_latest!(lc::LatestChannel{T}, v::T) where {T}
    lock(lc.lock) do
        isready(lc.ch) && take!(lc.ch)
        put!(lc.ch, v)
    end
    return nothing
end

function take_latest!(lc::LatestChannel{T}) where {T}
    return lock(lc.lock) do
        isready(lc.ch) ? take!(lc.ch) : missing
    end
end

# wait_latest! deliberately does not take `lock` -- it must be able to
# block waiting for a value without holding a lock that put_latest!
# needs in order to complete. This is safe without the lock: a put! on
# an empty Channel with a pending take! waiter is a direct,
# non-blocking handoff. This does require that wait_latest! and
# take_latest! are never both called against the same LatestChannel
# instance -- true of every use in this package (see the amendment
# note above this block).
wait_latest!(lc::LatestChannel{T}) where {T} = take!(lc.ch)
```

Modify `src/F1Tenth.jl` to add the new `include`, without touching the
`export` lines yet (matching Phase 2/3's pattern — exports are only
extended once, in the final task):

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/async/latest_channel.jl test/async/latest_channel_test.jl test/runtests.jl
git commit -m "Add LatestChannel primitive for Phase 4 async pipeline"
```

---

### Task 2: `AsyncSim` core — physics tier, lifecycle, tick pacing, thread guard

**Files:**
- Create: `src/async/sim.jl`
- Modify: `src/F1Tenth.jl` (include the new file)
- Create: `test/async/testutils.jl`
- Create: `test/async/sim_test.jl`
- Modify: `test/runtests.jl` (include the new test file)

**Interfaces:**
- Consumes: `LatestChannel`, `put_latest!`, `take_latest!` (Task 1);
  `State`, `Action` (Phase 1); `Track`, `TrackBVH`, `build_bvh`,
  `build_track` (Phase 2); `VehicleParams`, `CarFootprint`,
  `LiDARConfig`, `step_multi_agent!` (Phase 1/2) — all unqualified,
  since `sim.jl` is `include`d into the `F1Tenth` module.
- Produces: `const RenderFrame = Tuple{Vector{State},
  Vector{Vector{Float64}}, Vector{Bool}}`.
- Produces: `mutable struct AsyncSim` with constructor
  `AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint,
  dt::Float64, cfg::LiDARConfig, initial_states::Vector{State},
  initial_actions::Vector{Action}; tick_hz::Union{Float64,Nothing}=nothing,
  rng::AbstractRNG=default_rng()) -> AsyncSim`.
- Produces: `start!(sim::AsyncSim) -> AsyncSim`,
  `stop!(sim::AsyncSim) -> AsyncSim`.
- Produces (test-only): `_small_track() -> Track`,
  `_async_test_setup(; n_agents=1) -> NamedTuple` with fields `track,
  bvh, params, car, cfg, states, actions`, reused by every async test
  file in this and later tasks.

- [ ] **Step 1: Write the failing tests**

Create `test/async/testutils.jl`:

```julia
using StaticArrays

function _small_track()
    n = 8
    R = 5.0
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
end

function _async_test_setup(; n_agents::Int=1)
    track = _small_track()
    bvh = F1Tenth.build_bvh(track.segments)
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=1.0, max_range=10.0, noise_std=0.0)
    states = [F1Tenth.State(Float64(i - 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in 1:n_agents]
    actions = [F1Tenth.Action(0.0, 0.0) for _ in 1:n_agents]
    return (; track, bvh, params, car, cfg, states, actions)
end
```

Create `test/async/sim_test.jl`:

```julia
using F1Tenth
using Test

include("testutils.jl")

@testset "AsyncSim: constructor requires one action per agent" begin
    setup = _async_test_setup(; n_agents=2)
    @test_throws ArgumentError F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01,
        setup.cfg, setup.states, [F1Tenth.Action(0.0, 0.0)])
end

@testset "AsyncSim: start! enforces the minimum thread count" begin
    setup = _async_test_setup()
    sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
        setup.states, setup.actions)
    if Threads.nthreads() < 2
        @test_throws ErrorException F1Tenth.start!(sim)
    else
        @warn "Skipping AsyncSim thread-guard test: needs Threads.nthreads() < 2 to exercise the failure path; got $(Threads.nthreads())"
    end
end

if Threads.nthreads() >= 2
    @testset "AsyncSim: start!/stop! lifecycle advances physics and produces output" begin
        setup = _async_test_setup()
        # Uses a nonzero initial velocity, not setup.states/setup.actions
        # (both all-zero): step_dynamics on an all-zero state with a
        # zero-order-held all-zero action is an exact fixed point (the
        # derivative is identically zero), so a "did the state change"
        # assertion against the default setup would always fail
        # regardless of whether the physics tier is ticking correctly.
        moving_states = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            moving_states, setup.actions)

        F1Tenth.start!(sim)
        local frame = missing
        for _ in 1:200
            frame = F1Tenth.take_latest!(sim.output)
            frame === missing || break
            sleep(0.01)
        end
        F1Tenth.stop!(sim)

        @test frame !== missing
        new_states, scans, collisions = frame
        @test new_states[1][1] > moving_states[1][1]  # x position advanced under constant forward velocity
        @test length(scans) == 1
        @test length(collisions) == 1
    end

    @testset "AsyncSim: input actions reach the physics tier (zero-order hold)" begin
        setup = _async_test_setup()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions)

        F1Tenth.start!(sim)
        F1Tenth.put_latest!(sim.input[1], F1Tenth.Action(0.0, 5.0))
        sleep(0.2)
        F1Tenth.stop!(sim)

        @test sim.last_actions[1] == F1Tenth.Action(0.0, 5.0)
        @test sim.states[1][4] > 0.0  # velocity increased under sustained acceleration
    end

    @testset "AsyncSim: tick_hz real-time mode runs without error" begin
        setup = _async_test_setup()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; tick_hz=20.0)

        F1Tenth.start!(sim)
        sleep(0.3)
        F1Tenth.stop!(sim)

        @test F1Tenth.take_latest!(sim.output) !== missing
    end
else
    @warn "Skipping AsyncSim lifecycle tests: needs Threads.nthreads() >= 2; got $(Threads.nthreads())"
end
```

Modify `test/runtests.jl` (add after `latest_channel_test.jl`):

```julia
    include("async/latest_channel_test.jl")
    include("async/sim_test.jl")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.AsyncSim` not defined.

- [ ] **Step 3: Implement `sim.jl`**

Create `src/async/sim.jl`:

```julia
using Random: AbstractRNG, default_rng
using Base.Threads: Atomic, @spawn, nthreads

const RenderFrame = Tuple{Vector{State},Vector{Vector{Float64}},Vector{Bool}}

mutable struct AsyncSim
    bvh::TrackBVH
    params::VehicleParams
    car::CarFootprint
    dt::Float64
    cfg::LiDARConfig
    tick_hz::Union{Float64,Nothing}
    input::Vector{LatestChannel{Action}}
    last_actions::Vector{Action}
    states::Vector{State}
    output::LatestChannel{RenderFrame}
    running::Atomic{Bool}
    physics_task::Union{Task,Nothing}
    error::Ref{Union{Nothing,Exception}}
    error_lock::ReentrantLock
    rng::AbstractRNG
end

function AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64,
    cfg::LiDARConfig, initial_states::Vector{State}, initial_actions::Vector{Action};
    tick_hz::Union{Float64,Nothing}=nothing, rng::AbstractRNG=default_rng())
    n = length(initial_states)
    length(initial_actions) == n || throw(ArgumentError(
        "initial_actions must have one entry per agent (got $(length(initial_actions)) for $n agents)"))
    input = [LatestChannel{Action}() for _ in 1:n]
    return AsyncSim(bvh, params, car, dt, cfg, tick_hz, input, copy(initial_actions),
        copy(initial_states), LatestChannel{RenderFrame}(), Atomic{Bool}(false),
        nothing, Ref{Union{Nothing,Exception}}(nothing), ReentrantLock(), rng)
end

_required_threads(sim::AsyncSim) = 2

function _record_error!(sim::AsyncSim, e::Exception)
    lock(sim.error_lock) do
        sim.error[] === nothing && (sim.error[] = e)
    end
    return nothing
end

function _close_all_channels!(sim::AsyncSim)
    for lc in sim.input
        close(lc.ch)
    end
    close(sim.output.ch)
    return nothing
end

function _physics_tick!(sim::AsyncSim)
    n = length(sim.states)
    actions = Vector{Action}(undef, n)
    for i in 1:n
        a = take_latest!(sim.input[i])
        a === missing || (sim.last_actions[i] = a)
        actions[i] = sim.last_actions[i]
    end
    new_states, scans, collisions = step_multi_agent!(sim.states, actions, sim.bvh,
        sim.params, sim.car, sim.dt, sim.cfg; rng=sim.rng)
    sim.states = new_states
    put_latest!(sim.output, (new_states, scans, collisions))
    return nothing
end

function _physics_loop(sim::AsyncSim)
    deadline = time() + (sim.tick_hz === nothing ? 0.0 : 1 / sim.tick_hz)
    try
        while sim.running[]
            _physics_tick!(sim)
            if sim.tick_hz !== nothing
                sleep(max(0.0, deadline - time()))
                deadline += 1 / sim.tick_hz
            end
        end
    catch e
        if !(e isa InvalidStateException)
            _record_error!(sim, e)
            sim.running[] = false
            _close_all_channels!(sim)
            rethrow()
        end
    end
    return nothing
end

function start!(sim::AsyncSim)
    sim.running[] && error("AsyncSim: already running")
    required = _required_threads(sim)
    nthreads() >= required || error(
        "AsyncSim: needs Threads.nthreads() >= $required; got $(nthreads()). " *
        "Start Julia with `julia --threads=$required` or set JULIA_NUM_THREADS=$required.")
    sim.error[] = nothing
    sim.running[] = true
    sim.physics_task = @spawn _physics_loop(sim)
    return sim
end

function stop!(sim::AsyncSim)
    sim.running[] = false
    _close_all_channels!(sim)
    if sim.physics_task !== nothing
        try
            fetch(sim.physics_task)
        catch
            # the real cause, if any, is captured in sim.error and rethrown below
        end
    end
    sim.physics_task = nothing
    if sim.error[] !== nothing
        err = sim.error[]
        sim.error[] = nothing
        throw(err)
    end
    return sim
end
```

Modify `src/F1Tenth.jl` to add `include("async/sim.jl")` immediately
after `include("async/latest_channel.jl")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If run with the default single Julia thread, the
thread-guard test exercises its failure path and the three
lifecycle/pacing tests are skipped with a `@warn` — expected, not a
failure to chase (Task 4 adds the CI thread bump that makes them run
for real in CI).

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/F1Tenth.jl src/async/sim.jl test/async/testutils.jl test/async/sim_test.jl test/runtests.jl
git commit -m "Add AsyncSim physics tier, lifecycle, and tick pacing"
```

---

### Task 3: Renderer tier — `RenderBackpressure`, `RendererEntry`, `reset!`

**Files:**
- Modify: `src/async/sim.jl`
- Modify: `test/async/testutils.jl` (add `FakeRenderer`)
- Modify: `test/async/sim_test.jl`

**Interfaces:**
- Consumes: `AbstractRenderer`, `render!`, `reset!` (Phase 3);
  everything from Task 2.
- Produces: `abstract type RenderBackpressure end`, `struct DropLatest
  <: RenderBackpressure end`, `struct Blocking <: RenderBackpressure;
  buffer::Int end` (buffer lives on `Blocking` itself, not as a
  keyword on `RendererEntry`'s constructor — see the amendment below).
- Produces: `struct RendererEntry` with one constructor,
  `RendererEntry(renderer::AbstractRenderer, policy::RenderBackpressure)
  -> RendererEntry`, dispatching internally on `policy`'s concrete type
  to build the right channel kind.
- Produces: `AsyncSim`'s constructor gains
  `renderers::Vector{RendererEntry}=RendererEntry[]` keyword; `start!`
  now also enforces `nthreads() >= 2 + length(sim.renderers)`, spawns
  one `Task` per entry, and (see the amendment below) rebuilds every
  channel — input, output, and each renderer entry's — before spawning,
  since a prior `stop!` permanently closed them; `stop!` fetches the
  renderer tasks too.
- Produces: `reset!(sim::AsyncSim, track::Track,
  initial_states::Vector{State}) -> AsyncSim`.
- Produces (test-only): `FakeRenderer <: AbstractRenderer` in
  `test/async/testutils.jl` — records every `render!` call's frame and
  can be configured to throw on a specific call number, for Task 4's
  fault-injection test.

- [ ] **Step 1: Write the failing tests**

Modify `test/async/testutils.jl` to add `FakeRenderer` (append after
`_async_test_setup`):

```julia
mutable struct FakeRenderer <: F1Tenth.AbstractRenderer
    frames::Vector{Any}
    fail_on_call::Int
    calls::Int
end

FakeRenderer(; fail_on_call::Int=0) = FakeRenderer(Any[], fail_on_call, 0)

F1Tenth.init!(r::FakeRenderer, track::F1Tenth.Track) = r

function F1Tenth.render!(r::FakeRenderer, states::Vector{F1Tenth.State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    r.calls += 1
    r.fail_on_call != 0 && r.calls == r.fail_on_call && error("FakeRenderer: forced failure")
    push!(r.frames, (copy(states), copy(scans), copy(collisions)))
    return nothing
end

F1Tenth.close!(r::FakeRenderer) = nothing
F1Tenth.reset!(r::FakeRenderer, track::F1Tenth.Track) = (empty!(r.frames); r.calls = 0; r)
```

Append to `test/async/sim_test.jl`:

```julia
@testset "AsyncSim: start! enforces threads for physics + every renderer" begin
    setup = _async_test_setup()
    renderers = [F1Tenth.RendererEntry(FakeRenderer(), F1Tenth.DropLatest())
                 for _ in 1:(Threads.nthreads() + 3)]
    sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
        setup.states, setup.actions; renderers=renderers)
    @test_throws ErrorException F1Tenth.start!(sim)
end

if Threads.nthreads() >= 2
    @testset "AsyncSim: DropLatest renderer receives frames without blocking physics" begin
        setup = _async_test_setup()
        fake = FakeRenderer()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; renderers=[F1Tenth.RendererEntry(fake, F1Tenth.DropLatest())])

        F1Tenth.start!(sim)
        sleep(0.2)
        F1Tenth.stop!(sim)

        @test fake.calls > 0
    end

    @testset "AsyncSim: tick_hz paces every tick to the wall-clock rate (Blocking renderer)" begin
        setup = _async_test_setup()
        fake = FakeRenderer()
        hz = 20.0
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; tick_hz=hz,
            renderers=[F1Tenth.RendererEntry(fake, F1Tenth.Blocking())])

        F1Tenth.start!(sim)
        t0 = time()
        sleep(1.0)
        elapsed = time() - t0
        F1Tenth.stop!(sim)

        expected = hz * elapsed
        @test fake.calls > 0.5 * expected
        @test fake.calls < 1.5 * expected
    end

    @testset "AsyncSim: reset! requires the sim to be stopped, then clears state" begin
        setup = _async_test_setup()
        fake = FakeRenderer()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; renderers=[F1Tenth.RendererEntry(fake, F1Tenth.DropLatest())])

        F1Tenth.start!(sim)
        @test_throws ErrorException F1Tenth.reset!(sim, setup.track, setup.states)
        F1Tenth.stop!(sim)

        new_states = [F1Tenth.State(9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        F1Tenth.reset!(sim, setup.track, new_states)

        @test sim.states == new_states
        @test all(a -> a == zero(F1Tenth.Action), sim.last_actions)
        @test isempty(fake.frames)
    end

    @testset "AsyncSim: start! after stop!/reset! actually resumes ticking (regression)" begin
        setup = _async_test_setup()
        fake = FakeRenderer()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; renderers=[F1Tenth.RendererEntry(fake, F1Tenth.DropLatest())])

        F1Tenth.start!(sim)
        sleep(0.1)
        F1Tenth.stop!(sim)

        moving_states = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        F1Tenth.reset!(sim, setup.track, moving_states)

        F1Tenth.start!(sim)
        local frame = missing
        for _ in 1:200
            frame = F1Tenth.take_latest!(sim.output)
            frame === missing || break
            sleep(0.01)
        end
        F1Tenth.stop!(sim)

        @test frame !== missing
        fresh_states, _, _ = frame
        @test fresh_states[1][1] > moving_states[1][1]  # x advanced, i.e. physics genuinely resumed
    end
else
    @warn "Skipping AsyncSim renderer-tier tests: needs Threads.nthreads() >= 2; got $(Threads.nthreads())"
end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: FAIL — `F1Tenth.RendererEntry`/`F1Tenth.DropLatest`/
`F1Tenth.Blocking` not defined, and `AsyncSim`'s constructor has no
`renderers` keyword.

> **Amendment (post-Task-3 review):** The first version of this task
> had two problems, both caught by the implementer's own self-review
> before ever reaching the task reviewer:
>
> 1. **Restart after `stop!` was silently broken.** `stop!` closes
>    every channel via `_close_all_channels!`, and Julia `Channel`s
>    cannot reopen once closed. `reset!` requires the sim already
>    stopped and its documented workflow is "stop! → reset! → start!
>    again" — but the original `start!` reused the same
>    already-closed `Channel` objects on a restart. The physics tier's
>    very first tick after restart would hit `put!`/`take!` on a
>    closed channel, raise `InvalidStateException`, and since that
>    exception type is `_physics_loop`'s designated "clean shutdown,
>    do nothing" signal, the loop would just silently return —
>    `running[]` stuck `true`, no error surfaced, no further ticking,
>    forever, until the next explicit `stop!()`. Fix: `start!` now
>    rebuilds `sim.input`, `sim.output`, and every renderer entry's
>    channel from scratch before spawning any task — safe to do
>    unconditionally, including on the very first `start!` call, since
>    nothing is expected to have pre-seeded the input channels before
>    a sim has ever started (the physics tier always falls back to
>    `sim.last_actions`, which is unaffected by this and persists
>    across restarts, when an input channel has nothing new on it).
> 2. **Rebuilding a renderer's channel needs to preserve its
>    `Blocking` buffer size.** The original design put `buffer` on
>    `RendererEntry`'s constructor as a keyword consumed once at
>    construction and then forgotten — a restart that reconstructs the
>    entry from `(renderer, policy)` alone would silently reset any
>    custom buffer size back to the default. Fix: `buffer` now lives as
>    a field directly on `Blocking` itself, so it survives being
>    read back out of `entry.policy` on every restart.
>
> Both are implementation-detail fixes within this task's own scope —
> neither changes the public behavior described in the spec's Decision
> 4 (per-renderer backpressure choice) or Decision 7 (`reset!`'s
> signature and "stop, reset, start again" workflow); they make that
> workflow actually work. Code below reflects the fix; ship this one.

- [ ] **Step 3: Extend `sim.jl`**

Modify `src/async/sim.jl`: insert the following block immediately
after the `const RenderFrame = ...` line and before `mutable struct
AsyncSim`:

```julia
abstract type RenderBackpressure end
struct DropLatest <: RenderBackpressure end
struct Blocking <: RenderBackpressure
    buffer::Int
end
Blocking(; buffer::Int=8) = Blocking(buffer)

struct RendererEntry
    renderer::AbstractRenderer
    policy::RenderBackpressure
    channel::Union{LatestChannel{RenderFrame},Channel{RenderFrame}}
end

_render_channel(::DropLatest) = LatestChannel{RenderFrame}()
_render_channel(policy::Blocking) = Channel{RenderFrame}(policy.buffer)

RendererEntry(renderer::AbstractRenderer, policy::RenderBackpressure) =
    RendererEntry(renderer, policy, _render_channel(policy))
```

Replace the `mutable struct AsyncSim ... end` block with:

```julia
mutable struct AsyncSim
    bvh::TrackBVH
    params::VehicleParams
    car::CarFootprint
    dt::Float64
    cfg::LiDARConfig
    tick_hz::Union{Float64,Nothing}
    renderers::Vector{RendererEntry}
    input::Vector{LatestChannel{Action}}
    last_actions::Vector{Action}
    states::Vector{State}
    output::LatestChannel{RenderFrame}
    running::Atomic{Bool}
    physics_task::Union{Task,Nothing}
    renderer_tasks::Vector{Task}
    error::Ref{Union{Nothing,Exception}}
    error_lock::ReentrantLock
    rng::AbstractRNG
end
```

Replace the `AsyncSim(...)` outer constructor with:

```julia
function AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64,
    cfg::LiDARConfig, initial_states::Vector{State}, initial_actions::Vector{Action};
    tick_hz::Union{Float64,Nothing}=nothing, renderers::Vector{RendererEntry}=RendererEntry[],
    rng::AbstractRNG=default_rng())
    n = length(initial_states)
    length(initial_actions) == n || throw(ArgumentError(
        "initial_actions must have one entry per agent (got $(length(initial_actions)) for $n agents)"))
    input = [LatestChannel{Action}() for _ in 1:n]
    return AsyncSim(bvh, params, car, dt, cfg, tick_hz, renderers, input, copy(initial_actions),
        copy(initial_states), LatestChannel{RenderFrame}(), Atomic{Bool}(false),
        nothing, Task[], Ref{Union{Nothing,Exception}}(nothing), ReentrantLock(), rng)
end
```

Replace `_required_threads`:

```julia
_required_threads(sim::AsyncSim) = 2 + length(sim.renderers)
```

Replace `_close_all_channels!`:

```julia
function _close_all_channels!(sim::AsyncSim)
    for lc in sim.input
        close(lc.ch)
    end
    close(sim.output.ch)
    for entry in sim.renderers
        close(entry.channel isa LatestChannel ? entry.channel.ch : entry.channel)
    end
    return nothing
end
```

Replace `_physics_tick!`:

```julia
function _physics_tick!(sim::AsyncSim)
    n = length(sim.states)
    actions = Vector{Action}(undef, n)
    for i in 1:n
        a = take_latest!(sim.input[i])
        a === missing || (sim.last_actions[i] = a)
        actions[i] = sim.last_actions[i]
    end
    new_states, scans, collisions = step_multi_agent!(sim.states, actions, sim.bvh,
        sim.params, sim.car, sim.dt, sim.cfg; rng=sim.rng)
    sim.states = new_states
    frame = (new_states, scans, collisions)
    put_latest!(sim.output, frame)
    for entry in sim.renderers
        if entry.channel isa LatestChannel
            put_latest!(entry.channel, frame)
        else
            put!(entry.channel, frame)
        end
    end
    return nothing
end
```

Add, right after `_physics_loop`:

```julia
function _renderer_loop(sim::AsyncSim, entry::RendererEntry)
    try
        while true
            frame = entry.channel isa LatestChannel ? wait_latest!(entry.channel) : take!(entry.channel)
            render!(entry.renderer, frame...)
        end
    catch e
        if !(e isa InvalidStateException)
            _record_error!(sim, e)
            sim.running[] = false
            _close_all_channels!(sim)
            rethrow()
        end
    end
    return nothing
end
```

Replace `start!`:

```julia
function start!(sim::AsyncSim)
    sim.running[] && error("AsyncSim: already running")
    required = _required_threads(sim)
    nthreads() >= required || error(
        "AsyncSim: needs Threads.nthreads() >= $required (1 physics thread + " *
        "$(length(sim.renderers)) renderer thread(s)); got $(nthreads()). " *
        "Start Julia with `julia --threads=$required` or set JULIA_NUM_THREADS=$required.")
    sim.error[] = nothing
    # A prior stop! permanently closed every channel (Julia Channels
    # cannot reopen) -- rebuild them here so start! works both the
    # first time and after a stop!/reset!/start! cycle. Harmless to do
    # unconditionally on the very first call too: nothing is expected
    # to have queued input before a sim has ever started.
    sim.input = [LatestChannel{Action}() for _ in 1:length(sim.states)]
    sim.output = LatestChannel{RenderFrame}()
    sim.renderers = [RendererEntry(entry.renderer, entry.policy) for entry in sim.renderers]
    sim.running[] = true
    empty!(sim.renderer_tasks)
    for entry in sim.renderers
        push!(sim.renderer_tasks, @spawn _renderer_loop(sim, entry))
    end
    sim.physics_task = @spawn _physics_loop(sim)
    return sim
end
```

Replace `stop!`:

```julia
function stop!(sim::AsyncSim)
    sim.running[] = false
    _close_all_channels!(sim)
    tasks = sim.physics_task === nothing ? sim.renderer_tasks : [sim.physics_task; sim.renderer_tasks]
    for t in tasks
        try
            fetch(t)
        catch
            # real cause, if any, is captured in sim.error and rethrown below
        end
    end
    sim.physics_task = nothing
    empty!(sim.renderer_tasks)
    if sim.error[] !== nothing
        err = sim.error[]
        sim.error[] = nothing
        throw(err)
    end
    return sim
end
```

Add, at the end of the file:

```julia
function reset!(sim::AsyncSim, track::Track, initial_states::Vector{State})
    sim.running[] && error("AsyncSim: reset! requires the sim to be stopped first (call stop! first)")
    n = length(sim.last_actions)
    length(initial_states) == n || throw(ArgumentError(
        "reset! requires the same number of agents ($n); got $(length(initial_states))"))
    for entry in sim.renderers
        reset!(entry.renderer, track)
    end
    sim.states = copy(initial_states)
    fill!(sim.last_actions, zero(Action))
    take_latest!(sim.output)
    for lc in sim.input
        take_latest!(lc)
    end
    sim.error[] = nothing
    return sim
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS (with the same single-thread skip behavior as Task 2
for the guarded testsets; the thread-guard test itself always runs,
since it deliberately requests more renderers than any environment
has threads).

- [ ] **Step 5: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add src/async/sim.jl test/async/testutils.jl test/async/sim_test.jl
git commit -m "Add AsyncSim renderer tier, backpressure policies, and reset!"
```

---

### Task 4: Fail-fast fault injection, module exports, CI thread bump

**Files:**
- Modify: `test/async/sim_test.jl`
- Modify: `src/F1Tenth.jl`
- Modify: `test/exports_test.jl`
- Modify: `.github/workflows/CI.yml`

**Interfaces:**
- Consumes: everything produced in Tasks 1-3.
- Produces: public names exported from `F1Tenth` in addition to Phase
  1-3's exports: `LatestChannel`, `put_latest!`, `take_latest!`,
  `wait_latest!`, `AsyncSim`, `start!`, `stop!`, `RenderBackpressure`,
  `DropLatest`, `Blocking`, `RendererEntry`. (`reset!` is already
  exported from Phase 3 — Task 3 only added a new method to the same
  generic function, not a new name.)

- [ ] **Step 1: Write the failing fault-injection tests**

Append to `test/async/sim_test.jl`:

```julia
if Threads.nthreads() >= 2
    @testset "AsyncSim: a physics-tier exception propagates through stop! and halts renderers too" begin
        setup = _async_test_setup()
        fake = FakeRenderer()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; renderers=[F1Tenth.RendererEntry(fake, F1Tenth.DropLatest())])

        F1Tenth.start!(sim)
        sleep(0.05)
        pop!(sim.last_actions)  # corrupt internal state: fewer cached actions than agents
        sleep(0.05)

        @test_throws BoundsError F1Tenth.stop!(sim)
        @test sim.running[] == false
        @test sim.physics_task === nothing
        @test isempty(sim.renderer_tasks)
    end

    @testset "AsyncSim: a renderer-tier exception propagates through stop! and halts physics too" begin
        setup = _async_test_setup()
        fake = FakeRenderer(; fail_on_call=2)
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; renderers=[F1Tenth.RendererEntry(fake, F1Tenth.Blocking())])

        F1Tenth.start!(sim)
        sleep(0.3)

        @test_throws ErrorException F1Tenth.stop!(sim)
        @test sim.running[] == false
        @test sim.physics_task === nothing
        @test isempty(sim.renderer_tasks)
    end
else
    @warn "Skipping AsyncSim fault-injection tests: needs Threads.nthreads() >= 2; got $(Threads.nthreads())"
end
```

- [ ] **Step 2: Run tests to confirm the fail-fast machinery already built in Tasks 2-3 actually works end-to-end**

This step is a validation run, not a red/green TDD step — the
fail-fast machinery (`_record_error!`, the `catch` blocks in
`_physics_loop`/`_renderer_loop`, `stop!`'s rethrow) was already
implemented in Tasks 2-3 for `AsyncSim` to work at all; these two
tests are the first ones to actually exercise a real (non-shutdown)
exception on each tier.

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && JULIA_NUM_THREADS=4 julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS. If either new testset fails, that's a real bug in
Task 2/3's error-handling code (most likely culprit: `stop!` clearing
`sim.physics_task`/`sim.renderer_tasks` in the wrong order relative to
reading `sim.error[]`, or `_close_all_channels!` not being called from
both tiers' `catch` blocks) — fix it in `src/async/sim.jl` before
proceeding, rather than adjusting the test to match broken behavior.

- [ ] **Step 3: Add exports**

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

- [ ] **Step 4: Extend `exports_test.jl`**

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
    ]
    for name in exported_names
        @test isdefined(F1Tenth, name)
    end
end
```

- [ ] **Step 5: Bump CI thread count**

Modify `.github/workflows/CI.yml`: add a job-level `env:` block (after
`permissions:`, before `strategy:`) so every Julia process CI spawns
— build and test — has enough threads for Phase 4's `start!` guard and
its multi-renderer tests to actually run rather than skip:

```yaml
    permissions: # needed to allow julia-actions/cache to proactively delete old caches that it has created
      actions: write
      contents: read
    env:
      JULIA_NUM_THREADS: '4'
    strategy:
```

- [ ] **Step 6: Run the full test suite**

Run: `cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl && JULIA_NUM_THREADS=4 julia --project=. -e 'import Pkg; Pkg.test()'`
Expected: PASS — every testset from Tasks 1-4 runs for real (no
`@warn`-skips) with 4 threads locally, plus every Phase 1/2/3 testset
still passing unchanged.

- [ ] **Step 7: Commit**

```bash
cd /home/pganguli/Downloads/f1tenth/F1Tenth.jl
git add test/async/sim_test.jl src/F1Tenth.jl test/exports_test.jl .github/workflows/CI.yml
git commit -m "Add Phase 4 fault-injection tests, exports, and CI thread bump"
```
