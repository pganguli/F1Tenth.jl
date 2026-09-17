# Phase 4 — Async Concurrency & Control Channels: Design

## Context

Phase 1 (`src/dynamics/`) provides `State`/`Action` and `step_dynamics`.
Phase 2 (`src/geometry/`) adds `step_multi_agent!(states, actions, bvh,
params, car, dt, cfg) -> (new_states, scans, collisions)` — a stateless,
synchronous per-tick step over all agents. Phase 3 (`src/render/`) adds
`AbstractRenderer`/`init!`/`render!`/`close!`/`reset!`, called
synchronously, once per tick, by whatever loop calls it — Phase 3's
own Non-goals section explicitly deferred loop design to Phase 4.

Today nothing in the package actually loops. A caller wanting a live
sim writes their own `while` loop calling `step_multi_agent!` then
`render!` in lockstep, on one thread, blocking on rendering every
tick. Phase 4 replaces that with an owned async subsystem: a
multi-threaded sim loop connected to the outside world (actions in,
observations and renders out) via `Channel`s, so physics can run
decoupled from a live-drawing renderer and from whatever external code
supplies control input.

## Non-goals

- No changes to any file under `src/dynamics/`, `src/geometry/`, or
  `src/render/`. `step_multi_agent!` and `render!` are called
  unmodified; Phase 4 only adds orchestration around them.
- No literal third "input" thread. External code (a policy, a human
  input handler, an RL training loop) writes directly into per-agent
  input channels from whatever thread it already runs on — see
  Decision 2 for why a dedicated input-processing task was rejected as
  unnecessary machinery.
- No network/IPC transport for actions (sockets, ROS, shared memory
  across processes). Input is an in-process `Channel`; anything
  bridging that to an external transport is the caller's problem.
- No Phase 5 (reachability/telemetry) integration. The output channel
  Phase 4 exposes is a natural place for a future `Arrow.jl` telemetry
  sink to attach, but wiring that up is out of scope here.
- No automatic episode termination on collision. Collisions flow out
  through the output channel like any other data; the loop keeps
  ticking regardless, matching `step_multi_agent!`'s existing
  behavior. The caller decides whether/when to stop or `reset!`.
- No numeric-fidelity claim to `../f1tenth_gym`'s threading/stepping
  model — this is a from-scratch design against this package's actual
  Phase 1-3 shapes, not a port.

## Decisions

### 1. `LatestChannel{T}`: the one primitive both input and (default) render backpressure share

```julia
struct LatestChannel{T}
    ch::Channel{T}   # capacity 1
end

put_latest!(lc::LatestChannel{T}, v::T) where {T}   # never blocks the producer
take_latest!(lc::LatestChannel{T}) where {T}        # -> T or missing, never blocks
```

`put_latest!` does a non-blocking `take!` to drain any stale value (if
present) before `put!`ing the new one — so a producer that outruns its
consumer always overwrites rather than blocking or queuing. This one
type implements both "reuse last action" (Decision 3) and "drop
oldest, keep latest" render backpressure (Decision 4) — it is the only
channel primitive Phase 4 defines; no separate bespoke mechanism for
input vs. render.

### 2. No dedicated input-tier thread

The blueprint names a 3-tier pipeline (input / physics / render).
Given actions are pushed by external code directly into per-agent
`LatestChannel`s (a training loop calling `put_latest!` from its own
thread, a human-input handler doing the same), a dedicated
input-processing `Task` would just be a pass-through hop with no
behavioral benefit — it would add a stop-condition to manage for
nothing gained. Rejected as unnecessary machinery (YAGNI). The "input
tier" is structural (a `Vector` of per-agent input channels that live
on `AsyncSim`), not a literal third thread; the two owned threads are
physics and (per-renderer) render.

### 3. Physics tier: cached last-action, unmodified `step_multi_agent!`

`AsyncSim` holds one `LatestChannel{Action}` per agent and a
`Vector{Action}` of cached last-actions, seeded at `start!` (see
Decision 6 — every agent must have an initial action before the loop
can start). Each tick:

1. For each agent, `take_latest!` its input channel; on `missing`,
   reuse the cached last action; on a value, use it and update the
   cache. This is a zero-order hold, so one silent/slow agent never
   stalls the tick for others.
2. Call `step_multi_agent!(states, actions, bvh, params, car, dt, cfg)`
   unmodified.
3. `put_latest!` `(new_states, scans, collisions)` to the output
   channel (Decision 5).
4. For each attached renderer entry, forward the same tuple per
   Decision 4's per-renderer policy.

### 4. Render tier: `Vector` of independent renderer entries, per-entry backpressure policy

```julia
abstract type RenderBackpressure end
struct DropLatest <: RenderBackpressure end   # default; wraps a LatestChannel
struct Blocking   <: RenderBackpressure end   # wraps a plain Channel(N), put! blocks

struct RendererEntry
    renderer::AbstractRenderer
    policy::RenderBackpressure
    channel::Union{LatestChannel{RenderFrame}, Channel{RenderFrame}}
end
```

Each `RendererEntry` gets its own `Threads.@spawn` `Task` that loops
`take!`ing its channel and calling the existing `render!` unmodified.
`DropLatest` (the default) is for live/interactive renderers
(`GLMakieRenderer`, or a `CairoMakieRenderer` used only for its final
summary SVG) — physics never blocks on them. `Blocking` is for a
`CairoMakieRenderer` constructed with `output_dir` set (per-tick
flip-book recording, where every tick must be captured) — physics
*will* stall if that renderer falls behind, which is the accepted
trade-off for that use case; a caller who wants flip-book recording
without throttling physics should use a separate, later, synchronous
replay pass instead of attaching it live. Renderer entries are
supplied at `AsyncSim` construction (`renderers::Vector{RendererEntry}`,
possibly empty for a headless training run with no rendering at all).

### 5. Output channel: async, symmetric with input

`AsyncSim` holds one `output::LatestChannel{RenderFrame}` (where
`RenderFrame = Tuple{Vector{State}, Vector{Vector{Float64}},
Vector{Bool}}`, i.e. exactly `step_multi_agent!`'s return shape).
External consumers (a training loop's observation side) call
`take_latest!(sim.output)`. This mirrors the input side's push model
rather than offering a blocking `step!(sim, actions) -> observations`
call, which would reintroduce the synchronous coupling the async
pipeline exists to remove.

### 6. Tick pacing: one loop, two modes

`AsyncSim` is constructed with `tick_hz::Union{Float64,Nothing}`.
`tick_hz === nothing` means throughput mode: the physics tier loops
steps 1-4 above with no sleep, as fast as the CPU allows (training use
case). A numeric `tick_hz` means real-time mode: the physics tier
computes each tick's wall-clock deadline (`next_deadline = start +
n/tick_hz`) and sleeps to it before starting the next tick, so a slow
tick's overrun doesn't accumulate drift across the run — sleep is
`max(0, next_deadline - time())`, and an overrun tick just runs the
following tick immediately with no catch-up loop. Both modes share
every other component; this is not two separate implementations.

### 7. Lifecycle: `AsyncSim`, `start!`, `stop!`, `reset!`

```julia
mutable struct AsyncSim
    # config: bvh, params, car, dt, cfg, tick_hz, renderers::Vector{RendererEntry}
    # state: input::Vector{LatestChannel{Action}}, last_actions::Vector{Action},
    #        output::LatestChannel{RenderFrame}, states::Vector{State},
    #        physics_task::Union{Task,Nothing}, running::Threads.Atomic{Bool}
end

start!(sim::AsyncSim) -> sim   # requires last_actions fully seeded; spawns physics + renderer Tasks
stop!(sim::AsyncSim) -> sim    # flips running to false, closes channels, fetch()es every Task
reset!(sim::AsyncSim, track::Track, initial_states::Vector{State}) -> sim
```

`start!` asserts `Threads.nthreads() >= 2 + length(sim.renderers)`
(physics + one thread per renderer) before spawning anything, erroring
with a clear message naming the required `JULIA_NUM_THREADS` value
rather than silently degrading — the same "fail loud on a missing
prerequisite" precedent as Phase 1's `Manifest.toml`-not-instantiated
check. Each tier's `Threads.@spawn`'d `Task` loops on
`sim.running[]`; `stop!` sets it to `false` and closes every channel
so any `take!`/`put!` blocked inside a tier unblocks (via
`InvalidStateException`, caught inside the tier's loop as the normal
exit signal) rather than hanging. `stop!` then `fetch`es every `Task`,
which is also where a real (non-shutdown) exception surfaces — see
Decision 8.

`reset!` requires the sim to already be stopped (errors otherwise). It
calls each renderer's existing `reset!(renderer, track)`, replaces
`sim.states` with `initial_states`, clears `last_actions` (the caller
must seed fresh initial actions before the next `start!`, same as a
brand-new `AsyncSim`), and drains any stale value left in the output
channel. The caller calls `start!` again to resume.

### 8. Error handling: fail-fast across all tiers

Any tier's `Task` — physics or any renderer — that throws sets
`running[]` to `false` and closes every channel itself (not just its
own) before rethrowing, so every other tier observes the same
shutdown signal on its next blocking channel operation and exits
promptly rather than continuing to run against a now-inconsistent sim.
`stop!`'s `fetch` on each `Task` surfaces the original exception
(unwrapped from Julia's `TaskFailedException` for a clear message)
to the caller — a crash is loud, not silently swallowed. This applies
uniformly; there is no "renderer errors are best-effort" carve-out —
a broken renderer stops the whole `AsyncSim`, same as a broken
physics tick.

## Testing

- `LatestChannel` unit tests: `put_latest!` never blocks and overwrites
  a pending value; `take_latest!` returns `missing` on empty and drains
  exactly once.
- Full lifecycle test using a fake zero-delay `AbstractRenderer` and
  pre-seeded input actions (no real wall-clock waiting): `start!` →
  several ticks → assert output channel reflects `step_multi_agent!`
  applied correctly → `stop!` → assert all `Task`s finished cleanly.
- `Threads.nthreads()` guard test: asserts `start!` throws the documented
  error under insufficient threads; only meaningfully exercised when CI
  runs with `JULIA_NUM_THREADS >= 3` (see below), so this test is
  written to pass or be explicitly skipped/marked depending on the
  actual thread count at test time, rather than assuming one.
- Fault-injection test: a physics tier forced to throw (e.g. a
  deliberately-invalid `dt`) propagates through `stop!`/`fetch` as the
  original exception, and a subsequently-attached renderer `Task` is
  confirmed to have exited too (not left hanging).
- Real-time pacing is tested by asserting elapsed wall-clock time for
  N ticks falls within a tolerance band around `N/tick_hz`, not by
  asserting exact timing (inherently flaky otherwise).

## Required CI change

`.github/workflows/CI.yml` currently runs Julia's default (single
thread). Phase 4's minimum-thread `start!` guard means any test that
actually spawns an `AsyncSim` with a renderer needs
`JULIA_NUM_THREADS >= 3` set in CI, or those tests will only ever
exercise the guard's error path. This is a required CI change
alongside the code, not an optional follow-up.

## Open Questions

- Phase 3's `GLMakieRenderer`/`CairoMakieRenderer` being unconditional
  regular dependencies (Phase 3 load-bearing fact 1) means every
  `AsyncSim` renderer `Task` pays their load cost regardless of which
  renderer is actually attached — unchanged by Phase 4, just inherited.
- Whether `AsyncSim` itself should be `Threads.@spawn`-safe to
  construct/inspect from multiple external threads concurrently (e.g.
  two training-loop workers sharing one sim) is unaddressed; the
  design above assumes one external "owner" thread drives
  `start!`/`stop!`/`reset!` and other threads only ever `put_latest!`/
  `take_latest!` on the channels themselves.
