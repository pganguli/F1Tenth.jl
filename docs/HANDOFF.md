# Handoff: F1Tenth.jl

Written for: a developer or agent picking up this package with no memory of
earlier sessions.

## What this is

A port of the Python `f1tenth_gym` (sibling checkout at `../f1tenth_gym`).
**The Python code is the gold standard:** this package has its features and
nothing more, and matches its inputs and outputs. Internals may differ
wherever that makes it faster, as long as no output changes.

Work that went beyond Python is kept on other branches: the earlier design
(ModelingToolkit dynamics, vector tracks, async sim, reachability, telemetry)
on `archive/extended-features`, and the Frenet/tck rewrite on `frenet-tck`.

## Layout

Files mirror the Python modules in `f110_gym/envs`:

| Julia | Python |
|---|---|
| `src/params.jl` | the `params` dict → `VehicleParams` |
| `src/dynamic_models.jl` | `dynamic_models.py` |
| `src/collision_models.jl` | `collision_models.py` |
| `src/laser_models.jl` | `laser_models.py` |
| `src/maps.jl` | map loading and distance transform (`ScanSimulator2D.set_map`, `get_dt`) |
| `src/lidar_tracing.jl` | the beam loop (`get_scan`, `trace_ray`), optimized |
| `src/base_classes.jl` | `base_classes.py`, plus `Observation` (the obs dict) |
| `src/f110_env.jl` | `f110_env.py` |
| `src/rendering.jl`, `ext/F1TenthGLMakieExt.jl` | `rendering.py` (pyglet → GLMakie, loaded by `using GLMakie`) |
| `examples/waypoint_follow.jl` | `examples/waypoint_follow.py` |
| `maps/` | `envs/maps/` |

## How parity is checked

`test/reference/generate.py` runs the real Python code (with stub `gym` and
`pyglet` modules in `test/reference/stubs/`) and freezes its outputs into
`test/reference/data/`. Rerun it if the reference changes:

```sh
uv venv refvenv && VIRTUAL_ENV=refvenv uv pip install numpy scipy numba pillow pyyaml
refvenv/bin/python test/reference/generate.py ../f1tenth_gym
```

It writes nothing into the Python checkout. The tests compare:

- dynamics, PID, constraints on 3000 random states: rtol 1e-12
- vertices, GJK, pairwise collisions: exact
- distance transforms of all 6 maps and random bitmaps: exact
- noise-free scans on every map plus a rotated-origin map, for both tracers:
  **bit-identical**
- car occlusion (`ray_cast!`): rtol 1e-12; iTTC: exact
- 4 whole rollouts replayed with Python's actions (two laps; two-car crash;
  Euler + LiDAR offset + wall crash; mid-run `update_params!`): within 1e-10
  (observed ≤ 4e-15); every collision, lap count, lap time and `done` exact
- the example's planner, closed loop: Python's exact two laps

Python's own unit-test checks (`dynamics_test`, collision tests, scan
`test_rng`, fps thresholds) are ported too. Noise is turned off in tests the
Python way, by setting a car's `scan_rng` to `nothing` (`rng=None`).

## Optimizations (none may change an output)

Run the scan tests after touching `lidar_tracing.jl` or `maps.jl`.

- **Interleaved tracing:** each beam is a chain of dependent table lookups,
  so 16 beams are kept in flight round-robin and their memory reads overlap.
- **SIMD tracing** (SIMD.jl): 16 beams per vector with masked gathers, the
  same IEEE operations per lane. `F1Tenth.TRACER[]` is `:auto`: both tracers
  are timed once per session (never at precompile time) and the faster is
  used; SIMD wins where gathers are fast (e.g. AVX-512). Masked-off lanes
  always get an in-bounds index.
- Unrotated maps (all bundled ones) skip the identity origin rotation, which
  is exact. Each scan's first lookup, shared by all beams, is done once.
- Maps are cached by path and mtime; the distance transform is an exact,
  threaded Felzenszwalb–Huttenlocher pass (as fast as scipy on one thread).
- Scan noise is drawn in one batch per scan (`randn!`).
- `step!` allocates nothing on one thread; observations are reused buffers.
- With several threads, cars are stepped in parallel (`THREAD_AGENTS[]`).
- Measured and rejected: splitting one scan's beams across threads (no gain
  at ~11 µs per scan), multiplying by a reciprocal instead of dividing
  (slower), a compact `UInt32` distance table (slower: scans are compute
  bound).

## Additions beyond Python

The one exception to "nothing more", requested by the project owner: the 23
tracks of `f1tenth/f1tenth_racetracks` load by name (`F1Tenth.RACETRACKS`).
That repo is GPL-3.0, so its maps are a lazy artifact (`Artifacts.toml`,
pinned to commit b95c4ef), downloaded on first use and never bundled.

## Intentional differences from Python

- 1-based indices: `ego_idx`, `update_params!(…; index)`, `collision_idx`
  (−1 still means none).
- LiDAR noise is N(0, 0.01) per beam, from Julia's Xoshiro instead of numpy,
  so the stream differs. Each car restarts it from `seed` on reset, as in
  Python (all cars draw the same sequence).
- Inputs that crash Python work: `map="vegas"` by name (Python looks for
  `vegas.yaml` in the working directory), `"levine"` with the default
  extension (its image is `.pgm`), `map_ext` without the dot (as Python's
  docstring writes it), and rendering with a built-in or default map.
- No process-wide state: Python's `RaceCar` keeps the scan simulator and beam
  tables as class attributes, so all envs in a process share one map and the
  first car's params; its renderer is also shared. Here each `Simulator` and
  env owns its own (cars of one simulator share them, as in Python).
- `obs` and `info` are reused buffers (Python builds new lists every step).
- Rendering is GLMakie: the map is one texture of the pixels Python draws as
  GL points; camera, zoom/pan math, colors, car shapes, label and FPS readout
  follow Python. Python callbacks add to `batch`; Julia ones draw on `axis`.

Kept although odd, because Python does it: `obs.ego_idx` is always 1 (the env
never passes `ego_idx` to its simulator); a wall hit (iTTC) zeroes velocity
**and yaw**, yaw rate and slip; car-to-car collisions keep the body size given
at construction after `update_params!`; off-map points read the distance
table's last cell; the window's label is never updated; `update_map!` doesn't
redraw an open window; RK4 prints a one-time warning; `reset!` takes one
zero-input step and returns its result.

## Not ported

`unittest/random_trackgen.py` (a standalone map generator using OpenCV and
shapely), the pyglet test scripts, `f110_env_backup.py`, gym registration
(`gym.make`; use `F110Env(; kwargs...)`), the Sphinx docs and the Dockerfile.

## Gotchas

- `Pkg.test()` forces bounds checks, which move the SIMD tracer's scratch
  vectors to the heap (64 bytes per step); the allocation test allows that
  only in that mode.
- Makie rejects StaticArrays vectors of points; comprehensions over an
  `SVector` return a `SizedVector`, so build plain `Vector{Point2f}`s.
- The render tests need a display (CI starts Xvfb) and wait for the window
  manager to finish resizing the window before checking the camera.
