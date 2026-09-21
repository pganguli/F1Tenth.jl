# Frenet-Frame Simulator & tck Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the F1Tenth.jl simulator to Frenet coordinates `(s, d, δ, v, μ, ψ̇, β)` with maps stored as tck B-splines (CSV produced by `tck.py`), replacing the polyline `Track`/KD-tree geometry.

**Architecture:** A `Track` wraps a periodic cubic B-spline over `[x, y, w]` plus an arc-length table, and exposes O(1)-ish Frenet queries (`track_point`, `to_frenet`, `to_cartesian`, `world_pose`). Vehicle dynamics are integrated in curvilinear form by reusing the validated Cartesian body dynamics. Walls are the implicit boundary `|d| ≤ w(s)`: collision samples footprint points, LiDAR sphere-traces the boundary. Reachability is rewritten over a 10-D Frenet state with curvature as a boxed constant dimension.

**Tech Stack:** Julia ≥ 1.10 (`StaticArrays`, `ReachabilityAnalysis`, `LazySets`, `GLMakie`/`CairoMakie`, `Arrow`); Python (`numpy`, `scipy`) only for fixture generation and `tck.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-frenet-tck-design.md` (read it first, including its "Implementation-time amendments" section, which lists where this plan departs from the original text and why; the same list is repeated below).

## Global Constraints

- **Language/spelling:** en-US in all code, comments, docs and commit messages (never en-GB). No attribution lines in commit messages.
- **Julia:** `julia = "1.10.10"` compat floor stays. **No new package dependencies.** `NearestNeighbors` is removed (Task 7). `DelimitedFiles` stays (tests use it).
- **Python:** `source .venv/bin/activate` before any `python` command (it has numpy/scipy; it does *not* have `cv2`/`matplotlib`, which only `tck.py` itself needs).
- **Run tests from `F1Tenth.jl/`** with threads: `julia --project=. -t 4 -e '...'` (the async tests need ≥ 3 threads to run rather than skip).
- **Notation:** physical quantities use Unicode identifiers (`κ`, `δ`, `ψ̇`, `μ`, `β`, `𝐱`); every new symbol gets a one-line comment where first declared (per `docs/HANDOFF.md`).
- **Conventions (verbatim from the spec):** `+d` is left of travel; `w_left(s) = w_right(s) = w(s)`; the loop is counterclockwise; `s` is unwrapped and increasing (a lap is `⌊s/L⌋`), wrapped only inside track queries; `μ = ψ − ψ_ref(s)` wrapped to `(−π, π]`; state layout keeps 7 slots `(s, d, δ, v, μ, ψ̇, β)`; CSV columns `t,cx,cy,cw`, degree `k = 3`, periodic on `u ∈ [0, 1]`.
- **Load-time validation:** `Track` throws `ArgumentError` when `w ≤ 0`, when the curve does not close, or when `|κ|·w ≥ 0.9` anywhere (the user chose strict rejection over clipping).
- **Benchmark gate:** a 1080-beam scan must take ≤ 5 ms single-thread (measured 3.3 ms on the lumpy fixture).
- **Execution environment:** run in a git worktree (`.worktrees/` is git-ignored) created via `superpowers:using-git-worktrees`. The full suite is *expected to be red* between Task 3 and Task 9 (old-API tests still exist); each task therefore runs only its own test files. The suite goes green at Task 10.

## Spec amendments (found while building and verifying the design)

1. **CSV padding.** SciPy's *periodic* `splprep` returns coefficient arrays of length `len(t) − k − 1` (not padded to `len(t)` as the spec said). `save_tck_csv` zero-pads them to `len(t)` so the file has one row per knot; `load_tck_csv` ignores the padding.
2. **`step_dynamics` naming.** The Cartesian RK4 step is renamed `step_dynamics_cartesian` (kept: it is the reference model and what the Python-gym parity tests exercise). The Frenet `step_dynamics(𝐱, action, params, track, dt)` takes the public name.
3. **No Frenet pre-filter for car-car collisions.** The spec's `s`-gap/`|Δd|` pre-filter is dropped: it is not provably safe on curves when `|d| ≠ 0`, and n is small. Car-car stays SAT in the world frame. `frenet_gap` is still provided for gap analysis.
4. **`to_frenet` never throws.** If Newton fails from the hint it retries from a table-scan seed, then accepts the last iterate. A crashed car far off-track must not take the sim thread down. (The spec's "throws if it fails" is dropped.)
5. **Validation is pointwise.** `max_s |κ(s)|·w(s) < 0.9`, not `max|κ|·max w`; the error message reports the worst location (`s`, world x/y, turn radius, half-width) and the remedy. The Lipschitz constant and reachability window scale use the same pointwise value (`Track.κw_max`).
6. **No knot-span hint.** de Boor uses `searchsortedlast` (binary search); measured cost already meets the gate.
7. **Fold-over warning not implemented.** The spec's load-time warning for tracks whose sections come closer than a corridor width is left out (no target map needs it yet); it is listed under known deferred minors. On such a track `to_frenet` from a bad hint could lock onto the wrong branch off-track.
8. **Real-map finding (Austin).** `Austin_map.png` converted with the current `tck.py` (`smoothing=0.001`) has `|κ|·w` up to 1.42 (radius 0.46 m vs half-width 0.68 m) over ~1% of the lap, so it is **rejected** by the strict validation. Raising `smoothing` does not reliably fix it (best measured: 0.94 at `smoothing=0.03`, moving the centerline up to 1 m). The user must re-fit that map (or accept editing the raster); the demo therefore defaults to the lumpy test track until an `Austin_tck.csv` that passes exists.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/geometry/tck.jl` | `BSpline{P}` (de Boor eval, derivative), `load_tck_csv` | Create (Task 2) |
| `src/geometry/track.jl` | `Segment`, `TrackPoint`, `Track` (arc-length table, validation), queries, `to_frenet`/`to_cartesian`/`world_pose`, `track_boundaries` | Rewrite (Task 3) |
| `src/dynamics/frenet.jl` | `frenet_rhs`, Frenet `step_dynamics` | Create (Task 4) |
| `src/dynamics/step.jl` | Rename Cartesian step to `step_dynamics_cartesian` | Modify (Task 4) |
| `src/geometry/raycast.jl` | `LiDARConfig`, `CarFootprint`, `car_vertices(pose, car)`, sphere-traced `cast_ray`, `scan` | Rewrite (Task 5) |
| `src/geometry/collision.jl` | SAT, `check_car_collisions`, Frenet `check_track_collision` | Rewrite (Task 6) |
| `src/geometry/step2.jl` | `step_multi_agent!` over a `Track` | Rewrite (Task 7) |
| `src/geometry/bvh.jl` | (KD-tree of wall segments) | Delete (Task 7) |
| `src/async/sim.jl` | `AsyncSim` holds a `Track` | Modify (Task 7) |
| `src/render/*.jl`, `src/telemetry/sink.jl`, `src/geometry/lazysets_export.jl` | World-frame drawing, Frenet telemetry doc, corridor polygons | Modify (Task 8) |
| `src/reachability/*.jl` | 10-D Frenet mirrors, κ window, safety check | Modify (Task 9) |
| `test/fixtures/*` | tck fixtures + generator | Create (Task 1) |
| `benchmarks/lidar_scan.jl` | 5 ms LiDAR gate | Create (Task 5) |
| `../main.jl`, `../tck.py`, `docs/HANDOFF.md` | Demo, CSV writer, handoff | Modify (Tasks 1, 10) |

Test-running helper used throughout (from `F1Tenth.jl/`):

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/tck_test.jl")'
```

---

### Task 1: Fixtures and `tck.py` CSV writer

**Files:**
- Create: `test/fixtures/make_tck_fixtures.py`, `test/fixtures/fixtures.jl`, generated `test/fixtures/{circle_tck.csv,lumpy_tck.csv,lumpy_ref.csv}`
- Modify: `../tck.py` (outside the git repo: add `save_tck_csv`, call it from `__main__`)

**Interfaces:**
- Produces: `fixture_path(name)`, `circle_track()`, `lumpy_track()` (Julia, in `test/fixtures/fixtures.jl`); CSV format `# k=3` / `t,cx,cy,cw` / one row per knot; `lumpy_ref.csv` columns `u,x,y,w,dx,dy,dw,ddx,ddy` (scipy `splev` values of the lumpy fit).
- Fixture facts later tasks rely on: `circle_tck.csv` is a cubic *interpolant* of 200 samples of an R = 5 m circle, `w = 1.0`, CCW, `s = 0` at world `(5, 0)` heading `+y` (so curvature/heading are accurate only to ~4e-5, hence `atol=1e-4` in those tests). `lumpy_tck.csv` is `main.jl`'s racetrack (300 samples), `w = 1.1 + 0.15·sin(2t)`, length ≈ 115.8 m, `κ ∈ [0.018, 0.279]` (convex CCW loop).

- [ ] **Step 1: Create the fixture generator**

`test/fixtures/make_tck_fixtures.py`:

```python
"""Generate tck CSV fixtures (and scipy reference values) for the Julia tests.

Run from the repo root:  python3 test/fixtures/make_tck_fixtures.py
Writes next to this script: circle_tck.csv, lumpy_tck.csv, lumpy_ref.csv.
"""
import os
import numpy as np
from scipy.interpolate import splprep, splev

HERE = os.path.dirname(os.path.abspath(__file__))


def save_tck_csv(path, tck):
    """Same layout as tck.py:save_tck_csv (kept in sync by hand)."""
    t, c, k = tck
    t = np.asarray(t, dtype=float)
    cols = []
    for ci in c:
        ci = np.asarray(ci, dtype=float)
        pad = np.zeros(len(t))
        pad[: len(ci)] = ci
        cols.append(pad)
    with open(path, "w") as f:
        f.write(f"# k={int(k)}\n")
        f.write("t,cx,cy,cw\n")
        for row in zip(t, *cols):
            f.write(",".join(repr(float(v)) for v in row) + "\n")


def racetrack_curve(t):
    """Same lumpy CCW loop as main.jl."""
    x = 24 * np.cos(t) + 5 * np.cos(2 * t)
    y = 11 * np.sin(t) + 2 * np.sin(2 * t) - 0.8 * np.sin(3 * t)
    return x, y


def main():
    # 1. Circle: R = 5, half-width 1.0.
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    R = 5.0
    tck, _ = splprep([R * np.cos(th), R * np.sin(th), np.full_like(th, 1.0)], s=0, per=True)
    save_tck_csv(os.path.join(HERE, "circle_tck.csv"), tck)

    # 2. Lumpy racetrack with a varying half-width.
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    x, y = racetrack_curve(th)
    w = 1.1 + 0.15 * np.sin(2 * th)
    tck, _ = splprep([x, y, w], s=0, per=True)
    save_tck_csv(os.path.join(HERE, "lumpy_tck.csv"), tck)

    # 3. Reference values from scipy at fixed u.
    u = np.linspace(0.0, 1.0, 57, endpoint=False) + 0.003
    px, py, pw = splev(u, tck, der=0)
    dx, dy, dw = splev(u, tck, der=1)
    ddx, ddy, ddw = splev(u, tck, der=2)
    with open(os.path.join(HERE, "lumpy_ref.csv"), "w") as f:
        f.write("u,x,y,w,dx,dy,dw,ddx,ddy\n")
        for row in zip(u, px, py, pw, dx, dy, dw, ddx, ddy):
            f.write(",".join(repr(float(v)) for v in row) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the Julia fixtures helper**

`test/fixtures/fixtures.jl`:

```julia
# Shared test fixtures: tck CSVs generated by make_tck_fixtures.py.
#
#   circle_tck.csv  R = 5 m circle, half-width 1.0 (CCW; s = 0 at world (5, 0))
#   lumpy_tck.csv   the racetrack from main.jl, half-width 1.1 + 0.15·sin(2t)
#   lumpy_ref.csv   scipy splev reference values for lumpy_tck.csv
using F1Tenth

fixture_path(name) = joinpath(@__DIR__, name)
circle_track() = F1Tenth.load_track(fixture_path("circle_tck.csv"))
lumpy_track() = F1Tenth.load_track(fixture_path("lumpy_tck.csv"))
```

- [ ] **Step 3: Generate the fixtures**

```bash
cd /home/pganguli/Downloads/f1tenth && source .venv/bin/activate && cd F1Tenth.jl && python test/fixtures/make_tck_fixtures.py && deactivate
wc -l test/fixtures/*.csv
head -3 test/fixtures/circle_tck.csv
```

Expected: `circle_tck.csv` 208 lines (`# k=3`, header, 206 knots), `lumpy_tck.csv` 308 lines, `lumpy_ref.csv` 58 lines; the first line of each tck CSV is `# k=3`, the second `t,cx,cy,cw`.

- [ ] **Step 4: Add `save_tck_csv` to `../tck.py`**

Insert this function above the `if __name__ == "__main__":` line in `/home/pganguli/Downloads/f1tenth/tck.py`:

```python
def save_tck_csv(path, tck):
    """
    Write a `splprep` tck tuple `(t, [cx, cy, cw], k)` to CSV for F1Tenth.jl's `load_track`.

    Layout: a `# k=<degree>` comment line, a `t,cx,cy,cw` header, then one row per
    knot. SciPy's periodic fit returns coefficient arrays shorter than `t`
    (`len(t) - k - 1`); they are zero-padded to `len(t)` here, and the Julia loader
    ignores the padding.
    """
    t, c, k = tck
    t = np.asarray(t, dtype=float)
    cols = []
    for ci in c:
        padded = np.zeros(len(t))
        padded[: len(ci)] = np.asarray(ci, dtype=float)
        cols.append(padded)
    with open(path, "w") as f:
        f.write(f"# k={int(k)}\n")
        f.write("t,cx,cy,cw\n")
        for row in zip(t, *cols):
            f.write(",".join(repr(float(v)) for v in row) + "\n")
    print(f"tck CSV saved to: {path}")
```

and, in the `__main__` block, insert between the `raster_to_tck` call and the overlay rendering:

```python
    # 2. Save the tck for F1Tenth.jl (load with `load_track`)
    save_tck_csv("Austin_tck.csv", tck)
```

(renumber the following comment to `# 3.`). `tck.py` is the project owner's file and sits outside the git repo, so do not commit it; keep the change to exactly this.

- [ ] **Step 5: Commit**

```bash
git add test/fixtures
git commit -m "Add tck fixtures and generator"
```

---

### Task 2: B-spline evaluation and CSV loader

**Files:**
- Create: `src/geometry/tck.jl`, `test/geometry/tck_test.jl`
- Modify: `src/F1Tenth.jl` (add one include)

**Interfaces:**
- Consumes: fixtures from Task 1.
- Produces (all unexported, used as `F1Tenth.<name>`):
  - `struct BSpline{P}` with fields `t::Vector{Float64}` (knots) and `c::Matrix{Float64}` (3 × n coefficients); callable `(sp::BSpline{P})(u::Float64)::SVector{3,Float64}` (allocation-free)
  - `ncoef(sp)::Int`, `domain(sp)::Tuple{Float64,Float64}`, `derivative(sp::BSpline{P})::BSpline{P-1}`
  - `load_tck_csv(path)::BSpline{3}` (throws `ArgumentError` on malformed input)

- [ ] **Step 1: Write the failing test**

`test/geometry/tck_test.jl`:

```julia
using F1Tenth
using DelimitedFiles: readdlm
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "load_tck_csv + BSpline: matches scipy splev on the lumpy fixture" begin
    sp = F1Tenth.load_tck_csv(fixture_path("lumpy_tck.csv"))
    d1 = F1Tenth.derivative(sp)
    d2 = F1Tenth.derivative(d1)
    ref = readdlm(fixture_path("lumpy_ref.csv"), ','; skipstart=1)
    @test size(ref, 1) == 57
    for i in 1:size(ref, 1)
        u = ref[i, 1]
        p, q, a = sp(u), d1(u), d2(u)
        @test isapprox(collect(p), ref[i, 2:4]; atol=1e-9)   # x, y, w
        @test isapprox(collect(q), ref[i, 5:7]; atol=1e-9)   # dx, dy, dw
        @test isapprox(collect(a)[1:2], ref[i, 8:9]; atol=1e-7)  # ddx, ddy
    end
end

@testset "BSpline: periodic domain and closure" begin
    sp = F1Tenth.load_tck_csv(fixture_path("circle_tck.csv"))
    u0, u1 = F1Tenth.domain(sp)
    @test (u0, u1) == (0.0, 1.0)
    @test isapprox(collect(sp(u0)), collect(sp(u1)); atol=1e-9)
end

@testset "BSpline: evaluation is allocation-free" begin
    sp = F1Tenth.load_tck_csv(fixture_path("lumpy_tck.csv"))
    evaluate(sp, u) = sp(u)
    evaluate(sp, 0.3)  # compile
    @test (@allocated evaluate(sp, 0.3)) == 0
end

@testset "load_tck_csv: rejects malformed files with ArgumentError" begin
    mktempdir() do dir
        function write_csv(name, lines)
            path = joinpath(dir, name)
            open(io -> foreach(l -> println(io, l), lines), path, "w")
            return path
        end
        good_rows = ["$(0.1i),$i.0,$(2i).0,1.0" for i in 1:10]

        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("nohdr.csv", good_rows))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("badhdr.csv", ["a,b,c,d"; good_rows]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("cols.csv", ["t,cx,cy,cw"; "0.0,1.0,2.0"]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("nonnum.csv", ["t,cx,cy,cw"; "0.0,1.0,abc,1.0"]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("nonfinite.csv", ["t,cx,cy,cw"; "0.0,1.0,NaN,1.0"]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("k5.csv", ["# k=5"; "t,cx,cy,cw"; good_rows]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("short.csv", ["t,cx,cy,cw"; good_rows[1:5]]))
        @test_throws ArgumentError F1Tenth.load_tck_csv(
            write_csv("unsorted.csv", ["t,cx,cy,cw"; reverse(good_rows)]))
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/tck_test.jl")'
```

Expected: FAIL/ERROR with `UndefVarError: load_tck_csv` (or `F1Tenth.load_tck_csv` undefined).

- [ ] **Step 3: Implement**

`src/geometry/tck.jl`:

```julia
using StaticArrays

# Clamped/periodic B-spline of degree `P` in R³ with knot vector `t` (1-based)
# and coefficients `c` (3 × n, n = length(t) - P - 1). Mirrors the (t, c, k)
# tuple SciPy's `splprep` returns; this is the `[x, y, w]` curve `tck.py` fits.
struct BSpline{P}
    t::Vector{Float64}
    c::Matrix{Float64}
end

ncoef(sp::BSpline) = size(sp.c, 2)

# (u_lo, u_hi): the interval on which the spline is a genuine curve. For a
# periodic fit the knots extend past it on both sides.
domain(sp::BSpline{P}) where {P} = (sp.t[P+1], sp.t[ncoef(sp)+1])

# Non-zero basis functions N[1:P+1] at `u` for knot span `i` (t[i] <= u < t[i+1]),
# Piegl & Tiller, "The NURBS Book", algorithm A2.2, shifted to 1-based indexing.
@inline function _basis(t::Vector{Float64}, i::Int, u::Float64, ::Val{P}) where {P}
    N = MVector{P + 1,Float64}(undef)
    left = MVector{P + 1,Float64}(undef)
    right = MVector{P + 1,Float64}(undef)
    N[1] = 1.0
    for j in 1:P
        left[j+1] = u - t[i+1-j]
        right[j+1] = t[i+j] - u
        saved = 0.0
        for r in 0:(j-1)
            tmp = N[r+1] / (right[r+2] + left[j-r+1])
            N[r+1] = saved + right[r+2] * tmp
            saved = left[j-r+1] * tmp
        end
        N[j+1] = saved
    end
    return SVector(N)
end

@inline function (sp::BSpline{P})(u::Float64) where {P}
    t = sp.t
    n = ncoef(sp)
    i = clamp(searchsortedlast(t, u), P + 1, n)
    N = _basis(t, i, u, Val(P))
    acc = zero(SVector{3,Float64})
    for l in 0:P
        j = i - P + l
        acc += N[l+1] * SVector(sp.c[1, j], sp.c[2, j], sp.c[3, j])
    end
    return acc
end

# Derivative spline: degree P-1, knots t[2:end-1],
# d_j = P (c_{j+1} - c_j) / (t_{j+P+1} - t_{j+1}).
function derivative(sp::BSpline{P}) where {P}
    P >= 1 || throw(ArgumentError("cannot differentiate a degree-0 spline"))
    n = ncoef(sp)
    d = zeros(3, n - 1)
    for j in 1:(n-1)
        den = sp.t[j+P+1] - sp.t[j+1]
        for r in 1:3
            d[r, j] = den == 0 ? 0.0 : P * (sp.c[r, j+1] - sp.c[r, j]) / den
        end
    end
    return BSpline{P - 1}(sp.t[2:(end-1)], d)
end

"""
    load_tck_csv(path) -> BSpline{3}

Read a tck CSV written by `tck.py:save_tck_csv`: an optional `# k=3` comment
line, a `t,cx,cy,cw` header, then one row per knot. SciPy's coefficient arrays
are zero-padded to `length(t)`; only the first `length(t) - k - 1` entries are
real. Throws `ArgumentError` on any malformed content.
"""
function load_tck_csv(path::AbstractString)
    k = 3
    rows = Vector{NTuple{4,Float64}}()
    header_seen = false
    for (ln, line) in enumerate(eachline(path))
        line = strip(line)
        isempty(line) && continue
        if startswith(line, "#")
            m = match(r"k\s*=\s*(\d+)", line)
            m === nothing || (k = parse(Int, m.captures[1]))
            continue
        end
        if !header_seen
            line == "t,cx,cy,cw" || throw(ArgumentError(
                "$path:$ln: expected header `t,cx,cy,cw`, got `$line`"))
            header_seen = true
            continue
        end
        cols = split(line, ',')
        length(cols) == 4 || throw(ArgumentError(
            "$path:$ln: expected 4 columns, got $(length(cols))"))
        vals = tryparse.(Float64, cols)
        any(isnothing, vals) && throw(ArgumentError("$path:$ln: non-numeric value in `$line`"))
        all(isfinite, vals) || throw(ArgumentError("$path:$ln: non-finite value in `$line`"))
        push!(rows, (vals[1], vals[2], vals[3], vals[4]))
    end
    header_seen || throw(ArgumentError("$path: missing `t,cx,cy,cw` header"))
    k == 3 || throw(ArgumentError("$path: only degree k=3 is supported, got k=$k"))
    m = length(rows)
    n = m - k - 1
    n >= k + 1 || throw(ArgumentError("$path: too few knots ($m) for a degree-$k spline"))
    t = [r[1] for r in rows]
    issorted(t) || throw(ArgumentError("$path: knot vector `t` is not non-decreasing"))
    c = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        c[1, j], c[2, j], c[3, j] = rows[j][2], rows[j][3], rows[j][4]
    end
    return BSpline{3}(t, c)
end
```

In `src/F1Tenth.jl`, add `include("geometry/tck.jl")` immediately before the existing `include("geometry/track.jl")` line.

- [ ] **Step 4: Run to verify it passes**

Same command as Step 2. Expected: all `tck_test.jl` testsets pass (172 + 2 + 1 + 8 assertions). The scipy comparison agrees to the last bit (both use the same recurrence), the tolerance is 1e-9.

- [ ] **Step 5: Commit**

```bash
git add src/geometry/tck.jl src/F1Tenth.jl test/geometry/tck_test.jl
git commit -m "Add tck B-spline evaluation and CSV loader"
```

---

### Task 3: `Track` with arc length, Frenet queries and validation

**Files:**
- Rewrite: `src/geometry/track.jl` (replaces the polyline `Track`, `build_track`, `load_track`)
- Rewrite: `test/geometry/track_test.jl`
- Modify: `src/F1Tenth.jl` (exports)

**Interfaces:**
- Consumes: `BSpline{P}`, `derivative`, `domain`, `load_tck_csv` (Task 2).
- Produces:
  - `struct Segment(p1, p2)` (kept for car-car checks); `struct TrackPoint(x, y, ψ, κ, w, dw)`
  - `struct Track` with fields `spline, d1, d2, u0, period, L, cell, s_tab, speed_tab, xy_tab, κ_max, w_min, w_max, dw_max, κw_max, lipschitz`; constructor `Track(spline::BSpline{3}; table_ds=0.05)`; `load_track(path; kwargs...)`
  - `track_point(track, s::Float64)::TrackPoint` (any real `s`, wrapped mod `L`); `centerline(track, s)::SVector{2}`, `heading`, `curvature`, `halfwidth` (unexported)
  - `frenet_gap(track, s_a, s_b)` (wrapped, in `[−L/2, L/2)`), `to_cartesian(track, s, d)::SVector{2}`
  - `to_frenet(track, x, y; hint=nothing)::Tuple{Float64,Float64}` `(s, d)` (never throws; with `hint`, `s` is the unwrapped value nearest `hint`)
  - `world_pose(track, 𝐱)::SVector{3}` `(x, y, ψ)` for a Frenet state
  - `track_boundaries(track; n=1000)::(left, right)` polylines of `SVector{2}`
  - internals used by later tasks: `_u_of_s`, `_s_of_u`, `_wrap`, `_frenet_u(track, px, py, u_hint)::(u, d, w)`, `_project_u`, `_seed_u`

- [ ] **Step 1: Write the failing test (full file replacement)**

`test/geometry/track_test.jl`:

```julia
using F1Tenth
using StaticArrays
using LinearAlgebra: norm
using Random: MersenneTwister
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "Track (circle R=5, w=1): length, curvature, width" begin
    track = circle_track()
    @test isapprox(track.L, 2π * 5; rtol=1e-6)
    # The fixture is a cubic *interpolant* of 200 circle samples, so its
    # curvature is only accurate to ~4e-5 relative.
    @test isapprox(F1Tenth.curvature(track, 3.0), 0.2; atol=1e-4)
    @test isapprox(F1Tenth.halfwidth(track, 1.0), 1.0; atol=1e-9)
    @test F1Tenth.centerline(track, 0.0) ≈ SVector(5.0, 0.0) atol=1e-9
    @test isapprox(F1Tenth.heading(track, 0.0), π / 2; atol=1e-4)   # CCW: heading +y at (5, 0)
    # s is periodic: s and s + L are the same place.
    @test F1Tenth.centerline(track, 2.0) ≈ F1Tenth.centerline(track, 2.0 + track.L) atol=1e-9
end

@testset "Track: arc-length table is consistent (|dP/ds| = 1)" begin
    track = lumpy_track()
    for s in range(0.1, track.L - 0.1; length=25)
        h = 1e-4
        ds = norm(F1Tenth.centerline(track, s + h) - F1Tenth.centerline(track, s - h)) / 2h
        @test isapprox(ds, 1.0; atol=1e-5)
    end
end

@testset "to_cartesian/to_frenet: round trip on the circle" begin
    track = circle_track()
    for (s, d) in [(0.5, 0.3), (7.0, -0.8), (20.0, 0.0), (31.0, 0.99)]
        x, y = to_cartesian(track, s, d)
        # +d is left of travel = toward the center of a CCW circle: d = R - r.
        @test isapprox(d, 5 - hypot(x, y); atol=1e-6)
        s2, d2 = to_frenet(track, x, y)
        @test isapprox(s2, mod(s, track.L); atol=1e-7)
        @test isapprox(d2, d; atol=1e-7)
        # With a hint the result is the unwrapped s nearest the hint.
        s3, d3 = to_frenet(track, x, y; hint=s + 2track.L)
        @test isapprox(s3, s + 2track.L; atol=1e-7)
    end
end

@testset "to_frenet: round trip on the lumpy track, hinted and unhinted" begin
    track = lumpy_track()
    rng = MersenneTwister(1)
    for _ in 1:500
        s = rand(rng) * track.L
        d = (rand(rng) - 0.5) * 2 * track.w_min
        x, y = to_cartesian(track, s, d)
        s1, d1 = to_frenet(track, x, y; hint=s + 0.3 * randn(rng))
        @test isapprox(s1, s; atol=1e-6)
        @test isapprox(d1, d; atol=1e-7)
        s2, d2 = to_frenet(track, x, y)
        @test isapprox(s2, s; atol=1e-6)
        @test isapprox(d2, d; atol=1e-7)
    end
end

@testset "to_frenet: never throws for far-off points" begin
    track = circle_track()
    for (s, d) in ((9.0, 9.0), (0.0, 5.0), (3.0, -40.0), (1.0, 1e3))
        x, y = to_cartesian(track, s, d)
        s2, d2 = to_frenet(track, x, y; hint=s)
        @test isfinite(s2) && isfinite(d2)
    end
    s2, d2 = to_frenet(track, 0.0, 0.0)   # the center: every foot point is equidistant
    @test isfinite(s2) && isfinite(d2)
end

@testset "frenet_gap: wrapped signed gap" begin
    track = circle_track()
    @test F1Tenth.frenet_gap(track, 1.0, track.L - 1.0) ≈ 2.0
    @test F1Tenth.frenet_gap(track, track.L - 1.0, 1.0) ≈ -2.0
    @test F1Tenth.frenet_gap(track, 5.0, 3.0) ≈ 2.0
end

@testset "world_pose: Frenet state -> (x, y, ψ)" begin
    track = circle_track()
    𝐱 = F1Tenth.State(0.0, 0.3, 0.0, 1.0, 0.1, 0.0, 0.0)   # s=0, d=0.3, μ=0.1
    x, y, ψ = world_pose(track, 𝐱)
    @test x ≈ 4.7 atol=1e-4                 # 0.3 m toward the center of the circle
    @test y ≈ 0.0 atol=1e-4
    @test ψ ≈ π / 2 + 0.1 atol=1e-4
end

@testset "track_boundaries: walls sit w from the centerline" begin
    track = lumpy_track()
    left, right = track_boundaries(track; n=200)
    @test length(left) == length(right) == 200
    for i in 1:200
        s = track.L * (i - 1) / 200
        c = F1Tenth.centerline(track, s)
        w = F1Tenth.halfwidth(track, s)
        @test isapprox(norm(left[i] - c), w; atol=1e-9)
        @test isapprox(norm(right[i] - c), w; atol=1e-9)
        # left wall is at +d
        @test isapprox(to_frenet(track, left[i][1], left[i][2]; hint=s)[2], w; atol=1e-7)
    end
end

@testset "Track: validation rejects bad maps" begin
    sp = F1Tenth.load_tck_csv(fixture_path("circle_tck.csv"))
    n = size(sp.c, 2)
    with_c(c) = F1Tenth.BSpline{3}(sp.t, c)

    wide = copy(sp.c); wide[3, :] .= 6.0       # κ·w = 0.2 · 6 = 1.2 ≥ 0.9
    @test_throws ArgumentError F1Tenth.Track(with_c(wide))
    # The message says where the map is too tight and what to do about it.
    msg = try; F1Tenth.Track(with_c(wide)); "" ; catch e; sprint(showerror, e); end
    @test occursin("s ≈", msg)
    @test occursin("turn radius", msg)
    @test occursin("tck.py", msg)

    negative = copy(sp.c); negative[3, :] .= -1.0
    @test_throws ArgumentError F1Tenth.Track(with_c(negative))

    open_loop = copy(sp.c); open_loop[1, 1] += 1.0   # first coefficient no longer wraps
    @test_throws ArgumentError F1Tenth.Track(with_c(open_loop))
end

@testset "Track queries are allocation-free" begin
    track = lumpy_track()
    alloc_point(t, s) = @allocated F1Tenth.track_point(t, s)
    alloc_frenet(t, x, y, h) = @allocated to_frenet(t, x, y; hint=h)
    alloc_point(track, 5.0); alloc_frenet(track, 10.0, 3.0, 10.0)   # compile
    @test alloc_point(track, 5.0) == 0
    @test alloc_frenet(track, 10.0, 3.0, 10.0) == 0
end
```

- [ ] **Step 2: Run it to verify it fails**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/track_test.jl")'
```

Expected: ERROR at the first testset (`circle_track()` calls the old `load_track`, which reads polyline columns and fails on the tck CSV).

- [ ] **Step 3: Implement (full file replacement)**

`src/geometry/track.jl`:

```julia
using StaticArrays

struct Segment
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
end

# Everything the simulator needs to know about the map at one arc length.
struct TrackPoint
    x::Float64    # centerline position, world frame
    y::Float64
    ψ::Float64    # centerline heading (tangent angle)
    κ::Float64    # signed curvature (positive = turning left)
    w::Float64    # half-width (w_left = w_right = w)
    dw::Float64   # dw/ds
end

"""
    Track

A closed racetrack stored as a tck B-spline over `[x, y, w]` (centerline and
half-width) plus an arc-length table. `s` is arc length along the centerline,
`d` is signed lateral offset (positive = left of travel).
"""
struct Track
    spline::BSpline{3}          # (x, y, w)(u)
    d1::BSpline{2}              # first derivative w.r.t. u
    d2::BSpline{1}              # second derivative w.r.t. u
    u0::Float64                 # start of the parameter domain
    period::Float64             # length of the parameter domain
    L::Float64                  # total centerline length
    cell::Float64               # table spacing in u
    s_tab::Vector{Float64}      # arc length at each table node
    speed_tab::Vector{Float64}  # |P'(u)| at each table node
    xy_tab::Vector{SVector{2,Float64}}  # centerline at each table node
    κ_max::Float64              # max |κ| over the track
    w_min::Float64
    w_max::Float64
    dw_max::Float64             # max |dw/ds|
    κw_max::Float64             # max |κ|·w (< 0.9: the Frenet frame is non-singular)
    lipschitz::Float64          # 1 + dw_max / (1 - κw_max), for sphere tracing
end

# Gauss-Legendre nodes/weights on [-1, 1] by Newton on P_n.
function _gauss_legendre(n::Int)
    x = zeros(n)
    w = zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        dp = 0.0
        for _ in 1:100
            p1, p2 = 1.0, 0.0
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
            end
            dp = n * (z * p1 - p2) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-15 && break
        end
        x[i] = z
        w[i] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

const _GL_X, _GL_W = _gauss_legendre(6)

_speed(d1::BSpline{2}, u::Float64) = (q = d1(u); hypot(q[1], q[2]))

function Track(spline::BSpline{3}; table_ds::Float64=0.05)
    u0, u1 = domain(spline)
    period = u1 - u0
    d1 = derivative(spline)
    d2 = derivative(d1)

    # Table resolution: a coarse length estimate picks N so cells are ~table_ds.
    n_est = 4000
    L_est = sum(_speed(d1, u0 + period * (i - 0.5) / n_est) for i in 1:n_est) * period / n_est
    N = max(1000, ceil(Int, L_est / table_ds))
    cell = period / N

    s_tab = zeros(N + 1)
    speed_tab = zeros(N + 1)
    xy_tab = Vector{SVector{2,Float64}}(undef, N + 1)
    for i in 1:(N+1)
        u = u0 + (i - 1) * cell
        p = spline(min(u, u1))
        xy_tab[i] = SVector(p[1], p[2])
        speed_tab[i] = _speed(d1, min(u, u1))
    end
    for i in 1:N
        ua = u0 + (i - 1) * cell
        acc = 0.0
        for (xg, wg) in zip(_GL_X, _GL_W)
            acc += wg * _speed(d1, ua + cell * (xg + 1) / 2)
        end
        s_tab[i+1] = s_tab[i] + acc * cell / 2
    end
    L = s_tab[end]

    # Extremes on the table nodes (0.05 m apart: fine for smooth splines).
    κ_max, w_min, w_max, dw_max = 0.0, Inf, -Inf, 0.0
    tight, tight_i = 0.0, 1            # worst |κ|·w and where it occurs
    for i in 1:N
        u = u0 + (i - 1) * cell
        p, q, a = spline(u), d1(u), d2(u)
        sp = hypot(q[1], q[2])
        κ = (q[1] * a[2] - q[2] * a[1]) / sp^3
        κ_max = max(κ_max, abs(κ))
        w_min = min(w_min, p[3])
        w_max = max(w_max, p[3])
        dw_max = max(dw_max, abs(q[3]) / sp)
        if abs(κ) * p[3] > tight
            tight, tight_i = abs(κ) * p[3], i
        end
    end

    w_min > 0 || throw(ArgumentError("track half-width must be positive (min w = $w_min)"))
    if tight >= 0.9
        u_t = u0 + (tight_i - 1) * cell
        p_t = spline(u_t)
        κ_t = tight / p_t[3]
        throw(ArgumentError(
            "track too tight for its width: |κ|·w = $(round(tight; digits=3)) ≥ 0.9 at " *
            "s ≈ $(round(s_tab[tight_i]; digits=2)) m (world x=$(round(p_t[1]; digits=2)), " *
            "y=$(round(p_t[2]; digits=2))): turn radius $(round(1 / κ_t; digits=2)) m against " *
            "half-width $(round(p_t[3]; digits=2)) m. Beyond this the Frenet frame is singular " *
            "(the center of curvature lies inside the corridor). Re-fit the map with more " *
            "smoothing in tck.py or a smoother centerline."))
    end
    seam = spline(u0) - spline(u1)
    hypot(seam[1], seam[2]) < 1e-6 || throw(ArgumentError(
        "track centerline does not close (seam gap $(hypot(seam[1], seam[2])) m)"))

    lipschitz = 1 + dw_max / (1 - tight)
    return Track(spline, d1, d2, u0, period, L, cell, s_tab, speed_tab, xy_tab,
        κ_max, w_min, w_max, dw_max, tight, lipschitz)
end

load_track(path::AbstractString; kwargs...) = Track(load_tck_csv(path); kwargs...)

@inline _wrap(track::Track, s::Float64) = mod(s, track.L)

# s (already in [0, L]) -> u by cubic Hermite interpolation of the table.
@inline function _u_of_s(track::Track, s::Float64)
    st = track.s_tab
    N = length(st) - 1
    i = clamp(searchsortedlast(st, s), 1, N)
    ds = st[i+1] - st[i]
    θ = (s - st[i]) / ds
    u_a = track.u0 + (i - 1) * track.cell
    u_b = u_a + track.cell
    m_a = ds / track.speed_tab[i]
    m_b = ds / track.speed_tab[i+1]
    θ2 = θ * θ
    θ3 = θ2 * θ
    return (2θ3 - 3θ2 + 1) * u_a + (θ3 - 2θ2 + θ) * m_a +
           (-2θ3 + 3θ2) * u_b + (θ3 - θ2) * m_b
end

# u (in the domain) -> s by cubic Hermite interpolation of the table.
@inline function _s_of_u(track::Track, u::Float64)
    N = length(track.s_tab) - 1
    i = clamp(floor(Int, (u - track.u0) / track.cell) + 1, 1, N)
    u_a = track.u0 + (i - 1) * track.cell
    θ = (u - u_a) / track.cell
    h = track.cell
    m_a = track.speed_tab[i] * h
    m_b = track.speed_tab[i+1] * h
    θ2 = θ * θ
    θ3 = θ2 * θ
    return (2θ3 - 3θ2 + 1) * track.s_tab[i] + (θ3 - 2θ2 + θ) * m_a +
           (-2θ3 + 3θ2) * track.s_tab[i+1] + (θ3 - θ2) * m_b
end

@inline function _track_point_u(track::Track, u::Float64)
    p = track.spline(u)
    q = track.d1(u)
    a = track.d2(u)
    sp = hypot(q[1], q[2])
    return TrackPoint(p[1], p[2], atan(q[2], q[1]),
        (q[1] * a[2] - q[2] * a[1]) / sp^3, p[3], q[3] / sp)
end

"""Everything about the map at arc length `s` (any real `s`; wrapped mod `L`)."""
@inline track_point(track::Track, s::Float64) = _track_point_u(track, _u_of_s(track, _wrap(track, s)))

centerline(track::Track, s::Float64) = (tp = track_point(track, s); SVector(tp.x, tp.y))
heading(track::Track, s::Float64) = track_point(track, s).ψ
curvature(track::Track, s::Float64) = track_point(track, s).κ
halfwidth(track::Track, s::Float64) = track_point(track, s).w

"""Signed gap `s_a − s_b` wrapped to `[−L/2, L/2)`: positive when A is ahead of B."""
frenet_gap(track::Track, s_a::Float64, s_b::Float64) = mod(s_a - s_b + track.L / 2, track.L) - track.L / 2

"""World-frame position of Frenet point `(s, d)`."""
@inline function to_cartesian(track::Track, s::Float64, d::Float64)
    tp = track_point(track, s)
    return SVector(tp.x - d * sin(tp.ψ), tp.y + d * cos(tp.ψ))
end

# Newton on the foot-point condition (P(u) - p)·P'(u) = 0 from `u`. Returns
# `(u, P, P', converged)`, where P and P' are evaluated at the last iterate
# (within `tol` meters of the returned u), so callers can read d and w without
# re-evaluating the spline.
function _project_u(track::Track, px::Float64, py::Float64, u::Float64; tol::Float64=1e-9)
    p = track.spline(u)
    q = track.d1(u)
    for _ in 1:30
        a = track.d2(u)
        rx, ry = p[1] - px, p[2] - py
        f = rx * q[1] + ry * q[2]
        fp = q[1]^2 + q[2]^2 + rx * a[1] + ry * a[2]
        sp = hypot(q[1], q[2])
        # Clamp the step to 0.5 m of arc so a poor seed can't fly off.
        du = fp > 1e-12 ? -f / fp : -sign(f) * 0.5 / sp
        du = clamp(du, -0.5 / sp, 0.5 / sp)
        u += du
        u = track.u0 + mod(u - track.u0, track.period)
        abs(du) * sp < tol && return u, p, q, true
        p = track.spline(u)
        q = track.d1(u)
    end
    return u, p, q, false
end

# Signed lateral offset of (px, py) from the foot point with position `p` and tangent `q`.
@inline _lateral(px::Float64, py::Float64, p, q) =
    ((px - p[1]) * (-q[2]) + (py - p[2]) * q[1]) / hypot(q[1], q[2])

function _seed_u(track::Track, px::Float64, py::Float64)
    best, best_d = 1, Inf
    for (i, c) in enumerate(track.xy_tab)
        dd = (c[1] - px)^2 + (c[2] - py)^2
        if dd < best_d
            best, best_d = i, dd
        end
    end
    return track.u0 + (best - 1) * track.cell
end

# Projection returning (u, d, w): warm-started from `u_hint`, retrying from a
# table scan if Newton fails to converge. Never throws: a crashed car far off
# the track must not take the sim down, and if both attempts fail the last
# iterate is still within the table resolution of a local foot point.
@inline function _frenet_u(track::Track, px::Float64, py::Float64, u_hint::Float64)
    u, p, q, ok = _project_u(track, px, py, u_hint)
    if !ok
        u, p, q, _ = _project_u(track, px, py, _seed_u(track, px, py))
    end
    return u, _lateral(px, py, p, q), p[3]
end

"""
    to_frenet(track, x, y; hint=nothing) -> (s, d)

Project world point `(x, y)` onto the centerline. With `hint` (a previous
unwrapped `s`), Newton starts there and the returned `s` is the unwrapped
value nearest `hint`; without it, a table scan seeds Newton and `s ∈ [0, L)`.
"""
function to_frenet(track::Track, x::Float64, y::Float64; hint::Union{Nothing,Float64}=nothing)
    u_hint = hint === nothing ? _seed_u(track, x, y) : _u_of_s(track, _wrap(track, hint))
    u, d, _ = _frenet_u(track, x, y, u_hint)
    s = _s_of_u(track, u)
    hint === nothing || (s = hint + frenet_gap(track, s, hint))
    return s, d
end

"""World pose `(x, y, ψ)` of a Frenet state `(s, d, δ, v, μ, ψ̇, β)`."""
@inline function world_pose(track::Track, 𝐱)
    s, d, μ = 𝐱[1], 𝐱[2], 𝐱[5]
    tp = track_point(track, s)
    return SVector(tp.x - d * sin(tp.ψ), tp.y + d * cos(tp.ψ), tp.ψ + μ)
end

"""Left/right wall polylines `P(s) ± w(s)·N(s)` at `n` uniform arc-length samples (closed: last ≠ first)."""
function track_boundaries(track::Track; n::Int=1000)
    left = Vector{SVector{2,Float64}}(undef, n)
    right = Vector{SVector{2,Float64}}(undef, n)
    for i in 1:n
        tp = track_point(track, track.L * (i - 1) / n)
        nx, ny = -sin(tp.ψ), cos(tp.ψ)
        left[i] = SVector(tp.x + tp.w * nx, tp.y + tp.w * ny)
        right[i] = SVector(tp.x - tp.w * nx, tp.y - tp.w * ny)
    end
    return left, right
end
```

- [ ] **Step 4: Update exports**

In `src/F1Tenth.jl` replace

```julia
export Track, Segment, build_track, load_track
```

with

```julia
export Track, Segment, load_track, track_point, to_frenet, to_cartesian, world_pose
export frenet_gap, track_boundaries
```

(leave `export TrackBVH, build_bvh` for now; Task 7 removes it).

- [ ] **Step 5: Run to verify it passes**

Same command as Step 2. Expected: all `track_test.jl` testsets pass (incl. 2000 hinted/unhinted round trips and the two allocation-free assertions). If an allocation assertion fails, check `@inline` on `track_point`/`_frenet_u` and that nothing captures a closure over `track`.

- [ ] **Step 6: Commit**

```bash
git add src/geometry/track.jl src/F1Tenth.jl test/geometry/track_test.jl
git commit -m "Replace polyline Track with a tck-backed Frenet Track"
```

---

### Task 4: Frenet dynamics

**Files:**
- Create: `src/dynamics/frenet.jl`, `test/frenet_test.jl`
- Modify: `src/dynamics/step.jl` (rename), `test/step_test.jl` (rename calls), `src/F1Tenth.jl` (include)

**Interfaces:**
- Consumes: `Track`, `track_point`, `world_pose` (Task 3); `dynamics_rhs`, `State`, `Action`, `VehicleParams` (existing).
- Produces: `frenet_rhs(𝐱::State, action::Action, params::VehicleParams, track::Track)::State`; `step_dynamics(𝐱::State, action::Action, params::VehicleParams, track::Track, dt::Float64)::State` (exported); Cartesian `step_dynamics_cartesian(𝐱, action, params, dt)` (unexported).
- Mechanism: `frenet_rhs` evaluates the Cartesian `dynamics_rhs` at pose `(0, 0, μ)`; its `(ẋ, ẏ)` are then exactly the velocity along the track tangent and left normal, so the body dynamics and the KS/ST switch are reused verbatim.

- [ ] **Step 1: Write the failing test**

`test/frenet_test.jl`:

```julia
using F1Tenth
using Test

include(joinpath(@__DIR__, "fixtures", "fixtures.jl"))

# World-frame Cartesian state equivalent to Frenet state 𝐱.
function _cartesian_of(track, 𝐱)
    x, y, ψ = world_pose(track, 𝐱)
    return F1Tenth.State(x, y, 𝐱[3], 𝐱[4], ψ, 𝐱[6], 𝐱[7])
end

# Drive both frames with the same actions and return the worst world-position gap.
function _worst_position_gap(track, 𝐱_frenet, actions, dt)
    params = F1Tenth.VehicleParams()
    𝐱_cart = _cartesian_of(track, 𝐱_frenet)
    worst = 0.0
    for a in actions
        𝐱_frenet = F1Tenth.step_dynamics(𝐱_frenet, a, params, track, dt)
        𝐱_cart = F1Tenth.step_dynamics_cartesian(𝐱_cart, a, params, dt)
        x, y, _ = world_pose(track, 𝐱_frenet)
        worst = max(worst, hypot(x - 𝐱_cart[1], y - 𝐱_cart[2]))
    end
    return worst
end

@testset "Frenet dynamics match the Cartesian reference (ST, circle)" begin
    track = circle_track()
    actions = [F1Tenth.Action(0.5 * sin(0.01i), 1.0) for i in 1:1000]
    𝐱₀ = F1Tenth.State(0.0, 0.3, 0.0, 2.0, 0.1, 0.0, 0.0)
    @test _worst_position_gap(track, 𝐱₀, actions, 1e-3) < 1e-6
end

@testset "Frenet dynamics match the Cartesian reference (KS regime, lumpy)" begin
    track = lumpy_track()
    actions = [F1Tenth.Action(0.3, 0.2) for _ in 1:500]
    𝐱₀ = F1Tenth.State(20.0, -0.4, 0.0, 0.1, 0.05, 0.0, 0.0)   # |v| < 0.5 -> KS
    @test _worst_position_gap(track, 𝐱₀, actions, 1e-3) < 1e-6
end

@testset "Frenet dynamics match the Cartesian reference (ST, lumpy, long)" begin
    track = lumpy_track()
    actions = [F1Tenth.Action(0.2 * sin(0.02i), 0.5) for i in 1:1000]
    𝐱₀ = F1Tenth.State(60.0, 0.2, 0.0, 3.0, -0.1, 0.0, 0.0)
    @test _worst_position_gap(track, 𝐱₀, actions, 1e-3) < 1e-6
end

@testset "Frenet dynamics: driving straight along the centerline keeps d and μ at 0 on a straight-ish stretch" begin
    track = lumpy_track()
    # On a straight, with δ = 0 the car should hold d = μ = 0 while s advances at v.
    # Find the straightest point on the track and start there.
    s_straight = argmin(s -> abs(F1Tenth.curvature(track, s)), range(0.0, track.L; length=400))
    𝐱 = F1Tenth.State(s_straight, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    params = F1Tenth.VehicleParams()
    𝐱′ = F1Tenth.step_dynamics(𝐱, F1Tenth.Action(0.0, 0.0), params, track, 0.01)
    @test isapprox(𝐱′[1] - 𝐱[1], 2.0 * 0.01; atol=1e-4)
    @test abs(𝐱′[2]) < 1e-4
end

@testset "Frenet dynamics: heading error is wrapped to (-π, π]" begin
    track = circle_track()
    𝐱 = F1Tenth.State(0.0, 0.0, 0.0, 2.0, 3.1, 0.0, 0.0)
    for _ in 1:200
        𝐱 = F1Tenth.step_dynamics(𝐱, F1Tenth.Action(0.0, 0.0), F1Tenth.VehicleParams(), track, 0.01)
        @test -π <= 𝐱[5] <= π
    end
end

@testset "Frenet dynamics: off-track states stay finite (1 - κd guard)" begin
    track = circle_track()
    params = F1Tenth.VehicleParams()
    # d = 5 on the R = 5 circle is the circle's center: 1 - κd = 0 without the guard.
    for d in (4.5, 5.0, 5.5, 40.0)
        𝐱 = F1Tenth.step_dynamics(F1Tenth.State(1.0, d, 0.0, 2.0, 0.0, 0.0, 0.0),
            F1Tenth.Action(0.0, 0.0), params, track, 0.01)
        @test all(isfinite, 𝐱)
    end
end

@testset "step_dynamics (Frenet) is allocation-free" begin
    track = lumpy_track()
    params = F1Tenth.VehicleParams()
    a = F1Tenth.Action(0.1, 1.0)
    𝐱 = F1Tenth.State(10.0, 0.3, 0.0, 2.0, 0.1, 0.0, 0.0)
    F1Tenth.step_dynamics(𝐱, a, params, track, 0.01)  # warmup
    @test (@allocated F1Tenth.step_dynamics(𝐱, a, params, track, 0.01)) == 0
end
```

- [ ] **Step 2: Rename the Cartesian step and its tests**

Apply to `src/dynamics/step.jl`:

```diff
--- a/src/dynamics/step.jl
+++ b/src/dynamics/step.jl
@@ -30,7 +30,11 @@
 
 # k₁..k₄: classical RK4 stage derivatives (dynamics_rhs evaluated at the
 # start, two midpoint estimates, and the endpoint of the step).
-function step_dynamics(𝐱::State, action::Action, params::VehicleParams, dt::Float64)
+#
+# Cartesian-frame step: the reference model the Frenet `step_dynamics`
+# (dynamics/frenet.jl) is validated against, and what the Python-gym
+# comparison tests exercise.
+function step_dynamics_cartesian(𝐱::State, action::Action, params::VehicleParams, dt::Float64)
     k₁ = dynamics_rhs(𝐱, action, params)
     k₂ = dynamics_rhs(𝐱 + dt * (k₁ / 2), action, params)
     k₃ = dynamics_rhs(𝐱 + dt * (k₂ / 2), action, params)
```

then update the existing parity tests:

```bash
sed -i 's/F1Tenth\.step_dynamics(/F1Tenth.step_dynamics_cartesian(/g; s/"step_dynamics is allocation-free"/"step_dynamics_cartesian is allocation-free"/' test/step_test.jl
```

- [ ] **Step 3: Run the new test to verify it fails**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/frenet_test.jl")'
```

Expected: ERROR (`step_dynamics` has no 5-argument method / `frenet_rhs` undefined).

- [ ] **Step 4: Implement**

`src/dynamics/frenet.jl`:

```julia
# Frenet-frame vehicle dynamics. State: (s, d, δ, v, μ, ψ̇, β), i.e. the
# Cartesian (x, y, δ, v, ψ, ψ̇, β) with x → s (arc length along the
# centerline), y → d (signed lateral offset, + = left) and ψ → μ (heading
# relative to the centerline tangent).
#
# Only the pose kinematics differ from the Cartesian model:
#     ṡ = v·cos(μ+β) / (1 − κ d)
#     ḋ = v·sin(μ+β)
#     μ̇ = ψ̇ − κ ṡ
# The body dynamics (δ̇, v̇, ψ̈, β̇, and the KS/ST switch) are reused verbatim by
# evaluating the Cartesian `dynamics_rhs` at pose (0, 0, μ): its (ẋ, ẏ) are then
# exactly the velocity components along the track tangent and left normal.
# Smallest allowed value of `1 − κ d`; off-track states (already flagged as
# collisions) can approach the center of curvature and would otherwise blow up.
const _MIN_FRENET_SCALE = 0.1

@inline function frenet_rhs(𝐱::State, action::Action, params::VehicleParams, track::Track)
    s, d, δ, v, μ, ψ̇, β = 𝐱
    cart = dynamics_rhs(State(0.0, 0.0, δ, v, μ, ψ̇, β), action, params)
    κ = track_point(track, s).κ
    ṡ = cart[1] / max(1 - κ * d, _MIN_FRENET_SCALE)
    return State(ṡ, cart[2], cart[3], cart[4], cart[5] - κ * ṡ, cart[6], cart[7])
end

# Classical RK4, as in the Cartesian `step_dynamics_cartesian`.
function step_dynamics(𝐱::State, action::Action, params::VehicleParams, track::Track, dt::Float64)
    k₁ = frenet_rhs(𝐱, action, params, track)
    k₂ = frenet_rhs(𝐱 + dt * (k₁ / 2), action, params, track)
    k₃ = frenet_rhs(𝐱 + dt * (k₂ / 2), action, params, track)
    k₄ = frenet_rhs(𝐱 + dt * k₃, action, params, track)

    s, d, δ, v, μ, ψ̇, β = 𝐱 + dt * (1 / 6) * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    μ = mod(μ + π, 2π) - π
    return State(s, d, δ, v, μ, ψ̇, β)
end
```

In `src/F1Tenth.jl` add `include("dynamics/frenet.jl")` immediately after `include("geometry/track.jl")`.

- [ ] **Step 5: Run to verify both files pass**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/frenet_test.jl"); include("test/step_test.jl")'
```

Expected: all pass. The Frenet-vs-Cartesian equivalence gap is ~1e-10 m (test bound 1e-6), and `step_dynamics` (Frenet) allocates 0 bytes.

- [ ] **Step 6: Commit**

```bash
git add src/dynamics test/frenet_test.jl test/step_test.jl src/F1Tenth.jl
git commit -m "Add Frenet-frame dynamics; keep Cartesian step as reference"
```

---

### Task 5: LiDAR: sphere-traced `cast_ray` and the benchmark gate

**Files:**
- Rewrite: `src/geometry/raycast.jl`, `test/geometry/raycast_test.jl`
- Create: `benchmarks/lidar_scan.jl`

**Interfaces:**
- Consumes: `Track`, `Segment`, `world_pose`, `_frenet_u`, `_u_of_s`, `_wrap`, `track_point`, `track_boundaries`, `Track.lipschitz` (Task 3).
- Produces: `LiDARConfig`, `CarFootprint` (unchanged API); `car_vertices(pose::SVector{3,Float64}, car::CarFootprint)::Vector{SVector{2,Float64}}` (**signature change**: takes a world pose, not a `State`); `cast_ray(track, origin::SVector{2,Float64}, angle::Float64, s_hint::Float64, max_range::Float64; step_min=0.1, tol=1e-4)::Float64`; `scan(state::State, track::Track, other_agents::Vector{State}, car, cfg; rng=nothing)::Vector{Float64}`; `_cast_ray_against_segments`, `_other_agent_segments(track, others, car)`.
- `angle` and `s_hint` must be `Float64` (pass `Float64(π)`, not `π`).

- [ ] **Step 1: Write the failing test (full file replacement)**

`test/geometry/raycast_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Random: MersenneTwister
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

# Circle fixture: R = 5 centerline, half-width 1 -> walls at r = 4 (inner, +d)
# and r = 6 (outer, -d). s = 0 is world (5, 0) heading +y.

@testset "cast_ray: analytic ranges on the circle track" begin
    track = circle_track()
    origin = SVector(5.0, 0.0)
    # Along the tangent (+y) the ray leaves through the outer wall: 25 + t² = 36.
    @test isapprox(F1Tenth.cast_ray(track, origin, π / 2, 0.0, 30.0), sqrt(11.0); atol=1e-3)
    # Straight toward the center: inner wall at r = 4, one meter away.
    @test isapprox(F1Tenth.cast_ray(track, origin, Float64(π), 0.0, 30.0), 1.0; atol=1e-3)
    # Straight outward: outer wall at r = 6.
    @test isapprox(F1Tenth.cast_ray(track, origin, 0.0, 0.0, 30.0), 1.0; atol=1e-3)
end

@testset "cast_ray: clips to max_range, and returns 0 from outside the track" begin
    track = circle_track()
    @test F1Tenth.cast_ray(track, SVector(5.0, 0.0), π / 2, 0.0, 2.0) == 2.0
    @test F1Tenth.cast_ray(track, SVector(6.5, 0.0), π / 2, 0.0, 30.0) == 0.0
end

@testset "cast_ray: agrees with a brute-force ray/segment cast on the lumpy track" begin
    track = lumpy_track()
    n = 20000
    left, right = track_boundaries(track; n=n)
    segments = F1Tenth.Segment[]
    for i in 1:n
        j = mod1(i + 1, n)
        push!(segments, F1Tenth.Segment(left[i], left[j]))
        push!(segments, F1Tenth.Segment(right[i], right[j]))
    end
    rng = MersenneTwister(3)
    worst = 0.0
    for _ in 1:100
        s = rand(rng) * track.L
        d = (rand(rng) - 0.5) * 1.2 * track.w_min
        x, y = to_cartesian(track, s, d)
        ψ = F1Tenth.heading(track, s) + 0.5 * randn(rng)
        for angle in range(ψ - 2.35, ψ + 2.35; length=21)
            sphere = F1Tenth.cast_ray(track, SVector(x, y), angle, s, 30.0)
            brute = F1Tenth._cast_ray_against_segments(SVector(x, y), angle, segments, 30.0)
            worst = max(worst, abs(sphere - brute))
        end
    end
    # The dense polyline is itself within ~0.01 mm of the true boundary.
    @test worst < 1e-3
end

@testset "car_vertices: axis-aligned car at the world origin" begin
    car = F1Tenth.CarFootprint(; length=2.0, width=1.0)
    verts = F1Tenth.car_vertices(SVector(0.0, 0.0, 0.0), car)
    @test length(verts) == 4
    @test SVector(1.0, 0.5) in verts
    @test SVector(-1.0, -0.5) in verts
end

@testset "scan: sees the wall and a blocking agent" begin
    track = circle_track()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.0, max_range=30.0, noise_std=0.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # fov=0.0 with num_beams=3 sends all 3 beams straight along the heading.
    ranges_no_agents = F1Tenth.scan(state, track, F1Tenth.State[], car, cfg)
    @test all(isapprox.(ranges_no_agents, sqrt(11.0); atol=1e-3))

    blocking_agent = F1Tenth.State(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # 0.5 m ahead
    ranges_with_agent = F1Tenth.scan(state, track, [blocking_agent], car, cfg)
    @test all(r -> r < 1.0, ranges_with_agent)
end

@testset "scan: deterministic noise via explicit rng" begin
    track = circle_track()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=30.0, noise_std=0.05)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    r1 = F1Tenth.scan(state, track, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r2 = F1Tenth.scan(state, track, F1Tenth.State[], car, cfg; rng=MersenneTwister(1))
    r3 = F1Tenth.scan(state, track, F1Tenth.State[], car, cfg; rng=MersenneTwister(2))
    @test r1 == r2
    @test r1 != r3
end

@testset "scan: output is always clamped to [0, max_range] even with large noise" begin
    track = circle_track()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=50, fov=6.0, max_range=10.0, noise_std=100.0)
    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for seed in 1:10
        ranges = F1Tenth.scan(state, track, F1Tenth.State[], car, cfg; rng=MersenneTwister(seed))
        @test all(r -> r >= 0.0, ranges)
        @test all(r -> r <= cfg.max_range, ranges)
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/raycast_test.jl")'
```

Expected: ERROR (`cast_ray` has no method taking a `Track`).

- [ ] **Step 3: Implement (full file replacement)**

`src/geometry/raycast.jl`:

```julia
using StaticArrays
using Random: AbstractRNG, randn

struct LiDARConfig
    num_beams::Int
    fov::Float64
    max_range::Float64
    noise_std::Float64
end

LiDARConfig(; num_beams::Int=1080, fov::Float64=4.7, max_range::Float64=30.0,
    noise_std::Float64=0.01) = LiDARConfig(num_beams, fov, max_range, noise_std)

struct CarFootprint
    length::Float64
    width::Float64
end

CarFootprint(; length::Float64=0.58, width::Float64=0.31) = CarFootprint(length, width)

# Corner order: front-left, front-right, rear-right, rear-left (vehicle
# frame: +x forward, +y left). `pose` is the world pose (x, y, ψ).
function car_vertices(pose::SVector{3,Float64}, car::CarFootprint)
    x, y, ψ = pose
    c, s = cos(ψ), sin(ψ)
    hl, hw = car.length / 2, car.width / 2
    local_corners = (SVector(hl, hw), SVector(hl, -hw), SVector(-hl, -hw), SVector(-hl, hw))
    return [SVector(x + c * p[1] - s * p[2], y + s * p[1] + c * p[2]) for p in local_corners]
end

function _ray_segment_intersection(origin::SVector{2,Float64}, dir::SVector{2,Float64}, seg::Segment)
    s = seg.p2 - seg.p1
    rxs = dir[1] * s[2] - dir[2] * s[1]
    if abs(rxs) < 1e-12
        return nothing
    end
    qp = seg.p1 - origin
    t = (qp[1] * s[2] - qp[2] * s[1]) / rxs
    u = (qp[1] * dir[2] - qp[2] * dir[1]) / rxs
    return (t >= 0 && 0 <= u <= 1) ? t : nothing
end

"""
    cast_ray(track, origin, angle, s_hint, max_range; step_min=0.1, tol=1e-4)

Distance along the world-frame ray from `origin` at `angle` to the first point
outside `|d| ≤ w(s)`, or `max_range`. Sphere tracing: `m = w(s) − |d|` is a
lower bound on the distance to the wall up to the factor `track.lipschitz`, so
stepping `m / lipschitz` never jumps a wall. Steps are floored at `step_min`
(bounds the cost of near-parallel rays). Once a sample lands outside, bisect the
last inside/outside bracket down to `tol`. `s_hint` warm-starts the projection.
"""
function cast_ray(track::Track, origin::SVector{2,Float64}, angle::Float64, s_hint::Float64,
    max_range::Float64; step_min::Float64=0.1, tol::Float64=1e-4)
    dir = SVector(cos(angle), sin(angle))
    t_in = 0.0
    t = 0.0
    u = _u_of_s(track, _wrap(track, s_hint))
    while true
        p = origin + t * dir
        u, d, w = _frenet_u(track, p[1], p[2], u)
        m = w - abs(d)
        if m < 0
            t == 0.0 && return 0.0        # origin already outside the track
            lo, hi = t_in, t
            while hi - lo > tol
                mid = (lo + hi) / 2
                pm = origin + mid * dir
                um, dm, wm = _frenet_u(track, pm[1], pm[2], u)
                if wm - abs(dm) < 0
                    hi = mid
                else
                    lo = mid
                    u = um
                end
            end
            return (lo + hi) / 2
        end
        t >= max_range && return max_range
        t_in = t
        t = min(t + max(m / track.lipschitz, step_min), max_range)
    end
end

function _cast_ray_against_segments(origin::SVector{2,Float64}, angle::Float64,
    segments::Vector{Segment}, max_range::Float64)
    dir = SVector(cos(angle), sin(angle))
    best = max_range
    for seg in segments
        t_hit = _ray_segment_intersection(origin, dir, seg)
        if t_hit !== nothing && t_hit < best
            best = t_hit
        end
    end
    return best
end

function _other_agent_segments(track::Track, other_agents::Vector{State}, car::CarFootprint)
    segments = Segment[]
    for agent in other_agents
        verts = car_vertices(world_pose(track, agent), car)
        for i in 1:4
            push!(segments, Segment(verts[i], verts[mod1(i + 1, 4)]))
        end
    end
    return segments
end

function scan(state::State, track::Track, other_agents::Vector{State},
    car::CarFootprint, cfg::LiDARConfig; rng::Union{Nothing,AbstractRNG}=nothing)
    x, y, ψ = world_pose(track, state)
    origin = SVector(x, y)
    other_segments = _other_agent_segments(track, other_agents, car)
    angles = cfg.num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=cfg.num_beams)
    ranges = Vector{Float64}(undef, cfg.num_beams)
    for (i, angle) in enumerate(angles)
        t_wall = cast_ray(track, origin, angle, state[1], cfg.max_range)
        t_agents = _cast_ray_against_segments(origin, angle, other_segments, cfg.max_range)
        ranges[i] = min(t_wall, t_agents)
    end
    if rng !== nothing && cfg.noise_std > 0
        ranges .+= cfg.noise_std .* randn(rng, cfg.num_beams)
    end
    return clamp.(ranges, 0.0, cfg.max_range)
end
```

- [ ] **Step 4: Run to verify it passes**

Same command as Step 2. Expected: all pass, including the brute-force comparison (100 poses × 21 beams within 1e-3 m; measured worst 5e-5 m).

- [ ] **Step 5: Create and run the benchmark gate**

`benchmarks/lidar_scan.jl`:

```julia
# LiDAR scan timing gate (spec: a 1080-beam scan must take <= 5 ms single-thread).
#
#   julia --project=. benchmarks/lidar_scan.jl [track_tck.csv]
#
# Defaults to the lumpy test track. Scans from several arc lengths around the
# lap (the cost depends on how far beams travel) and reports the worst mean.
# Exits nonzero if the gate fails, so it can be wired into CI.
using F1Tenth

const GATE_MS = 5.0

path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "test", "fixtures", "lumpy_tck.csv")
track = load_track(path)
car = CarFootprint()
cfg = LiDARConfig(; num_beams=1080, fov=4.7, max_range=30.0, noise_std=0.0)

worst_ms = 0.0
scan(State(0.0, 0.2, 0.0, 1.0, 0.05, 0.0, 0.0), track, State[], car, cfg)   # compile
for s in range(0.0, track.L; length=9)[1:(end-1)]
    𝐱 = State(s, 0.2, 0.0, 1.0, 0.05, 0.0, 0.0)
    reps = 10
    ms = 1000 * (@elapsed for _ in 1:reps
        scan(𝐱, track, State[], car, cfg)
    end) / reps
    global worst_ms = max(worst_ms, ms)
    println("s = $(round(s; digits=1)) m: $(round(ms; digits=2)) ms")
end
println("worst mean: $(round(worst_ms; digits=2)) ms (gate $(GATE_MS) ms) on $(basename(path))")
exit(worst_ms <= GATE_MS ? 0 : 1)
```

```bash
julia --project=. benchmarks/lidar_scan.jl; echo "exit=$?"
```

Expected: `worst mean: ~3.3 ms (gate 5.0 ms)`, `exit=0`. **If the gate fails** (> 5 ms on your machine): first re-run once (JIT/first-run noise), then thread over beams inside `scan` (`Threads.@threads` over `enumerate(angles)`; beams are independent and `cast_ray` is allocation-free). Only if that is still not enough, stop and report to the project owner; the spec's fallback is a derived-polyline raycaster for LiDAR only.

- [ ] **Step 6: Commit**

```bash
git add src/geometry/raycast.jl test/geometry/raycast_test.jl benchmarks/lidar_scan.jl
git commit -m "Sphere-trace the tck boundary for LiDAR; add scan benchmark gate"
```

---

### Task 6: Collision (footprint samples in Frenet, SAT for cars)

**Files:**
- Rewrite: `src/geometry/collision.jl`, `test/geometry/collision_test.jl`

**Interfaces:**
- Consumes: `world_pose`, `to_frenet`, `track_point` (Task 3); `car_vertices(pose, car)` (Task 5).
- Produces: `collides(vertices_a, vertices_b)::Bool` (unchanged SAT); `check_car_collisions(states::Vector{State}, track::Track, car::CarFootprint)::BitVector` (**new `track` argument**); `check_track_collision(state::State, track::Track, car::CarFootprint)::Bool` (**takes a `Track`, not a BVH**).

- [ ] **Step 1: Write the failing test (full file replacement)**

`test/geometry/collision_test.jl`:

```julia
using F1Tenth
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "collides: SAT on axis-aligned rectangles" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    overlapping = [SVector(1.0, 1.0), SVector(3.0, 1.0), SVector(3.0, 3.0), SVector(1.0, 3.0)]
    separated = [SVector(5.0, 5.0), SVector(7.0, 5.0), SVector(7.0, 7.0), SVector(5.0, 7.0)]
    touching = [SVector(2.0, 0.0), SVector(4.0, 0.0), SVector(4.0, 2.0), SVector(2.0, 2.0)]

    @test F1Tenth.collides(a, overlapping)
    @test !F1Tenth.collides(a, separated)
    @test F1Tenth.collides(a, touching)
end

@testset "collides: SAT on a rotated rectangle" begin
    a = [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 2.0), SVector(0.0, 2.0)]
    # A 45-degree-rotated square centered on a's center (1,1), sized to
    # clearly overlap a's interior (not just touch at a boundary point,
    # which would make the test fragile to floating-point rounding).
    rotated = [SVector(1.0, -0.2), SVector(2.2, 1.0), SVector(1.0, 2.2), SVector(-0.2, 1.0)]
    @test F1Tenth.collides(a, rotated)
end

@testset "check_car_collisions: all-pairs over an agent list" begin
    track = circle_track()
    car = F1Tenth.CarFootprint(; length=0.5, width=0.3)
    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),    # overlaps agent 1
        F1Tenth.State(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),   # isolated
    ]
    flags = F1Tenth.check_car_collisions(states, track, car)
    @test flags == [true, true, false]
end

# Circle fixture: half-width 1, so a car is off-track once a footprint sample has |d| > 1.
@testset "check_track_collision: centered, penetrating and sideways poses" begin
    track = circle_track()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    at(d, μ) = F1Tenth.State(2.0, d, 0.0, 0.0, μ, 0.0, 0.0)

    @test !F1Tenth.check_track_collision(at(0.0, 0.0), track, car)
    @test !F1Tenth.check_track_collision(at(0.8, 0.0), track, car)     # corners at d = 0.955
    @test F1Tenth.check_track_collision(at(0.9, 0.0), track, car)      # corners at d = 1.055
    @test F1Tenth.check_track_collision(at(-0.9, 0.0), track, car)     # right wall too
    # Sideways (μ = π/2): the car's length now spans d.
    @test !F1Tenth.check_track_collision(at(0.7, π / 2), track, car)   # front edge at d = 0.99
    @test F1Tenth.check_track_collision(at(0.75, π / 2), track, car)   # front edge at d = 1.04
end

@testset "check_track_collision: edge midpoints catch a wall bulging into the car" begin
    # The circle's inner wall (r = 4) is convex into the track. With the car
    # tangent to it, the side-edge midpoint reaches r = 5 - (d + 0.155) while
    # the corners sit ~1 cm farther out (0.29² / (2·4)). At d = 0.85 the
    # midpoint is 5 mm inside the wall and both corners are still inside the
    # track: only the midpoint sample can see it.
    track = circle_track()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    @test F1Tenth.check_track_collision(F1Tenth.State(2.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0), track, car)
    @test !F1Tenth.check_track_collision(F1Tenth.State(2.0, 0.84, 0.0, 0.0, 0.0, 0.0, 0.0), track, car)
end

@testset "check_track_collision: far-off-track states are collisions, not errors" begin
    track = circle_track()
    car = F1Tenth.CarFootprint()
    for (s, d) in ((9.0, 9.0), (0.0, 5.0), (3.0, -40.0))
        @test F1Tenth.check_track_collision(F1Tenth.State(s, d, 0.0, 0.0, 0.0, 0.0, 0.0), track, car)
    end
end
```

- [ ] **Step 2: Run it to verify it fails**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/collision_test.jl")'
```

Expected: ERROR/FAIL in the `check_car_collisions` and `check_track_collision` testsets (old methods take no `Track`); the two SAT testsets still pass.

- [ ] **Step 3: Implement (full file replacement)**

`src/geometry/collision.jl`:

```julia
using StaticArrays
using LinearAlgebra: dot, normalize

function _sat_axes(vertices::Vector{SVector{2,Float64}})
    n = length(vertices)
    return [normalize(SVector(-(e[2]), e[1]))
            for e in (vertices[mod1(i + 1, n)] - vertices[i] for i in 1:n)]
end

function _project(vertices::Vector{SVector{2,Float64}}, axis::SVector{2,Float64})
    return extrema(dot(v, axis) for v in vertices)
end

function collides(vertices_a::Vector{SVector{2,Float64}}, vertices_b::Vector{SVector{2,Float64}})
    for axis in vcat(_sat_axes(vertices_a), _sat_axes(vertices_b))
        min_a, max_a = _project(vertices_a, axis)
        min_b, max_b = _project(vertices_b, axis)
        if max_a < min_b || max_b < min_a
            return false
        end
    end
    return true
end

function check_car_collisions(states::Vector{State}, track::Track, car::CarFootprint)
    n = length(states)
    verts = [car_vertices(world_pose(track, s), car) for s in states]
    flags = falses(n)
    for i in 1:n-1, j in i+1:n
        if collides(verts[i], verts[j])
            flags[i] = true
            flags[j] = true
        end
    end
    return flags
end


# Footprint sample points in the car frame: 4 corners then 4 edge midpoints.
# Midpoints matter on the inside of a curve, where the (convex-into-the-track)
# wall can touch an edge before a corner: up to length²/(8R), ~2 cm at R = 2 m.
@inline function _footprint_samples(car::CarFootprint)
    hl, hw = car.length / 2, car.width / 2
    return (SVector(hl, hw), SVector(hl, -hw), SVector(-hl, -hw), SVector(-hl, hw),
        SVector(hl, 0.0), SVector(0.0, -hw), SVector(-hl, 0.0), SVector(0.0, hw))
end

"""True if any footprint sample point is outside `|d| ≤ w(s)` (`w_left = w_right = w`)."""
function check_track_collision(state::State, track::Track, car::CarFootprint)
    x, y, ψ = world_pose(track, state)
    c, sψ = cos(ψ), sin(ψ)
    hint = state[1]
    for p in _footprint_samples(car)
        wx = x + c * p[1] - sψ * p[2]
        wy = y + sψ * p[1] + c * p[2]
        s_i, d_i = to_frenet(track, wx, wy; hint=hint)
        w = track_point(track, s_i).w
        (d_i > w || d_i < -w) && return true
        hint = s_i
    end
    return false
end
```

- [ ] **Step 4: Run to verify it passes**

Same command as Step 2. Expected: all pass, including the edge-midpoint case (`d = 0.85` collides, `d = 0.84` does not) and the far-off-track states (collisions, not errors).

- [ ] **Step 5: Commit**

```bash
git add src/geometry/collision.jl test/geometry/collision_test.jl
git commit -m "Check wall collisions in Frenet space with 8 footprint samples"
```

---

### Task 7: Multi-agent step and `AsyncSim` on a `Track`; drop the BVH

**Files:**
- Rewrite: `src/geometry/step2.jl`, `test/geometry/step2_test.jl`, `test/async/testutils.jl`
- Modify: `src/async/sim.jl`, `test/async/sim_test.jl`, `src/F1Tenth.jl`, `Project.toml`, `test/exports_test.jl`
- Delete: `src/geometry/bvh.jl`, `test/geometry/bvh_test.jl`

**Interfaces:**
- Consumes: Frenet `step_dynamics`, `scan`, `check_car_collisions`, `check_track_collision` (Tasks 4–6).
- Produces: `step_multi_agent!(states, actions, track::Track, params, car, dt, cfg; rng)` → `(new_states, scans, collisions)`; `AsyncSim(track::Track, params, car, dt, cfg, initial_states, initial_actions; tick_hz, renderers, rng)` with field `sim.track`; `reset!(sim, track, initial_states)` now also sets `sim.track = track`. Async test setup NamedTuple is `(; track, params, car, cfg, states, actions)` (no `bvh`).

- [ ] **Step 1: Write the failing tests**

`test/geometry/step2_test.jl` (full replacement):

```julia
using F1Tenth
using StaticArrays
using Random: MersenneTwister
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "step_multi_agent!: Frenet state evolution is unaffected" begin
    track = circle_track()
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.0)

    state = F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.1, 1.0)
    dt = 0.01

    expected = F1Tenth.step_dynamics(state, action, params, track, dt)
    new_states, _, _ = F1Tenth.step_multi_agent!([state], [action], track, params, car, dt, cfg)

    @test new_states[1] == expected
end

@testset "step_multi_agent!: flags a car driven into the wall" begin
    track = circle_track()
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)
    action = F1Tenth.Action(0.0, 0.0)

    # Heading 0.6 rad off the centerline at 5 m/s: after 0.1 s the car has moved
    # ~0.28 m toward the inner wall (d: 0.6 -> 0.88) and its corners cross d = 1.
    state = F1Tenth.State(2.0, 0.6, 0.0, 5.0, 0.6, 0.0, 0.0)
    _, _, collisions = F1Tenth.step_multi_agent!([state], [action], track, params, car, 0.1, cfg)
    @test collisions[1]

    # The same car driving along the centerline is fine.
    centered = F1Tenth.State(2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    _, _, collisions = F1Tenth.step_multi_agent!([centered], [action], track, params, car, 0.1, cfg)
    @test !collisions[1]
end

@testset "step_multi_agent!: flags two cars driven into each other" begin
    track = circle_track()
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint(; length=0.58, width=0.31)
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=0.5, max_range=10.0, noise_std=0.0)

    states = [
        F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        F1Tenth.State(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    actions = [F1Tenth.Action(0.0, 0.0), F1Tenth.Action(0.0, 0.0)]

    _, scans, collisions = F1Tenth.step_multi_agent!(states, actions, track, params, car, 0.01, cfg)
    @test collisions == [true, true]
    @test length(scans) == 2
    @test length(scans[1]) == cfg.num_beams
end

@testset "step_multi_agent!: LiDAR noise is actually applied to scans" begin
    track = circle_track()
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.05)

    state = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    action = F1Tenth.Action(0.0, 0.0)

    _, scans1, _ = F1Tenth.step_multi_agent!([state], [action], track, params, car, 0.0, cfg;
        rng=MersenneTwister(1))
    _, scans2, _ = F1Tenth.step_multi_agent!([state], [action], track, params, car, 0.0, cfg;
        rng=MersenneTwister(2))
    _, scans_clean, _ = F1Tenth.step_multi_agent!([state], [action], track, params, car, 0.0,
        F1Tenth.LiDARConfig(; num_beams=5, fov=1.0, max_range=10.0, noise_std=0.0))

    @test scans1[1] != scans2[1]
    @test scans1[1] != scans_clean[1]
end
```

`test/async/testutils.jl` (full replacement):

```julia
include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

function _async_test_setup(; n_agents::Int=1)
    track = circle_track()
    params = F1Tenth.VehicleParams()
    car = F1Tenth.CarFootprint()
    cfg = F1Tenth.LiDARConfig(; num_beams=3, fov=1.0, max_range=10.0, noise_std=0.0)
    states = [F1Tenth.State(Float64(i - 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in 1:n_agents]
    actions = [F1Tenth.Action(0.0, 0.0) for _ in 1:n_agents]
    return (; track, params, car, cfg, states, actions)
end

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

Update `test/async/sim_test.jl`: every `setup.bvh` becomes `setup.track`, the "x position advanced" comment becomes "s (arc length) advanced", and the reset test switches to a different track so it can check the new field:

```bash
sed -i 's/setup\.bvh/setup.track/g; s/# x position advanced under constant forward velocity/# s (arc length) advanced under constant forward velocity/' test/async/sim_test.jl
```

then in the testset "AsyncSim: reset! requires the sim to be stopped, then clears state" replace

```julia
        F1Tenth.reset!(sim, setup.track, new_states)

        @test sim.states == new_states
```

with

```julia
        other_track = lumpy_track()
        F1Tenth.reset!(sim, other_track, new_states)

        @test sim.track === other_track   # the physics tier now uses the new map
        @test sim.states == new_states
```

Update `test/exports_test.jl`:

```diff
--- a/test/exports_test.jl
+++ b/test/exports_test.jl
@@ -4,8 +4,8 @@
 @testset "F1Tenth: public API is exported" begin
     exported_names = [
         :VehicleParams, :State, :Action, :step_dynamics, :simulate, :KS_MODE, :ST_MODE,
-        :Track, :Segment, :build_track, :load_track,
-        :TrackBVH, :build_bvh,
+        :Track, :Segment, :load_track, :track_point, :to_frenet, :to_cartesian, :world_pose,
+        :frenet_gap, :track_boundaries,
         :LiDARConfig, :CarFootprint, :car_vertices, :cast_ray, :scan,
         :collides, :check_car_collisions, :check_track_collision,
         :step_multi_agent!,
```

- [ ] **Step 2: Run them to verify they fail**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/step2_test.jl"); include("test/async/sim_test.jl"); include("test/exports_test.jl")'
```

Expected: ERROR (`step_multi_agent!`/`AsyncSim` still take a `TrackBVH`).

- [ ] **Step 3: Implement**

`src/geometry/step2.jl` (full replacement):

```julia
using Random: AbstractRNG, default_rng

function step_multi_agent!(states::Vector{State}, actions::Vector{Action},
    track::Track, params::VehicleParams, car::CarFootprint, dt::Float64, cfg::LiDARConfig;
    rng::AbstractRNG=default_rng())
    n = length(states)
    new_states = [step_dynamics(states[i], actions[i], params, track, dt) for i in 1:n]

    scans = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        others = [new_states[j] for j in 1:n if j != i]
        scans[i] = scan(new_states[i], track, others, car, cfg; rng=rng)
    end

    car_flags = check_car_collisions(new_states, track, car)
    collisions = [car_flags[i] || check_track_collision(new_states[i], track, car) for i in 1:n]

    return new_states, scans, collisions
end
```

`src/async/sim.jl`:

```diff
--- a/src/async/sim.jl
+++ b/src/async/sim.jl
@@ -42,7 +42,7 @@
 #   `sim.output` can raise `InvalidStateException` once a tier has
 #   failed (fail-fast closes every channel) or after `stop!` has run.
 mutable struct AsyncSim
-    bvh::TrackBVH
+    track::Track
     params::VehicleParams
     car::CarFootprint
     dt::Float64
@@ -61,7 +61,7 @@
     rng::AbstractRNG
 end
 
-function AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64,
+function AsyncSim(track::Track, params::VehicleParams, car::CarFootprint, dt::Float64,
     cfg::LiDARConfig, initial_states::Vector{State}, initial_actions::Vector{Action};
     tick_hz::Union{Float64,Nothing}=nothing, renderers::Vector{RendererEntry}=RendererEntry[],
     rng::AbstractRNG=default_rng())
@@ -69,7 +69,7 @@
     length(initial_actions) == n || throw(ArgumentError(
         "initial_actions must have one entry per agent (got $(length(initial_actions)) for $n agents)"))
     input = [LatestChannel{Action}() for _ in 1:n]
-    return AsyncSim(bvh, params, car, dt, cfg, tick_hz, renderers, input, copy(initial_actions),
+    return AsyncSim(track, params, car, dt, cfg, tick_hz, renderers, input, copy(initial_actions),
         copy(initial_states), LatestChannel{RenderFrame}(), Atomic{Bool}(false),
         nothing, Task[], Ref{Union{Nothing,Exception}}(nothing), ReentrantLock(), rng)
 end
@@ -102,7 +102,7 @@
         a === missing || (sim.last_actions[i] = a)
         actions[i] = sim.last_actions[i]
     end
-    new_states, scans, collisions = step_multi_agent!(sim.states, actions, sim.bvh,
+    new_states, scans, collisions = step_multi_agent!(sim.states, actions, sim.track,
         sim.params, sim.car, sim.dt, sim.cfg; rng=sim.rng)
     sim.states = new_states
     frame = (new_states, scans, collisions)
@@ -223,6 +223,7 @@
     for entry in sim.renderers
         reset!(entry.renderer, track)
     end
+    sim.track = track
     sim.states = copy(initial_states)
     fill!(sim.last_actions, zero(Action))
     take_latest!(sim.output)
```

`src/F1Tenth.jl`: delete the line `include("geometry/bvh.jl")` and delete the line `export TrackBVH, build_bvh`. Then:

```bash
git rm src/geometry/bvh.jl test/geometry/bvh_test.jl
julia --project=. -e 'using Pkg; Pkg.rm("NearestNeighbors")'
```

Expected: `Project.toml` loses the `NearestNeighbors` line under `[deps]` and `[compat]` (`Manifest.toml` is git-ignored).

- [ ] **Step 4: Run to verify they pass**

Same command as Step 2. Expected: all pass (the exports test passes only once `Track, Segment, load_track, track_point, to_frenet, to_cartesian, world_pose, frenet_gap, track_boundaries` are exported, as done in Task 3). The AsyncSim testsets run for real only with `-t 4`.

- [ ] **Step 5: Commit**

```bash
git add -A src test Project.toml
git commit -m "Run step_multi_agent! and AsyncSim on a Track; remove the KD-tree BVH"
```

---

### Task 8: Renderers, telemetry docs, LazySets export

**Files:**
- Modify: `src/render/glmakie.jl`, `src/render/cairomakie.jl`, `src/telemetry/sink.jl`, `test/render/glmakie_test.jl`, `test/telemetry/sink_test.jl`
- Rewrite: `src/geometry/lazysets_export.jl`, `test/geometry/lazysets_export_test.jl`, `test/render/testutils.jl`

**Interfaces:**
- Consumes: `track_boundaries`, `world_pose`, `car_vertices(pose, car)`, `to_cartesian` (Tasks 3, 5); `AsyncSim(track, ...)` (Task 7).
- Produces: renderers draw in the world frame (`GLMakieRendererImpl` gains a trailing field `track::Union{Nothing,Track}`); `to_lazysets(track::Track; n::Int=500)::Vector{LazySets.VPolygon}` (one quad per arc-length slice between consecutive left/right wall samples); `TelemetrySink` logs raw Frenet states (docstring updated).

- [ ] **Step 1: Update the tests**

`test/render/testutils.jl` (full replacement):

```julia
include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

small_test_track() = circle_track()
```

`test/geometry/lazysets_export_test.jl` (full replacement):

```julia
using F1Tenth
using StaticArrays
using LazySets: VPolygon, ∈
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "to_lazysets: one convex polygon per arc-length slice" begin
    track = lumpy_track()
    n = 200
    polys = F1Tenth.to_lazysets(track; n=n)

    @test length(polys) == n
    @test all(p -> p isa VPolygon, polys)

    # The centerline point halfway through each slice lies inside its polygon.
    for i in 1:n
        mid = F1Tenth.centerline(track, track.L * (i - 0.5) / n)
        @test mid ∈ polys[i]
    end
end
```

`test/render/glmakie_test.jl`:

```diff
--- a/test/render/glmakie_test.jl
+++ b/test/render/glmakie_test.jl
@@ -28,13 +28,15 @@
 
         @test length(renderer.car_polys) == 1
         @test length(renderer.car_polys[1][]) == 4
-        @test renderer.trail_points[1] == [Point2f(0.0, 0.0)]
+        # Trails are drawn in the world frame: Frenet (s, d) -> world (x, y).
+        world(s, d) = Point2f(to_cartesian(track, s, d)...)
+        @test renderer.trail_points[1] ≈ [world(0.0, 0.0)]
         @test renderer.car_colors[1][] == :blue
 
         states2 = [F1Tenth.State(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
         F1Tenth.render!(renderer, states2, scans1, [true])
 
-        @test renderer.trail_points[1] == [Point2f(0.0, 0.0), Point2f(1.0, 0.0)]
+        @test renderer.trail_points[1] ≈ [world(0.0, 0.0), world(1.0, 0.0)]
         @test renderer.car_colors[1][] == :red
 
         F1Tenth.reset!(renderer, track)
```

`test/telemetry/sink_test.jl`:

```diff
--- a/test/telemetry/sink_test.jl
+++ b/test/telemetry/sink_test.jl
@@ -1,19 +1,11 @@
 using F1Tenth
 using Arrow
-using StaticArrays
 using Test
 
 include(joinpath(@__DIR__, "..", "async", "testutils.jl"))
 
-function _small_test_track()
-    n = 8
-    R = 5.0
-    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
-    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
-end
-
 @testset "TelemetrySink: logs ticks across episodes and round-trips via Arrow" begin
-    track = _small_test_track()
+    track = circle_track()
     mktempdir() do dir
         path = joinpath(dir, "telemetry.arrow")
         sink = F1Tenth.TelemetrySink(path)
@@ -48,7 +40,7 @@
 end
 
 @testset "TelemetrySink: close! is idempotent" begin
-    track = _small_test_track()
+    track = circle_track()
     mktempdir() do dir
         sink = F1Tenth.TelemetrySink(joinpath(dir, "telemetry2.arrow"))
         F1Tenth.init!(sink, track)
@@ -70,7 +62,7 @@
             F1Tenth.init!(sink, setup.track)
 
             moving_states = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
-            sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
+            sim = F1Tenth.AsyncSim(setup.track, setup.params, setup.car, 0.01, setup.cfg,
                 moving_states, setup.actions;
                 renderers=[F1Tenth.RendererEntry(sink, F1Tenth.DropLatest())])
 
```

(`cairomakie_test.jl` and `null_test.jl` need no edits: they only pass Frenet-shaped states through and use `small_test_track()`.)

- [ ] **Step 2: Run them to verify they fail**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/geometry/lazysets_export_test.jl"); include("test/render/null_test.jl"); include("test/render/glmakie_test.jl"); include("test/render/cairomakie_test.jl"); include("test/telemetry/sink_test.jl")'
```

Expected: ERROR (`to_lazysets` still expects the polyline `Track`; renderers read `track.outer`). If there is no usable OpenGL context the GLMakie testsets are skipped with a warning; that is normal.

- [ ] **Step 3: Implement**

`src/geometry/lazysets_export.jl` (full replacement):

```julia
import LazySets

# One convex quad per arc-length slice: the corridor between consecutive
# left/right wall samples, `n` slices around the lap.
function to_lazysets(track::Track; n::Int=500)
    left, right = track_boundaries(track; n=n)
    return [LazySets.VPolygon([left[i], left[mod1(i + 1, n)],
                right[mod1(i + 1, n)], right[i]])
            for i in 1:n]
end
```

`src/render/glmakie.jl`:

```diff
--- a/src/render/glmakie.jl
+++ b/src/render/glmakie.jl
@@ -15,6 +15,7 @@
     scan_lines::Vector{Observable{Vector{Point2f}}}
     car_colors::Vector{Observable{Symbol}}
     initialized::Bool
+    track::Union{Nothing,Track}
 end
 
 function GLMakieRenderer(track::Track; camera_mode::Symbol=:fixed,
@@ -24,20 +25,22 @@
         throw(ArgumentError("camera_mode must be :fixed or :follow, got $camera_mode"))
     renderer = GLMakieRendererImpl(camera_mode, ego_index, car, cfg, nothing, nothing, nothing, 0.0,
         Observable{Vector{Point2f}}[], Observable{Vector{Point2f}}[], Vector{Point2f}[],
-        Observable{Vector{Point2f}}[], Observable{Symbol}[], false)
+        Observable{Vector{Point2f}}[], Observable{Symbol}[], false, nothing)
     init!(renderer, track)
     return renderer
 end
 
+# The renderer draws in the world frame: wall outlines are sampled from the
+# tck and each Frenet state is mapped through `world_pose`.
 function _plot_glmakie_track!(ax, track::Track)
-    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
-    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
-    lines!(ax, outer_pts; color=:black)
-    lines!(ax, inner_pts; color=:black)
+    left, right = track_boundaries(track)
+    lines!(ax, [Point2f(p[1], p[2]) for p in vcat(left, [left[1]])]; color=:black)
+    lines!(ax, [Point2f(p[1], p[2]) for p in vcat(right, [right[1]])]; color=:black)
 end
 
 function _glmakie_track_bounds(track::Track)
-    pts = vcat(track.outer, track.inner)
+    left, right = track_boundaries(track)
+    pts = vcat(left, right)
     xs = [p[1] for p in pts]
     ys = [p[2] for p in pts]
     return (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
@@ -52,6 +55,7 @@
     else
         empty!(renderer.ax)
     end
+    renderer.track = track
     _plot_glmakie_track!(renderer.ax, track)
     xmin, xmax, ymin, ymax = _glmakie_track_bounds(track)
     renderer.half_extent = max(xmax - xmin, ymax - ymin) / 2
@@ -87,8 +91,8 @@
     end
 end
 
-function _glmakie_scan_ray_points(state::State, beam_ranges::Vector{Float64}, cfg::LiDARConfig)
-    x, y, _, _, ψ, _, _ = state
+function _glmakie_scan_ray_points(pose::SVector{3,Float64}, beam_ranges::Vector{Float64}, cfg::LiDARConfig)
+    x, y, ψ = pose
     angles = cfg.num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=cfg.num_beams)
     points = Point2f[]
     for (angle, r) in zip(angles, beam_ranges)
@@ -105,17 +109,19 @@
     n = length(states)
     _ensure_glmakie_car_plots!(renderer, n)
     for i in 1:n
-        verts = car_vertices(states[i], renderer.car)
+        pose = world_pose(renderer.track, states[i])
+        verts = car_vertices(pose, renderer.car)
         renderer.car_polys[i][] = [Point2f(v[1], v[2]) for v in verts]
-        push!(renderer.trail_points[i], Point2f(states[i][1], states[i][2]))
+        push!(renderer.trail_points[i], Point2f(pose[1], pose[2]))
         renderer.trails[i][] = renderer.trail_points[i]
         renderer.car_colors[i][] = collisions[i] ? :red : :blue
-        renderer.scan_lines[i][] = _glmakie_scan_ray_points(states[i], scans[i], renderer.cfg)
+        renderer.scan_lines[i][] = _glmakie_scan_ray_points(pose, scans[i], renderer.cfg)
     end
     if renderer.camera_mode == :follow
         1 <= renderer.ego_index <= length(states) ||
             error("GLMakieRenderer: ego_index=$(renderer.ego_index) out of range for $(length(states)) agents")
-        ex, ey = states[renderer.ego_index][1], states[renderer.ego_index][2]
+        ego = world_pose(renderer.track, states[renderer.ego_index])
+        ex, ey = ego[1], ego[2]
         h = renderer.half_extent
         limits!(renderer.ax, ex - h, ex + h, ey - h, ey + h)
     end
```

`src/render/cairomakie.jl`:

```diff
--- a/src/render/cairomakie.jl
+++ b/src/render/cairomakie.jl
@@ -42,11 +42,12 @@
     return nothing
 end
 
+# World-frame drawing, as in glmakie.jl: walls sampled from the tck, states
+# mapped through `world_pose`.
 function _plot_cairomakie_track!(ax, track::Track)
-    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
-    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
-    lines!(ax, outer_pts; color=:black)
-    lines!(ax, inner_pts; color=:black)
+    left, right = track_boundaries(track)
+    lines!(ax, [Point2f(p[1], p[2]) for p in vcat(left, [left[1]])]; color=:black)
+    lines!(ax, [Point2f(p[1], p[2]) for p in vcat(right, [right[1]])]; color=:black)
 end
 
 function _draw_cairomakie_frame(track::Track, trails::Vector{Vector{Point2f}},
@@ -57,9 +58,10 @@
     _plot_cairomakie_track!(ax, track)
     for i in eachindex(states)
         isempty(trails[i]) || lines!(ax, trails[i]; color=:gray, linestyle=:dot)
-        verts = car_vertices(states[i], car)
+        pose = world_pose(track, states[i])
+        verts = car_vertices(pose, car)
         poly!(ax, [Point2f(v[1], v[2]) for v in verts]; color=collisions[i] ? :red : :blue)
-        x, y, _, _, ψ, _, _ = states[i]
+        x, y, ψ = pose
         num_beams = length(scans[i])
         angles = num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=num_beams)
         for (angle, r) in zip(angles, scans[i])
@@ -82,8 +84,8 @@
     else
         full_trails = [Point2f[] for _ in 1:n_cars]
         for tick in 1:n_ticks, i in 1:n_cars
-            s = renderer.history_states[tick][i]
-            push!(full_trails[i], Point2f(s[1], s[2]))
+            pose = world_pose(track, renderer.history_states[tick][i])
+            push!(full_trails[i], Point2f(pose[1], pose[2]))
         end
         summary_fig = _draw_cairomakie_frame(track, full_trails, renderer.history_states[end],
             renderer.history_scans[end], renderer.history_collisions[end], renderer.car, renderer.cfg)
@@ -94,8 +96,8 @@
         running_trails = [Point2f[] for _ in 1:n_cars]
         for tick in 1:n_ticks
             for i in 1:n_cars
-                s = renderer.history_states[tick][i]
-                push!(running_trails[i], Point2f(s[1], s[2]))
+                pose = world_pose(track, renderer.history_states[tick][i])
+                push!(running_trails[i], Point2f(pose[1], pose[2]))
             end
             frame_fig = _draw_cairomakie_frame(track, running_trails, renderer.history_states[tick],
                 renderer.history_scans[tick], renderer.history_collisions[tick], renderer.car, renderer.cfg)
```

`src/telemetry/sink.jl` (docstring only):

```diff
--- a/src/telemetry/sink.jl
+++ b/src/telemetry/sink.jl
@@ -15,6 +15,12 @@
 which -- once attached to an `AsyncSim` -- propagates and tears down the
 whole sim (physics tier included). See `test/telemetry/sink_test.jl` for an
 end-to-end example of the required `init!`-then-attach sequence.
+
+The logged `state` column holds the raw Frenet state
+`(s, d, δ, v, μ, ψ̇, β)`: `s` is unwrapped arc length along the track
+centerline, `d` the signed lateral offset (positive = left of travel) and
+`μ` the heading relative to the centerline tangent. Convert to world
+coordinates with `world_pose(track, state)`.
 """
 mutable struct TelemetrySink <: AbstractRenderer
     path::String
```

- [ ] **Step 4: Run to verify they pass**

Same command as Step 2. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src test
git commit -m "Draw renderers in the world frame; build corridor polygons from the tck"
```

---

### Task 9: Reachability in Frenet space

**Files:**
- Modify: `src/reachability/hybrid_system.jl`
- Rewrite: `src/reachability/analyze.jl`, `src/reachability/safety_check.jl`, `test/reachability/{hybrid_system,analyze,safety_check}_test.jl`

**Interfaces:**
- Consumes: `Track`, `track_point`, `Track.κw_max`, `frenet_rhs` (Tasks 3, 4).
- Produces: 10-D augmented state `(s, d, δ, v, μ, ψ̇, β, a, δ̇, κ)`; `build_hybrid_system(params)` (unchanged signature, now 10-D); `curvature_range(track, s_lo, s_hi)::(κ_min, κ_max)`; `halfwidth_min(track, s_lo, s_hi)::Float64`; `reachability_analyze(hs, initial_set::Zonotope, input_set::Zonotope, tspan, track::Track; alg=TMJets())` (**new `track` argument**; initial set is over the 7-D Frenet state, input set over `(a, δ̇)`); `check_flowpipe_safety(sol, track)::(Bool, Vector{Int})` (box test `|d| ≤ w_min` over each reach set's `s`-range).
- The `κ` dimension is a zero-derivative box over the reachable `s`-window. It is *not* a rigorous enclosure of the time-varying `κ(s(t))` (same caveat the existing `a`/`δ̇` augmentation carries); the docstrings say so.
- **World-age note:** `build_hybrid_system` `eval`s the `@taylorize`d mirrors, so tests must build `hs` at test top level and only then call the mirrors (the helper in `hybrid_system_test.jl` takes `hs` as an argument for this reason).

- [ ] **Step 1: Write the failing tests (full file replacements)**

`test/reachability/hybrid_system_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

# Augmented state: (s, d, δ, v, μ, ψ̇, β, a, δ̇, κ).

@testset "build_hybrid_system: automaton structure" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)

    @test nmodes(hs) == 2
    @test ntransitions(hs) == 2
end

# Cross-check each hand-written @taylorize mirror against `frenet_rhs` (which
# composes the validated Cartesian body dynamics with the Frenet pose
# kinematics) at several points on the track.
#
# `hs` must be built by the caller, at test top level: build_hybrid_system
# `eval`s the mirror functions, so calling them inside the same function that
# built them would hit Julia's world-age barrier.
function _mirror_matches_frenet_rhs(hs, params, mode_index, v)
    f! = mapping(mode(hs, mode_index))
    track = lumpy_track()
    for (s, d) in ((10.0, 0.3), (60.0, -0.5), (100.0, 0.7))
        κ = F1Tenth.curvature(track, s)
        a, δ̇ = 1.5, 0.2
        u = [s, d, 0.1, v, 0.05, 0.02, 0.01, a, δ̇, κ]
        du = zeros(10)
        f!(du, u, nothing, 0.0)

        state = F1Tenth.State(u[1], u[2], u[3], u[4], u[5], u[6], u[7])
        expected = F1Tenth.frenet_rhs(state, F1Tenth.Action(δ̇, a), params, track)
        du[1:7] ≈ collect(expected) || return false
        (du[8] == 0.0 && du[9] == 0.0 && du[10] == 0.0) || return false
    end
    return true
end

@testset "build_hybrid_system: :ks mode matches frenet_rhs" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    @test _mirror_matches_frenet_rhs(hs, params, 1, 0.3)   # v = 0.3 -> :ks regime
end

@testset "build_hybrid_system: :st mode matches frenet_rhs" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    @test _mirror_matches_frenet_rhs(hs, params, 2, 2.0)   # v = 2.0 -> :st regime
end
```

`test/reachability/analyze_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

# Initial sets are over the 7-D Frenet state (s, d, δ, v, μ, ψ̇, β); input sets
# over (a, δ̇). reachability_analyze appends the track-curvature interval.

@testset "curvature_range / halfwidth_min: window extremes" begin
    track = circle_track()
    κ_min, κ_max = F1Tenth.curvature_range(track, 2.0, 6.0)
    @test κ_min ≈ 0.2 atol=1e-4
    @test κ_max ≈ 0.2 atol=1e-4
    @test F1Tenth.halfwidth_min(track, 2.0, 6.0) ≈ 1.0 atol=1e-9

    lumpy = lumpy_track()
    lo, hi = F1Tenth.curvature_range(lumpy, 0.0, lumpy.L)
    @test 0 < lo < hi                                  # convex CCW loop: always turning left
    @test max(abs(lo), abs(hi)) ≈ lumpy.κ_max atol=1e-9
    # A window of a lap or more covers the whole track, however long.
    @test F1Tenth.curvature_range(lumpy, 5.0, 5.0 + 3lumpy.L) == F1Tenth.curvature_range(lumpy, 5.0, 5.0 + lumpy.L)
end

@testset "reachability_analyze: :st initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    initial_set = Zonotope([10.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1), track;
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
    @test length(sol.F[1]) > 0
    # The flowpipe lives in the 10-D augmented space.
    @test dim(set(sol.F[1][1])) == 10
end

@testset "reachability_analyze: :ks initial set solves without error" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    initial_set = Zonotope([10.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0], 0.001 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.001 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05), track;
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    @test !isempty(sol.F)
end

@testset "reachability_analyze: crosses the v=0.5 KS/ST mode boundary" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    # v0 ≈ 0.45, fully within the :ks regime (v_hi <= 0.5). a is state-augmented
    # with a zero derivative, so it stays ≈ 1.0 for the whole horizon (du[4] = a):
    # v(t) ≈ 0.45 + 1.0*t crosses 0.5 at t ≈ 0.05s, well inside a 0.15s tspan,
    # so the flowpipe should visit both the :ks and :st locations.
    initial_set = Zonotope([10.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0], 0.001 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([1.0, 0.0], 0.001 * Matrix{Float64}(I, 2, 2))

    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.15), track;
        alg=TMJets(abstol=1e-12, orderT=6, orderQ=1))

    @test length(sol.F) >= 2
end

@testset "reachability_analyze: initial_set straddling v=0.5 throws" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    initial_set = Zonotope([10.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], 0.3 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))

    @test_throws ArgumentError F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.1), track)
end
```

`test/reachability/safety_check_test.jl`:

```julia
using F1Tenth
using ReachabilityAnalysis
using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "fixtures", "fixtures.jl"))

@testset "check_flowpipe_safety: tight uncertainty inside the corridor is safe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    initial_set = Zonotope([40.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 0.01 * Matrix{Float64}(I, 7, 7))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05), track;
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test safe
    @test isempty(violating)
end

@testset "check_flowpipe_safety: lateral uncertainty wider than the corridor is unsafe" begin
    params = F1Tenth.VehicleParams()
    hs = F1Tenth.build_hybrid_system(params)
    track = lumpy_track()

    # ±5 m of lateral (d) uncertainty against a corridor about 1.1 m either side.
    initial_set = Zonotope([40.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        Diagonal([0.01, 5.0, 0.01, 0.01, 0.01, 0.01, 0.01]))
    input_set = Zonotope([0.0, 0.0], 0.01 * Matrix{Float64}(I, 2, 2))
    sol = F1Tenth.reachability_analyze(hs, initial_set, input_set, (0.0, 0.05), track;
        alg=TMJets(abstol=1e-10, orderT=5, orderQ=1))

    safe, violating = F1Tenth.check_flowpipe_safety(sol, track)
    @test !safe
    @test !isempty(violating)
end

@testset "check_flowpipe_safety: empty flowpipe throws ArgumentError" begin
    track = lumpy_track()
    fake_sol = (F=[],)
    @test_throws ArgumentError F1Tenth.check_flowpipe_safety(fake_sol, track)
end
```

- [ ] **Step 2: Run them to verify they fail**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/reachability/hybrid_system_test.jl")'
```

Expected: FAIL (mirrors are still 9-D Cartesian: `du[10]` out of bounds / mismatch).

- [ ] **Step 3: Implement**

`src/reachability/hybrid_system.jl` (10-D Frenet mirrors; the explanatory comments already in the file stay):

```diff
--- a/src/reachability/hybrid_system.jl
+++ b/src/reachability/hybrid_system.jl
@@ -1,6 +1,13 @@
 using ReachabilityAnalysis
 
-# Augmented state order shared by both modes: (x, y, δ, v, ψ, ψ̇, β, a, δ̇).
+# Augmented state order shared by both modes: (s, d, δ, v, μ, ψ̇, β, a, δ̇, κ),
+# i.e. the 7-D Frenet state (arc length, lateral offset, ..., heading error
+# relative to the track) followed by the control inputs and the track
+# curvature `κ`. `κ(s)` comes from a spline that `@taylorize` cannot evaluate,
+# so it is carried like `a`/`δ̇`: a zero-derivative dimension whose initial box
+# is the curvature range over the s-window the flowpipe can reach
+# (`reachability_analyze` computes it). Like `a`/`δ̇`, this is not a rigorous
+# enclosure of the true time-varying `κ(s(t))`.
 # `a`/`δ̇` (Phase 1's control inputs) are appended as extra
 # zero-derivative dimensions -- TMJets has no separate time-varying
 # input channel, so bounding them via the initial set is the standard
@@ -9,10 +16,11 @@
 # their initial box for the whole analysis horizon rather than varying
 # independently within it at every instant -- an accepted
 # over-approximation, not an under-approximation, so still sound.
-const _AUGMENTED_DIM = 9
+const _AUGMENTED_DIM = 10
 
-# These two functions are hand-written mirrors of `_ks_eqs`/`_st_eqs`
-# in src/dynamics/mtk_models.jl -- NOT calls into KS_RHS/ST_RHS/
+# These two functions are hand-written Frenet-frame mirrors of `frenet_rhs`
+# (src/dynamics/frenet.jl), whose body dynamics come from `_ks_eqs`/`_st_eqs`
+# in src/dynamics/mtk_models.jl -- NOT calls into frenet_rhs/KS_RHS/ST_RHS/
 # KS_MODE/ST_MODE. See this file's F1Tenth.jl-level comment header (and
 # the spec's amended Decision 1) for why: ReachabilityAnalysis.jl's
 # `@taylorize` is a macro operating on function *source code*, and
@@ -20,19 +28,20 @@
 # build_function-generated Expr -- there is no source form for
 # `@taylorize` to macroexpand. Keeping these in sync with
 # mtk_models.jl's equations by hand is a real, accepted maintenance
-# cost; this task's own tests (above) cross-check both mirrors against
-# KS_RHS/ST_RHS at sample points to catch silent divergence.
+# cost; test/reachability/hybrid_system_test.jl cross-checks both mirrors
+# against `frenet_rhs` at sample points to catch silent divergence.
 function _ks_taylor_expr(fname::Symbol, params::VehicleParams)
     lf = params.lᶠ
     lr = params.lʳ
     return quote
         @taylorize function $(fname)(du, u, p, t)
-            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
-            du[1] = v * cos(ψ)
-            du[2] = v * sin(ψ)
+            s, d, δ, v, μ, ψ̇, β, a, δ̇, κ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10]
+            ṡ = v * cos(μ) / (1 - κ * d)
+            du[1] = ṡ
+            du[2] = v * sin(μ)
             du[3] = δ̇
             du[4] = a
-            du[5] = v / ($lf + $lr) * tan(δ)
+            du[5] = v / ($lf + $lr) * tan(δ) - κ * ṡ
             # Matches step.jl's corrected KS ψ̈ term (dynamics_rhs's fallback
             # branch), NOT KS_RHS's own zeroed simplification -- see the
             # Fix 1 discussion above: a flowpipe that starts in :ks and
@@ -42,6 +51,7 @@
             du[7] = zero(v)
             du[8] = zero(v)
             du[9] = zero(v)
+            du[10] = zero(v)
             return du
         end
     end
@@ -59,12 +69,13 @@
     g = _MTK_G  # reuse mtk_models.jl's gravity constant rather than a second hardcoded copy
     return quote
         @taylorize function $(fname)(du, u, p, t)
-            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
-            du[1] = v * cos(β + ψ)
-            du[2] = v * sin(β + ψ)
+            s, d, δ, v, μ, ψ̇, β, a, δ̇, κ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10]
+            ṡ = v * cos(β + μ) / (1 - κ * d)
+            du[1] = ṡ
+            du[2] = v * sin(β + μ)
             du[3] = δ̇
             du[4] = a
-            du[5] = ψ̇
+            du[5] = ψ̇ - κ * ṡ
             du[6] = (-$μ * $m / (v * $Iz * ($lr + $lf)) *
                         ($lf^2 * $Cbf * ($g * $lr - a * $h) +
                          $lr^2 * $Cbr * ($g * $lf + a * $h)) * ψ̇ +
@@ -83,6 +94,7 @@
                          ($Cbf * ($g * $lr - a * $h)) * δ)
             du[8] = zero(v)
             du[9] = zero(v)
+            du[10] = zero(v)
             return du
         end
     end
@@ -108,23 +120,24 @@
 """
     build_hybrid_system(params::VehicleParams) -> ReachabilityAnalysis.HybridSystem
 
-Build a 2-location hybrid automaton over the 9-D augmented state
-`(x, y, δ, v, ψ, ψ̇, β, a, δ̇)`, switching on the longitudinal velocity `v`
+Build a 2-location hybrid automaton over the 10-D augmented Frenet state
+`(s, d, δ, v, μ, ψ̇, β, a, δ̇, κ)`, switching on the longitudinal velocity `v`
 (index 4) at the KS/ST threshold `v = 0.5`:
 
 - location `1` = `:ks` (kinematic single-track, active for `v <= 0.5`)
 - location `2` = `:st` (dynamic single-track, active for `v >= 0.5`)
 
-`a`/`δ̇` are carried as extra zero-derivative dimensions (state-augmented
-control inputs); see the `_AUGMENTED_DIM` comment above for why.
+`a`/`δ̇`/`κ` are carried as extra zero-derivative dimensions (state-augmented
+control inputs and track curvature); see the `_AUGMENTED_DIM` comment above for
+why.
 """
 function build_hybrid_system(params::VehicleParams)
     ks_name = gensym(:phase5_ks)
     st_name = gensym(:phase5_st)
 
-    # v is index 4 of the 9-D augmented state.
-    v_ks_invariant = HalfSpace([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5)   # v <= 0.5
-    v_st_invariant = HalfSpace([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.5) # v >= 0.5
+    # v is index 4 of the 10-D augmented state.
+    v_ks_invariant = HalfSpace([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5)   # v <= 0.5
+    v_st_invariant = HalfSpace([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.5) # v >= 0.5
 
     ks_mode = _build_mode(ks_name, _ks_taylor_expr(ks_name, params), v_ks_invariant, _AUGMENTED_DIM)
     st_mode = _build_mode(st_name, _st_taylor_expr(st_name, params), v_st_invariant, _AUGMENTED_DIM)
```

`src/reachability/analyze.jl` (full replacement):

```julia
using ReachabilityAnalysis

"""
    curvature_range(track, s_lo, s_hi) -> (κ_min, κ_max)

Extremes of the signed curvature over arc length `[s_lo, s_hi]` (unwrapped; a
window of a full lap or more covers the whole track), sampled every ≤ 0.05 m.
"""
function curvature_range(track::Track, s_lo::Float64, s_hi::Float64)
    s_hi - s_lo >= track.L && (s_hi = s_lo + track.L)
    n = max(2, ceil(Int, (s_hi - s_lo) / 0.05) + 1)
    κ_min, κ_max = Inf, -Inf
    for s in range(s_lo, s_hi; length=n)
        κ = track_point(track, s).κ
        κ_min, κ_max = min(κ_min, κ), max(κ_max, κ)
    end
    return κ_min, κ_max
end

"""Minimum half-width over arc length `[s_lo, s_hi]`, sampled every ≤ 0.05 m."""
function halfwidth_min(track::Track, s_lo::Float64, s_hi::Float64)
    s_hi - s_lo >= track.L && (s_hi = s_lo + track.L)
    n = max(2, ceil(Int, (s_hi - s_lo) / 0.05) + 1)
    return minimum(track_point(track, s).w for s in range(s_lo, s_hi; length=n))
end

"""
    reachability_analyze(hs, initial_set, input_set, tspan, track; alg=TMJets())

Flowpipe of the Frenet hybrid system `hs` from `initial_set` (a `Zonotope` over
the 7-D Frenet state `(s, d, δ, v, μ, ψ̇, β)`, entirely inside one KS/ST mode)
with bounded inputs `input_set` (over `(a, δ̇)`).

The 10th dimension of the analyzed state is the track curvature `κ`, boxed to
its range over the arc-length window the flowpipe can reach (on track,
`ṡ ≤ v / (1 − κw_max)`). Like the `a`/`δ̇` augmentation this holds `κ` constant
at some value in that box, which is not a rigorous enclosure of the true
time-varying `κ(s(t))`. The box widens on winding stretches, so shorten `tspan`
(run per window) where tighter results are needed.
"""
function reachability_analyze(hs, initial_set::Zonotope, input_set::Zonotope,
    tspan::Tuple{Float64,Float64}, track::Track; alg=TMJets())
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

    # κ window: the arc length the flowpipe can reach. On track, ṡ ≤ v / (1 − κw_max).
    T = tspan[2] - tspan[1]
    scale = 1 / (1 - track.κw_max)
    a_lo, a_hi = low(input_set)[1], high(input_set)[1]
    fwd = max(v_hi + max(a_hi, 0.0) * T, 0.0) * T * scale
    bwd = max(-(v_lo + min(a_lo, 0.0) * T), 0.0) * T * scale
    s_lo = low(initial_set)[1] - bwd
    s_hi = high(initial_set)[1] + fwd
    κ_min, κ_max = curvature_range(track, s_lo, s_hi)
    κ_set = Zonotope([(κ_min + κ_max) / 2], reshape([(κ_max - κ_min) / 2], 1, 1))

    # Nested lazy products don't convert in one step; do it pairwise.
    augmented = convert(Zonotope, convert(Zonotope, initial_set × input_set) × κ_set)
    prob = InitialValueProblem(hs, [(location, augmented)])
    return solve(prob; tspan=tspan, alg=alg, max_jumps=4)
end
```

`src/reachability/safety_check.jl` (full replacement):

```julia
using ReachabilityAnalysis

function check_flowpipe_safety(sol, track::Track)
    isempty(sol.F) &&
        throw(ArgumentError("check_flowpipe_safety: sol.F is empty -- no flowpipe segments to check"))
    violating = Int[]
    idx = 0
    for fp in sol.F
        for rs in fp
            idx += 1
            box = overapproximate(set(rs), Hyperrectangle)
            s_lo, s_hi = low(box)[1], high(box)[1]
            d_lo, d_hi = low(box)[2], high(box)[2]
            w = halfwidth_min(track, s_lo, s_hi)
            (d_lo >= -w && d_hi <= w) || push!(violating, idx)
        end
    end
    return (isempty(violating), violating)
end
```

- [ ] **Step 4: Run to verify they pass**

```bash
julia --project=. -t 4 -e 'using Test, F1Tenth; include("test/reachability/hybrid_system_test.jl"); include("test/reachability/analyze_test.jl"); include("test/reachability/safety_check_test.jl")'
```

Expected: all pass. Two of the flowpipe solves take about a minute each (Taylor-model compilation dominates the first call), and `ReachabilityAnalysis` prints many `Warning: basis matrix is ill-conditioned` lines; both are normal and were also seen before this change.

- [ ] **Step 5: Commit**

```bash
git add src/reachability test/reachability
git commit -m "Move reachability to Frenet space with a boxed curvature dimension"
```

---

### Task 10: Demo, docs, full suite

**Files:**
- Modify: `../main.jl` (outside the git repo), `test/runtests.jl`, `docs/HANDOFF.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a Frenet pure-pursuit demo that loads a tck CSV (`F1TENTH_TCK`, default `Austin_tck.csv` next to `main.jl` if present else the lumpy fixture); a green full suite.

- [ ] **Step 1: Replace `runtests.jl`**

`test/runtests.jl`:

```julia
using F1Tenth
using Test

@testset "F1Tenth.jl" begin
    include("params_test.jl")
    include("mtk_models_test.jl")
    include("step_test.jl")
    include("solve_test.jl")
    include("geometry/tck_test.jl")
    include("geometry/track_test.jl")
    include("frenet_test.jl")
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
    include("reachability/analyze_test.jl")
    include("reachability/safety_check_test.jl")
    include("telemetry/sink_test.jl")
    include("exports_test.jl")
end
```

- [ ] **Step 2: Run the full suite**

```bash
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
```

Expected: `F1Tenth tests passed` with 3504 passing assertions (about 5 minutes; reachability dominates). Any failure here is a regression from a task above, so fix it in the task's own files rather than loosening a test.

- [ ] **Step 3: Rewrite the demo**

`/home/pganguli/Downloads/f1tenth/main.jl` (full replacement; outside the git repo, do not commit):

```julia
# Pure-pursuit racetrack demo for F1Tenth.jl.
#
# Loads a closed racetrack stored as a tck spline (see tck.py), then drives one
# car around it in Frenet space with a pure-pursuit steering controller and a
# curvature-limited speed profile, drawing the run live in a GLMakie window.
#
#   julia main.jl                      # live GLMakie window
#   F1TENTH_HEADLESS=1 julia main.jl   # same sim, NullRenderer (no window)
#
# Optional env vars:
#   F1TENTH_TCK       tck CSV to drive (default: Austin_tck.csv next to this
#                     file if present, else F1Tenth.jl's lumpy test track)
#   F1TENTH_LAPS      laps to drive (default 2)
#   F1TENTH_REALTIME  1 (default) paces to wall-clock; 0 runs as fast as possible

using Pkg
Pkg.activate(joinpath(@__DIR__, "F1Tenth.jl"); io=devnull)

using F1Tenth
using StaticArrays

# ----------------------------------------------------------------------------
# Track
# ----------------------------------------------------------------------------

function default_track_path()
    haskey(ENV, "F1TENTH_TCK") && return ENV["F1TENTH_TCK"]
    austin = joinpath(@__DIR__, "Austin_tck.csv")
    return isfile(austin) ? austin : joinpath(@__DIR__, "F1Tenth.jl", "test", "fixtures", "lumpy_tck.csv")
end

# ----------------------------------------------------------------------------
# Pure pursuit, in Frenet space: the car's arc length `s` is part of its state,
# so there is no nearest-waypoint search and lookahead is just `s + L`.
# ----------------------------------------------------------------------------

Base.@kwdef struct PurePursuit
    track::Track
    wheelbase::Float64
    rear_offset::Float64            # CG -> rear axle distance
    lookahead_min::Float64 = 0.9    # m
    lookahead_gain::Float64 = 0.3   # s: lookahead grows with speed
    v_max::Float64 = 7.0            # m/s
    a_lat::Float64 = 5.0            # m/s², lateral accel used for the speed profile
    speed_horizon::Float64 = 10.0   # m of track scanned ahead for the sharpest turn
    kδ::Float64 = 10.0              # steering-angle P gain -> steering rate
    kv::Float64 = 4.0               # speed P gain -> acceleration
end

"""One control update for Frenet state `𝐱`: returns the `Action`."""
function control(pp::PurePursuit, 𝐱::State)
    s, _, δ, v, _, _, _ = 𝐱
    x, y, ψ = world_pose(pp.track, 𝐱)
    # Pure pursuit is defined about the rear axle.
    rear = SVector(x - pp.rear_offset * cos(ψ), y - pp.rear_offset * sin(ψ))

    L = max(pp.lookahead_min, pp.lookahead_gain * abs(v))
    to_target = F1Tenth.centerline(pp.track, s + L) - rear
    α = atan(to_target[2], to_target[1]) - ψ
    δ_des = atan(2 * pp.wheelbase * sin(α), L)

    κ_ahead = maximum(abs(F1Tenth.curvature(pp.track, s + k)) for k in 0:0.4:pp.speed_horizon)
    v_des = min(pp.v_max, sqrt(pp.a_lat / max(κ_ahead, 1e-3)))

    # The plant takes (steering rate, acceleration): close the loop on both.
    return Action(pp.kδ * (δ_des - δ), pp.kv * (v_des - v))
end

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

function main()
    headless = get(ENV, "F1TENTH_HEADLESS", "0") == "1"
    laps_goal = parse(Int, get(ENV, "F1TENTH_LAPS", "2"))
    realtime = !headless && get(ENV, "F1TENTH_REALTIME", "1") == "1"

    dt = 0.01                # physics step, s
    render_every = 2         # sim-time render rate: 1 / (dt * render_every) = 50 Hz

    params = VehicleParams()
    car = CarFootprint()
    cfg = LiDARConfig(num_beams=25, fov=Float64(π), max_range=30.0, noise_std=0.0)

    path = default_track_path()
    track = load_track(path)
    println("Track $(basename(path)): length $(round(track.L; digits=1)) m, ",
        "min turn radius $(round(1 / track.κ_max; digits=2)) m, ",
        "half-width $(round(track.w_min; digits=2))–$(round(track.w_max; digits=2)) m")

    pp = PurePursuit(; track=track, wheelbase=params.lᶠ + params.lʳ, rear_offset=params.lʳ)

    # Start on the centerline at s = 0, heading along it (μ = 0).
    𝐱 = State(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0)

    renderer = headless ? NullRenderer() : GLMakieRenderer(track; cfg=cfg, car=car)

    lap, crashed = 0, false
    t_wall = time()
    step_count = 0
    lap_start = 0.0
    while lap < laps_goal && step_count * dt < 300
        action = control(pp, 𝐱)

        states, scans, collisions = step_multi_agent!([𝐱], [action], track, params, car, dt, cfg)
        𝐱 = states[1]
        step_count += 1
        t = step_count * dt

        if floor(Int, 𝐱[1] / track.L) > lap      # unwrapped s: a lap is s ≥ L
            lap = floor(Int, 𝐱[1] / track.L)
            println("lap $lap done: $(round(t - lap_start; digits=2)) s")
            lap_start = t
        end

        if step_count % render_every == 0
            render!(renderer, states, scans, collisions)
            # Pace to wall-clock so the animation plays in real time.
            realtime && sleep(max(0.0, t_wall + t - time()))
        end

        if collisions[1]
            crashed = true
            render!(renderer, states, scans, collisions)
            x, y, _ = world_pose(track, 𝐱)
            println("collision at t=$(round(t; digits=2)) s, s=$(round(𝐱[1]; digits=2)) m, ",
                "d=$(round(𝐱[2]; digits=2)) m, position ($(round(x; digits=2)), $(round(y; digits=2)))")
            break
        end
    end

    println(crashed ? "Run ended in a collision." :
            lap >= laps_goal ? "Completed $lap laps in $(round(step_count * dt; digits=2)) s." :
            "Time limit reached after $lap laps.")

    if !headless
        print("Press Enter to close the window...")
        readline()
    end
    close!(renderer)
    return nothing
end

main()
```

- [ ] **Step 4: Run the demo headless**

```bash
cd /home/pganguli/Downloads/f1tenth && F1TENTH_HEADLESS=1 F1TENTH_LAPS=2 julia main.jl
```

Expected (lumpy track): `Track lumpy_tck.csv: length 115.8 m, min turn radius 3.58 m, half-width 0.95–1.25 m`, then `lap 1 done: 18.72 s`, `lap 2 done: 18.6 s`, `Completed 2 laps in 37.32 s.` and no collision line. If `Austin_tck.csv` exists and is too tight, the run stops at load with the located `|κ|·w ≥ 0.9` message (amendment 7); that is the intended behavior, not a bug.

- [ ] **Step 5: Update `docs/HANDOFF.md`**

At the top of the file, directly under the title's "Written for" paragraph, add:

```markdown
> **Update (2026-09-21): the simulator now runs in Frenet space on tck maps.**
> Phases 1–5 below describe the Cartesian/polyline design they were built
> with; where they conflict with the section "Frenet + tck migration" at the
> end, that section wins. Notably superseded: `Track` (polyline), `TrackBVH`,
> `build_track`, `build_bvh`, Cartesian `State`, and the 9-D Cartesian
> reachability mirrors.
```

and append at the end of the file:

```markdown
## Frenet + tck migration (post-Phase 5)

Spec: `docs/superpowers/specs/2026-09-21-frenet-tck-design.md` (with its
"Implementation-time amendments"); plan: `docs/superpowers/plans/2026-09-21-frenet-tck.md`.

### What changed

- **Map:** `Track` wraps a periodic cubic B-spline over `[x, y, w]` (the tck
  `tck.py` fits from a raster, saved with `save_tck_csv`) plus an arc-length
  table. `load_track(path)` reads the CSV (`# k=3`, `t,cx,cy,cw`). There is no
  polyline, KD-tree or `TrackBVH` any more; `NearestNeighbors` is gone.
- **State:** `(s, d, δ, v, μ, ψ̇, β)`: `s` unwrapped arc length (a lap is
  `⌊s/L⌋`), `d` signed lateral offset (`+` = left), `μ` heading relative to
  the centerline, wrapped to `(−π, π]`. `world_pose(track, 𝐱)` gives `(x, y, ψ)`.
- **Dynamics:** `step_dynamics(𝐱, action, params, track, dt)` is Frenet RK4.
  `frenet_rhs` evaluates the Cartesian `dynamics_rhs` at pose `(0, 0, μ)` and
  changes only the pose kinematics (`ṡ = v·cos(μ+β)/(1−κd)`, `ḋ = v·sin(μ+β)`,
  `μ̇ = ψ̇ − κṡ`). The Cartesian step survives as `step_dynamics_cartesian`
  (reference model + Python-gym parity tests).
- **Sensing:** walls are the implicit set `|d| ≤ w(s)`. Collision projects 8
  footprint points (4 corners + 4 edge midpoints) with `to_frenet`; LiDAR
  sphere-traces the boundary (step `m/Λ`, `Λ = track.lipschitz`) and bisects.
  Car-car overlap is still SAT in the world frame.
- **Reachability:** 10-D state `(s, d, δ, v, μ, ψ̇, β, a, δ̇, κ)`; `κ` is a
  zero-derivative box over the reachable s-window (not a rigorous enclosure of
  `κ(s(t))`, same caveat as `a`/`δ̇`). Safety is `|d| ≤ w_min` over each reach
  set's s-range.
- **Renderers** draw in the world frame; **telemetry** logs raw Frenet states.

### Load-bearing facts a fresh agent must not re-derive

1. **`|κ|·w < 0.9` is enforced at load, deliberately.** Beyond it the Frenet
   frame is singular (center of curvature inside the corridor). The user chose
   rejection over clipping the inside width. `Austin_map.png` converted with
   the current `tck.py` (`smoothing=0.001`) violates it (`|κ|·w` up to 1.42,
   ~1% of the lap); `smoothing` sweeps did not reliably fix it. The map must
   be re-fit before Austin can be loaded.
2. **`to_frenet` never throws.** Failure to converge falls back to a table-scan
   seed, then the last iterate. Do not reintroduce a throw: it would kill the
   physics thread on a crashed car.
3. **LiDAR budget:** 1080 beams ≈ 3.3 ms single-thread on the lumpy track
   (gate 5 ms, `benchmarks/lidar_scan.jl`). The circle/lumpy fixtures are cubic
   *interpolants*, so their curvature is only ~4e-5 accurate; tests use `1e-4`.
4. **`angle`/`s_hint` arguments are `Float64`-typed** (`cast_ray`); pass
   `Float64(π)`, not `π`.
5. **Reachability tests must build `hs` at test top level** (world age: the
   mirrors are `eval`ed inside `build_hybrid_system`).
6. **Fixtures:** `test/fixtures/make_tck_fixtures.py` (numpy + scipy, run from
   `.venv`) regenerates them; `save_tck_csv` there mirrors `tck.py`'s.

### Known deferred minors

- `w_left = w_right`; the tck carries one half-width.
- No fold-over warning: on a track whose sections come closer than a corridor
  width, `to_frenet` from a poor hint could lock onto the wrong branch.
- No knot-span hinting in de Boor (binary search); no beam threading in `scan`.
- The `κ` box widens on winding stretches; shorten `tspan` per window.
- `check_flowpipe_safety` uses the minimum half-width over each reach set's
  s-range (conservative).
```

- [ ] **Step 6: Commit**

```bash
git add test/runtests.jl docs/HANDOFF.md
git commit -m "Update runtests and handoff for the Frenet/tck migration"
```

---

## Self-Review

**Spec coverage** (spec section → task): §1 Map/CSV/`Track`/queries/validation/boundaries/deletions → Tasks 1–3, 7; §2 State and dynamics (unwrapped `s`, `μ`, RK4, guard, Cartesian reference) → Task 4; §3 wall collision, car-car, LiDAR, API, rendering, telemetry → Tasks 5–8; §4 reachability (10-D, ODEs, κ window, safety, validation) → Task 9; Error handling → Tasks 2, 3 (malformed CSV, validation), 3/6 (never-throw); Testing list → tests in Tasks 2–9 (scipy fixture, circle analytics, arc-length, equivalence, wall vs brute force via polyline comparison in LiDAR test + collision cases, sphere-trace brute-force, reachability); Benchmark gate → Task 5; Demo → Task 10; Risks (κ enclosure, sphere cost, fold-over, CSV format) → docstrings/Task 5/amendments. **Gaps knowingly closed differently:** the fold-over self-distance warning (spec Error handling/Risks) is *not* implemented (amendment 7, deferred minor); the wall-collision "agrees with brute-force polyline distance on random poses" test is covered by the footprint-sample cases plus the LiDAR brute-force comparison rather than a separate random-pose test.

**Placeholders:** none; every code step embeds the verified file content or a verified diff.

**Type consistency:** `world_pose(track, 𝐱)::SVector{3}` (Task 3) is consumed unchanged by Tasks 5–8; `car_vertices(pose::SVector{3,Float64}, car)` (Task 5) is consumed by Tasks 6, 8; `cast_ray(track, origin, angle, s_hint, max_range)` (Task 5) is called only from `scan`; `check_track_collision(state, track, car)` / `check_car_collisions(states, track, car)` (Task 6) are called by `step_multi_agent!` (Task 7); `AsyncSim(track, ...)` (Task 7) matches the Task 7/8 test setups; `Track.κw_max` (Task 3) is read by Task 9.
