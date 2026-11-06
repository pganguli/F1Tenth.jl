# F1Tenth.jl

[![CI](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://pganguli.github.io/F1Tenth.jl/dev/)

A Julia port of the Python [`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym)
simulator, with the same features and nothing more. For the same inputs it
produces the same outputs: car poses, LiDAR scans, collisions, lap counts and
lap times agree with the Python code to about 1e-14 over whole races. It runs
10–20× faster.

## Install

```julia
import Pkg
Pkg.add(url="https://github.com/pganguli/F1Tenth.jl")
Pkg.add("GLMakie")   # only needed for the render window
```

## Use

The API follows the Python env, with Julia's `!` for functions that change
state and 1-based car indices:

```julia
using F1Tenth

env = F110Env(map="berlin", num_agents=2)          # gym.make('f110_gym:f110-v0', ...)
obs, reward, done, info = reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])   # rows: x, y, θ
while !done
    action = [0.1 2.0; 0.0 1.5]                     # rows: steering angle, speed
    obs, reward, done, info = step!(env, action)
end
obs.poses_x, obs[:scans], info.checkpoint_done
```

`F110Env` takes Python's keyword arguments: `seed`, `map` (`"berlin"`,
`"skirk"`, `"levine"`, `"vegas"`, a real-world track such as `"Austin"` or
`"Spa"`, or a yaml path without its extension),
`map_ext`, `params` (a `VehicleParams` or a dict with every key),
`num_agents`, `timestep`, `ego_idx`, `integrator` (`Integrator.RK4` or
`Integrator.Euler`) and `lidar_dist`.

The 23 real-world tracks of
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) work by
name (`keys(F1Tenth.RACETRACKS)`). They are GPL-3.0 licensed, so they are not
bundled: Pkg downloads them (about 6 MB) the first time one is used. Any
other ROS map (yaml + image) works by path.

Also as in Python: `update_map!(env, path, ext)`,
`update_params!(env, params; index=-1)`, `add_render_callback!(env, f)`, and
the lower layers (`Simulator`, `RaceCar`, `ScanSimulator2D`,
`vehicle_dynamics_st`, `pid`, `collision`, `get_vertices`, …).

**Observations are reused:** each `step!` overwrites the arrays in `obs` and
`info`, so stepping allocates nothing. Use `copy(obs)` to keep one.

### Rendering

```julia
using GLMakie
render!(env)                    # or render!(env; mode=:human_fast)
```

Drag to pan, scroll to zoom. Render callbacks get an `EnvRenderer` with
Python's camera fields (`left`, `right`, `bottom`, `top`, in units of
50 × meters), `cars` and `score_label`; draw extra things on its `axis` with
GLMakie.

### Example

[examples/waypoint_follow.jl](examples/waypoint_follow.jl) is the Python
pure-pursuit example:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.add("GLMakie"); Pkg.instantiate()'
julia --project=examples examples/waypoint_follow.jl
```

## Performance

`step!` time on an AMD Ryzen 5 7640U (example map, LiDAR noise on):

| cars | Python (numba) | Julia, 1 thread | Julia, 4 threads |
|---|---|---|---|
| 1 | 166 µs | 17 µs | 17 µs |
| 2 | 426 µs | 34 µs | 30 µs |
| 4 | 1084 µs | 81 µs | 51 µs |

With several Julia threads (`julia -t auto`), each car is stepped on its own
thread; set `F1Tenth.THREAD_AGENTS[] = false` when you run many envs in
parallel yourself.

## Differences from Python

- Indices are 1-based (`ego_idx`, `update_params!(…; index)`).
- LiDAR noise has the same distribution, but comes from Julia's random number
  generator instead of numpy's.
- Observations are reused between steps (see above).
- Inputs that crash the Python code work: `map="vegas"` by name, `"levine"`
  without `map_ext=".pgm"`, `map_ext` without the dot.
- Python keeps the map and renderer in class attributes shared by every env
  in a process; here each env has its own.
- The render window uses GLMakie instead of pyglet: callbacks draw on
  `renderer.axis` instead of adding to `batch`. It keeps pixels square (a
  camera area that doesn't match the window's shape is widened, not
  stretched) and scales up on HiDPI monitors; set
  `GLMakie.activate!(scalefactor=...)` to override.
- The f1tenth_racetracks maps load by name (Python has no equivalent).
- Choosing `Integrator.RK4` doesn't print Python's "different from previous
  versions of the gym" warning.

Python's quirks are kept where they change outputs: `obs.ego_idx` is always
1, a wall hit zeroes the car's yaw as well as its speed, and car-to-car
collisions keep the body size given at construction.

## Credits

A port of [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) by Hongrui
Zheng, Joseph Auckley, Matthew O'Kelly and Aman Sinha (MIT license), whose
vehicle models follow the [CommonRoad](https://commonroad.in.tum.de/) single-track
models. The bundled maps come from f1tenth_gym; see [maps/LICENSE](maps/LICENSE).
The racetrack maps are downloaded from
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) (GPL-3.0).

## Tests

`julia --project=. -e 'import Pkg; Pkg.test()'` checks every layer against
outputs frozen from the Python code (`test/reference/`, regenerated by
`test/reference/generate.py`), and runs the Python unit tests' own checks.
