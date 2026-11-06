# F1Tenth.jl

[![CI](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://pganguli.github.io/F1Tenth.jl/dev/)

A Julia port of the Python [`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym)
simulator, with the same features and nothing more. For the same inputs it
produces the same outputs: car poses, LiDAR scans, collisions, lap counts and
lap times agree with the Python code to about 1e-14 over whole races. It runs
10–20× faster.

> **This branch (`realistic-dynamics`) departs from Python on purpose.** It
> simulates one real F1TENTH car, ForzaETH's NUC5, from published values:
> Magic Formula tires with a friction ellipse, and the car's actual
> low-level actuator chain (command limiter, steering servo, VESC speed
> loop) in place of Python's controller (see
> [Vehicle model changes](#vehicle-model-changes)). Results no longer match
> Python step for step.

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
`vehicle_dynamics_st`, `collision`, `get_vertices`, …).

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

## Vehicle model changes

**Tires.** Python's dynamic single-track model has linear tires: lateral
force grows with slip angle forever, so the car never runs out of grip.
`vehicle_dynamics_st` keeps the same state and single-track equations but
gives each axle a Magic Formula tire (Pacejka, *Tire and Vehicle Dynamics*),
in the form ForzaETH identify and simulate with:

```text
Fy = mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))
```

The rest of the model:

- **Load transfer.** As in CommonRoad, the axle loads `Fz` shift with
  longitudinal acceleration.
- **Drivetrain.** F1TENTH chassis are 4WD with an open center differential,
  which sends equal torque to both axles. Each axle gets half the drive force,
  limited by the grip of the weaker axle.
- **Friction ellipse.** Grip the drive force uses is taken from cornering:
  `Fy` is scaled by `sqrt(1 - (Fx / (mu*D*Fz))^2)`. This is the friction
  ellipse approximation of combined slip (Pacejka). With no longitudinal tire
  data, the peak force is assumed to be the same in every direction.
- **Body accelerations.** The force balance is ForzaETH's single-track model
  (their EKF and their f1tenth_gym fork), with the front drive force added.
  With no drive force it matches their `vehicle_dynamics_st_pacejka` to 1e-12
  (`test/reference/forza_ref.py`).
- **Rolling resistance and drag.** `C_roll * m * g + ½ ρ CdA v²` opposes the
  motion. Both are off by default: no identified values for an F1TENTH car
  have been published, so set `C_roll` and `CdA` from a coast-down test of
  your own car.
- **Low speed.** Below 0.5 m/s it is still the kinematic model, as in Python,
  but the speed obeys the same grip limit and resistance.

The tires and car (`lf`, `lr`, `h`, `m`, `I`) are ForzaETH's on-track
identification of their NUC5 car (`models/NUC5/NUC5_pacejka.txt` in
[On-Track-SysID](https://github.com/ForzaETH/On-Track-SysID); Dikici et al.,
arXiv:2411.17508). Only the tires were fitted there; the body values were
ForzaETH's inputs. `mu` scales every tire force for other floors
(1.0 is the identification floor). ForzaETH's published tire sets (their
NUC2, NUC5, NUC6, NUC7 and sim cars) peak between about 0.3 and 1.2 times
the load across cars and floors; on these tires that is `mu` from 1 up to
about 3, the range to randomize over.

What this car is like: peak grip is about 0.40 (front) and 0.32 (rear)
times the load. It understeers mildly in normal driving, but the rear lets
go first at the limit, so it spins if pushed. ForzaETH fitted the rear tire
only up to 0.2 rad of slip; past that the curve is an extrapolation.

**Actuators.** Python's `pid` (a P speed loop with gains of 10 and 2, a
bang-bang steering loop, a two-step steering delay) and its limits (`a_max`,
`v_switch`, which is CommonRoad's BMW 320i) are gone. Each step,
`F1Tenth.actuate!` runs the chain a real car runs, the way ForzaETH's
[f1tenth_system](https://github.com/ForzaETH/f1tenth_system) and race stack
run NUC5:

1. **Command limiter**, ForzaETH's `actuation_manager` at 75 Hz: clips the
   planner's steering to -0.327…+0.292 rad and its speed to ±14.375 m/s, then
   slews them at 2.46 rad/s and 2.5 m/s² (up and down).
2. **Steering servo**: follows the limited command as a first-order lag with
   ForzaETH's identified time constant, 0.158 s, within the mechanical limits
   (±0.4189 rad, ±3.2 rad/s).
3. **VESC speed loop**, as its firmware runs it at 1 kHz: the speed command
   (4800 ERPM per m/s) becomes a setpoint ramped at 25,000 ERPM/s; a PID on
   the motor speed sets the motor current, up to ±60 A; below 900 ERPM the
   motor is released. The ramp, gains and limits are the firmware's defaults
   ([vedderb/bldc](https://github.com/vedderb/bldc)), since ForzaETH don't
   publish their VESC settings. The loop is reimplemented from the firmware's
   behavior; no firmware code (GPL-3.0) is copied.
4. **Motor**: drive force = `1.5 * flux_linkage * erpm_gain * 2π/60` per amp
   (the standard torque of a permanent-magnet motor under field-oriented
   control, carried through the gearing), 0.585 N/A, so 35 N (10 m/s²) at
   60 A. The flux linkage, 0.000776 Wb, is the stock F1TENTH VESC
   configuration's (`f1tenth_original` in
   [TeoIlie/F1TENTH_System](https://github.com/TeoIlie/F1TENTH_System)).

Assumptions where nothing is published: no drivetrain losses, no motor or
wheel inertia, no battery sag, and the VESC sees the car's forward speed as
the motor speed (there is no wheel-slip state). The example's pure pursuit
needed retuning for the servo lag: see `WORK` in
[examples/waypoint_follow.jl](examples/waypoint_follow.jl).

## Differences from Python

- The tire model, the actuator chain and the car parameters (see
  [Vehicle model changes](#vehicle-model-changes)). `VehicleParams` has
  different keys; `pid`, `accl_constraints` and `vehicle_dynamics_ks` are
  gone, and `vehicle_dynamics_st` takes the motor's drive acceleration and no
  longer takes Python's positional parameter list.
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
models. The tire model and its parameters come from ForzaETH's
[On-Track-SysID](https://github.com/ForzaETH/On-Track-SysID) and their
[f1tenth_gym fork](https://github.com/ForzaETH/f1tenth_gym) (MIT license).
The bundled maps come from f1tenth_gym; see [maps/LICENSE](maps/LICENSE).
The racetrack maps are downloaded from
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) (GPL-3.0).

## Tests

`julia --project=. -e 'import Pkg; Pkg.test()'` checks every unchanged layer
against outputs frozen from the Python code (`test/reference/`, regenerated by
`test/reference/generate.py`), and runs the Python unit tests' own checks.
Whole-race runs no longer match Python, so they're checked by behavior
instead: closed-loop laps finish, crashes are detected, parameter updates
apply.
