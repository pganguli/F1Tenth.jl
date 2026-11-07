# F1Tenth.jl

[![CI](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://pganguli.github.io/F1Tenth.jl/dev/)

An F1TENTH simulator in Julia, ported from the Python
[`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym) and keeping its API,
that simulates one real F1TENTH car, ForzaETH's NUC5, from published values:
Magic Formula tires with a friction ellipse, the car's actual low-level
actuator chain (command limiter, steering servo, VESC speed loop) in place of
Python's controller, and its Hokuyo UST-10LX LiDAR as its specification
describes it (see [Vehicle model changes](#vehicle-model-changes)). Results
therefore don't match Python step for step.

> The strict port, with Python's features and nothing more, whose car poses,
> LiDAR scans, collisions, lap counts and lap times agree with the Python code
> to about 1e-14 over whole races (10–20× faster), is kept on the
> [`archive/python-compat`](https://github.com/pganguli/F1Tenth.jl/tree/archive/python-compat)
> branch.

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

`step!` time on an AMD Ryzen 5 7640U (example map, LiDAR noise on, every
car driving the raceline; [benchmark/step.jl](benchmark/step.jl)):

| cars | Python (numba) | Julia, 1 thread | Julia, 4 threads |
|---|---|---|---|
| 1 | 166 µs | 12 µs | 12 µs |
| 2 | 426 µs | 25 µs | 19 µs |
| 4 | 1084 µs | 51 µs | 21 µs |

The two don't do the same work: each Julia step integrates
the car ten times (1 ms sub-steps) but scans only at 40 Hz, where Python
integrates once and scans every step.

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
  (`test/reference/forza_ref.py`, on a fixed car).
- **Rolling resistance and drag.** `C_roll * m * g + ½ ρ CdA v²` opposes the
  motion. Both are off by default: no identified values for an F1TENTH car
  have been published, so set `C_roll` and `CdA` from a coast-down test of
  your own car.
- **Low speed.** Below 0.5 m/s it is still the kinematic model, as in Python,
  but the speed obeys the same grip limit and resistance.

The tires and body (`lf`, `lr`, `h`, `m`, `I`, `width`) are NUC5's, from
ForzaETH's race stack, the configuration they race the car with
([race_stack](https://github.com/ForzaETH/race_stack), branch `ros2-humble`:
`stack_master/config/NUC5/car_model.yaml`, and the tires ForzaETH identified
on their "dubi" floor, `stack_master/config/NUC5/pacejka/dubi/default.yaml`).
Their On-Track-SysID repository also has a NUC5 model
(`models/NUC5/NUC5_pacejka.txt`; Dikici et al., arXiv:2411.17508), but its
body values don't match the race stack's NUC5 (3.5 kg and a 0.32 m wheelbase
against 2.783 kg and 0.257 m), and its tires peak where its identification
drive's cornering stopped (about 0.3 g), so the race stack's are used. `mu`
scales every tire force for other floors (1.0 is the identification floor).
ForzaETH's published tire sets (their NUC2, NUC5, NUC6, NUC7 and sim cars)
peak between about 0.3 and 1.2 times the load across cars and floors; on
these tires that is `mu` from about 0.3 to 1.2, the range to randomize over.

What this car is like: the rear tires peak at 1.0 times the load, at 0.25 rad
of slip. The front tires build grip slowly (0.45, 0.77 and 1.11 times the
load at 0.1, 0.2 and 0.4 rad), so the car understeers: at the limit the
front slides first and the car runs wide rather than spinning. The rear
tire's fitted `Er` (1.692) is above 1, which makes its force fall past
0.25 rad and reverse past 0.63 rad; that is an artifact of the fit, at slip
angles where the car is already spinning.

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
   the motor speed sets the motor current, up to ±65 A. Below a 900 ERPM
   setpoint the PID stops and the firmware holds the duty cycle at zero,
   which shorts the windings: the back-EMF drives a braking current
   (`-ω λ R / (R² + (ω L)²)`), so a stop command brings the car to rest
   instead of letting it coast. ForzaETH don't publish their VESC settings,
   so the ramp, gains, limits and motor values all come from one published
   F1TENTH VESC configuration, `vesc1_apr9` in
   [TeoIlie/F1TENTH_System](https://github.com/TeoIlie/F1TENTH_System): the
   only published one for the stock F1TENTH motor on firmware that can run
   ForzaETH's commands (braking in speed control, up to 100,000 ERPM). Its
   speed loop is the firmware's defaults except the derivative filter (0.5,
   default 0.2). The loop is reimplemented from the behavior of the firmware
   ([vedderb/bldc](https://github.com/vedderb/bldc)); no firmware code
   (GPL-3.0) is copied.
4. **Motor**: drive force = `1.5 * flux_linkage * erpm_gain * 2π/60` per amp
   (the standard torque of a permanent-magnet motor under field-oriented
   control, carried through the gearing), 0.569 N/A, so 37 N (10.6 m/s²) at
   65 A. The flux linkage, 0.000755 Wb, and the winding resistance and
   inductance, 6.0 mΩ and 0.86 µH, are what the VESC's motor detection
   measured, in the same configuration.

Assumptions where nothing is published: no drivetrain losses, no motor or
wheel inertia, no battery sag, no VESC faults (stopping hard with the
limiter's deceleration lifted can short more than the VESC's 100 A fault
current), and the VESC sees the car's forward speed as
the motor speed (there is no wheel-slip state). The example's pure pursuit
now steers with this car's wheelbase and needed retuning for the servo lag:
see `WORK` in [examples/waypoint_follow.jl](examples/waypoint_follow.jl).

**LiDAR.** The car carries the stock F1TENTH LiDAR, a Hokuyo UST-10LX,
as its specification (C-42-04077) describes it: 1081 beams over 270°, 0.25°
apart, swept counterclockwise once every 25 ms (`scan_rate` = 40 Hz), with
range noise σ = 30 mm (`scan_noise`, the datasheet's bound σ < 30 mm; its
±40 mm accuracy offset is not modeled). Python takes 1080 beams over 4.7
rad, all at once, every step, with 10 mm noise.

- **Range.** The specification guarantees 0.06 to 10 m on a white target
  and 0.06 to 4 m at 10 % reflectance (30 m at most). Beams are traced out to
  `scan_range` = 10 m (set 4 m for dark walls); a beam that reaches no wall
  within it reads 65.533 m, and one closer than 21 mm reads 0.004 m: the
  sensor's error outputs (65533 and 4, in mm), without noise. Python traces
  to 30 m and reads 30 m, with noise, for a beam that reaches nothing.
- **Not modeled:** the ±40 mm accuracy offset, millimeter rounding (both
  under the noise), dropouts on dark or shiny surfaces short of
  `scan_range`, mixed returns at edges, interference from other LiDARs, and
  sunlight. Ranges come from ray marching on the map's grid, so they land
  within about a cell of the true distance.
- **Mounting.** NUC5 carries it 0.27 m ahead of the rear axle (`lidar_x`,
  from the race stack's `static_transforms.yaml`), just ahead of the front
  axle: `lidar_dist`, measured from the center of gravity, defaults to
  `lidar_x - lr` = 0.143 m. Python's `lidar_dist` defaults to 0, and its
  docstring says it is measured from the rear axle while its code measures
  it from the center of gravity, as here.

- **Sweep.** The 270° take 18.75 ms, and the car moves meanwhile: each beam
  is traced from the car's pose at its own time (to within 0.5 ms, from the
  1 ms sub-steps). Turning at 1 rad/s twists a scan by about 1°.
  `lidar_sweep=false` takes the whole scan from the pose at its first beam
  instead.
- **Timing.** A scan is published when its sweep ends and held until the
  next. `obs.scan_times` gives the time of its first beam, as ROS stamps a
  `LaserScan`, so labels like heading can be taken at that time.
- **Beam directions** come from an angle table with 14400 entries (ten per
  0.25° step), within 0.025° of exact; Python's 2000 entries snap beams by
  up to 0.18°.
- Other cars are added to a scan from the middle of its sweep, at their
  poses when it is published.

**Wall collisions** are ground truth, not read off the LiDAR: after every
1 ms sub-step the car's body (`length` × `width`, the same rectangle as for
car-to-car collisions) is tested against the map's wall cells, exactly for
the grid (`F1Tenth.body_hits_wall`). Away from walls one distance-map lookup
rules a hit out; near them only the few dozen cells under the body are
tested. On a hit the car stops where it is, keeping its heading. Python
instead runs a time-to-collision test on the noisy LiDAR scan once per
step, against a wheelbase-long box around the car's center: it can't see
the 90° behind the car, uses only the forward speed, lets a wall reach
13 cm into the nose, and zeroes the heading on a crash.

## Differences from Python

- The tire model, the actuator chain and the car parameters (see
  [Vehicle model changes](#vehicle-model-changes)). `VehicleParams` has
  different keys; `pid`, `accl_constraints` and `vehicle_dynamics_ks` are
  gone, and `vehicle_dynamics_st` takes the motor's drive acceleration and no
  longer takes Python's positional parameter list.
- The LiDAR (1081 beams, 270°, 40 Hz sweep, 30 mm noise, 10 m detection
  range with the sensor's error outputs, mounted where NUC5 carries it,
  `obs.scan_times`) and wall collisions (the body against the map every sub-step, heading
  kept on a crash; see [Vehicle model changes](#vehicle-model-changes)).
  `check_ttc!` is gone; `check_ttc` remains.
- Collisions with walls and other cars follow body sizes set by
  `update_params!`; Python keeps the size given at construction.
- `render!(env)` (`:human`) runs the window in real time; Python waits 5 ms
  per frame, which only approximates it. Each car is drawn at its body size
  (`length` × `width`); Python draws a fixed 0.58 × 0.31 m.
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

One Python quirk is kept: `obs.ego_idx` is always 1.

## Credits

A port of [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) by Hongrui
Zheng, Joseph Auckley, Matthew O'Kelly and Aman Sinha (MIT license), whose
vehicle models follow the [CommonRoad](https://commonroad.in.tum.de/) single-track
models. The tire model comes from ForzaETH's
[On-Track-SysID](https://github.com/ForzaETH/On-Track-SysID) and their
[f1tenth_gym fork](https://github.com/ForzaETH/f1tenth_gym) (MIT license);
NUC5's parameters come from their
[race_stack](https://github.com/ForzaETH/race_stack) configuration (values
only; no code is copied).
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
