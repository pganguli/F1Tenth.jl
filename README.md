# F1Tenth.jl

[![CI](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pganguli/F1Tenth.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://pganguli.github.io/F1Tenth.jl/dev/)

F1Tenth.jl is an F1TENTH simulator in Julia. It is a port of the Python
[`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym), and it keeps the
API of the Python code. It simulates one real F1TENTH car, ForzaETH's NUC5,
from published values:

- Magic Formula tires with a friction ellipse.
- The low-level actuator chain of the real car (command limiter, steering
  servo and VESC speed loop), instead of the controller of the Python code.
- The Hokuyo UST-10LX LiDAR, as its specification describes it.

Thus the results are not the same as the results of the Python code. For
the details, see [Vehicle model changes](#vehicle-model-changes).

> The [`archive/python-compat`](https://github.com/pganguli/F1Tenth.jl/tree/archive/python-compat)
> branch keeps the strict port. It has the features of the Python code and
> no more. Over full races, its car poses, LiDAR scans, collisions, lap
> counts and lap times agree with the Python code to approximately 1e-14.
> It is 10 to 20 times faster.

## Install

```julia
import Pkg
Pkg.add("F1Tenth")
Pkg.add("GLMakie")   # only for the render window
```

## Use

The API is the same as the API of the Python env, with two changes. The
names of functions that change the state end in `!`, as usual in Julia.
The car indices start at 1.

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

`F110Env` takes the keyword arguments of the Python env:

- `seed`
- `map`: `"berlin"`, `"skirk"`, `"levine"`, `"vegas"`, a real-world track
  (for example `"Austin"` or `"Spa"`), or the path of a yaml file without
  its extension
- `map_ext`
- `params`: a `VehicleParams`, or a dict with all of its keys
- `num_agents`, `timestep` and `ego_idx`
- `integrator`: `Integrator.RK4` or `Integrator.Euler`
- `lidar_dist`

You can use the 23 real-world tracks of
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) by their
names (`keys(F1Tenth.RACETRACKS)`). Their license is GPL-3.0, thus the
package does not include them. Pkg downloads them (approximately 6 MB) when
you use one for the first time. For all other ROS maps (a yaml file and an
image), use the path.

These functions are also the same as in the Python code:
`update_map!(env, path, ext)`, `update_params!(env, params; index=-1)` and
`add_render_callback!(env, f)`. The lower layers are also available
(`Simulator`, `RaceCar`, `ScanSimulator2D`, `vehicle_dynamics_st`,
`collision`, `get_vertices` and more).

**The env uses the same observation for all steps.** Each `step!` writes
the new values into the arrays of `obs` and `info`. Thus a step does not
allocate memory. To keep an observation, use `copy(obs)`.

### Rendering

```julia
using GLMakie
render!(env)                    # or render!(env; mode=:human_fast)
```

To move the view, drag it. To zoom, scroll. The render callbacks get an
`EnvRenderer`. It has the camera fields of the Python code (`left`, `right`,
`bottom` and `top`, in units of 50 × meters), `cars` and `score_label`. To
draw more items, use GLMakie on its `axis`.

### Example

[examples/waypoint_follow.jl](examples/waypoint_follow.jl) is the
pure-pursuit example of the Python code:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.add("GLMakie"); Pkg.instantiate()'
julia --project=examples examples/waypoint_follow.jl
```

## Performance

This table shows the time of one `step!` on an AMD Ryzen 5 7640U. The test
uses the example map, with LiDAR noise. All cars drive on the raceline
([benchmark/step.jl](benchmark/step.jl)).

| cars | Python (numba) | Julia, 1 thread | Julia, 4 threads |
|---|---|---|---|
| 1 | 166 µs | 12 µs | 12 µs |
| 2 | 426 µs | 25 µs | 19 µs |
| 4 | 1084 µs | 51 µs | 21 µs |

The two simulators do not do the same work. In each step, Julia integrates
the car ten times (1 ms sub-steps), but it scans only at 40 Hz. Python
integrates one time and scans in each step.

With more than one Julia thread (`julia -t auto`), each car steps on its own
thread. If you run many envs in parallel yourself, set
`F1Tenth.THREAD_AGENTS[] = false`.

## Vehicle model changes

**Tires.** The dynamic single-track model of the Python code has linear
tires. Their lateral force increases with the slip angle with no limit,
thus the car always has grip. `vehicle_dynamics_st` keeps the same state
and the same single-track equations. But it gives each axle a Magic Formula
tire (Pacejka, *Tire and Vehicle Dynamics*), in the form that ForzaETH use
to identify and simulate their tires:

```text
Fy = mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))
```

The other parts of the model:

- **Load transfer.** As in CommonRoad, the axle loads `Fz` change with the
  longitudinal acceleration.
- **Drivetrain.** F1TENTH chassis are 4WD with an open center differential.
  The differential sends equal torque to the two axles. Each axle gets half
  of the drive force. The grip of the weaker axle limits this force.
- **Friction ellipse.** The grip that the drive force uses is not available
  for the cornering force: `Fy` is multiplied by
  `sqrt(1 - (Fx / (mu*D*Fz))^2)`. This is the friction ellipse
  approximation of combined slip (Pacejka). There is no longitudinal tire
  data. Thus the model uses the same peak force in all directions.
- **Body accelerations.** The force balance is the single-track model of
  ForzaETH (in their EKF and in their f1tenth_gym fork), with the front
  drive force added. With no drive force, it agrees with their
  `vehicle_dynamics_st_pacejka` to 1e-12 (`test/reference/forza_ref.py`,
  on a car that does not move).
- **Rolling resistance and drag.** The force `C_roll * m * g + ½ ρ CdA v²`
  opposes the motion. The two are off by default, because there are no
  published values for an F1TENTH car. To use them, measure `C_roll` and
  `CdA` with a coast-down test of your car.
- **Low speed.** Below 0.5 m/s, the model is the kinematic model, as in the
  Python code. But the grip limit and the resistance also apply to the
  speed.

The tires and the body (`lf`, `lr`, `h`, `m`, `I` and `width`) are the
values of NUC5 in the ForzaETH race stack. This is the configuration that
ForzaETH race the car with
([race_stack](https://github.com/ForzaETH/race_stack), branch `ros2-humble`:
`stack_master/config/NUC5/car_model.yaml`, and the tires that ForzaETH
identified on their "dubi" floor,
`stack_master/config/NUC5/pacejka/dubi/default.yaml`).

Their On-Track-SysID repository also has a NUC5 model
(`models/NUC5/NUC5_pacejka.txt`; Dikici et al., arXiv:2411.17508). We do
not use it, for two reasons:

- Its body values are not the same as the values of NUC5 in the race stack
  (3.5 kg and a 0.32 m wheelbase, against 2.783 kg and 0.257 m).
- Its tires have their peak force where the cornering of its
  identification drive stopped (approximately 0.3 g).

`mu` multiplies all tire forces, for other floors (1.0 is the
identification floor). The published tire sets of ForzaETH (their NUC2,
NUC5, NUC6, NUC7 and sim cars) have their peak at approximately 0.3 to 1.2
times the load, on different cars and floors. On these tires, that is a
`mu` of approximately 0.3 to 1.2. Use this range for randomization.

How this car drives:

- The rear tires have their peak force at 1.0 times the load, at 0.25 rad
  of slip.
- The front tires get their grip slowly: 0.45, 0.77 and 1.11 times the load
  at 0.1, 0.2 and 0.4 rad. Thus the car understeers. At the limit, the
  front slides first, and the car goes wide. It does not spin.
- The fitted `Er` of the rear tire (1.692) is more than 1. Thus its force
  decreases after 0.25 rad and reverses after 0.63 rad. This is an artifact
  of the fit. It occurs only at slip angles where the car already spins.

**Actuators.** The package does not have the `pid` of the Python code (a P
speed loop with gains of 10 and 2, a bang-bang steering loop and a two-step
steering delay), or its limits (`a_max` and `v_switch`, which are the values
of the CommonRoad BMW 320i). In each step, `F1Tenth.actuate!` runs the chain
of the real car, as ForzaETH's
[f1tenth_system](https://github.com/ForzaETH/f1tenth_system) and race stack
run NUC5:

1. **Command limiter** (ForzaETH's `actuation_manager`, at 75 Hz). It clips
   the steering of the planner to -0.327 to +0.292 rad, and the speed to
   ±14.375 m/s. Then it limits their rates of change to 2.46 rad/s and
   2.5 m/s² (up and down).
2. **Steering servo.** It follows the limited command as a first-order lag,
   with the time constant that ForzaETH identified (0.158 s). The mechanical
   limits apply (±0.4189 rad and ±3.2 rad/s).
3. **VESC speed loop**, as the VESC firmware runs it at 1 kHz:
   - The speed command (4800 ERPM for each m/s) becomes a setpoint. The
     setpoint changes at a maximum of 25,000 ERPM/s.
   - A PID on the motor speed sets the motor current, to a maximum of
     ±65 A.
   - Below a setpoint of 900 ERPM, the PID stops, and the firmware holds the
     duty cycle at zero. This shorts the windings. The back-EMF then causes
     a braking current (`-ω λ R / (R² + (ω L)²)`). Thus a stop command stops
     the car. The car does not coast.

   ForzaETH do not publish their VESC settings. Thus all the values of the
   ramp, the gains, the limits and the motor come from one published F1TENTH
   VESC configuration: `vesc1_apr9` in
   [TeoIlie/F1TENTH_System](https://github.com/TeoIlie/F1TENTH_System). It is
   the only published configuration for the standard F1TENTH motor, on
   firmware that can run the commands of ForzaETH (braking in speed control,
   and up to 100,000 ERPM). Its speed loop uses the default values of the
   firmware, but not for the derivative filter (0.5; the default is 0.2).
   The package makes the loop again from the behavior of the firmware
   ([vedderb/bldc](https://github.com/vedderb/bldc)). It does not copy
   firmware code (GPL-3.0).
4. **Motor.** The drive force is `1.5 * flux_linkage * erpm_gain * 2π/60`
   for each amp. This is the standard torque of a permanent-magnet motor
   with field-oriented control, through the gearing. The value is
   0.569 N/A, thus 37 N at 65 A (13.3 m/s² on the 2.783 kg of NUC5). The
   flux linkage (0.000755 Wb), the winding resistance (6.0 mΩ) and the
   winding inductance (0.86 µH) are the values that the motor detection of
   the VESC measured, in the same configuration.

Where there are no published values, the model makes these assumptions:

- There are no drivetrain losses.
- There is no motor inertia or wheel inertia.
- The battery voltage does not decrease.
- There are no VESC faults. (A hard stop, without the deceleration limit of
  the command limiter, can cause a short-circuit current of more than the
  100 A fault limit of the VESC.)
- The VESC uses the forward speed of the car as the motor speed. (The model
  has no wheel slip.)

The pure pursuit of the example now steers with the wheelbase of this car.
It also needed new tuning for the servo lag: see `WORK` in
[examples/waypoint_follow.jl](examples/waypoint_follow.jl).

**LiDAR.** The car has the standard F1TENTH LiDAR, a Hokuyo UST-10LX, as its
specification (C-42-04077) describes it:

- 1081 beams over 270°, 0.25° apart.
- One counterclockwise sweep each 25 ms (`scan_rate` = 40 Hz).
- Range noise σ = 30 mm (`scan_noise`). This is the limit in the datasheet,
  σ < 30 mm. The model does not include the ±40 mm accuracy offset.

The Python code takes 1080 beams over 4.7 rad, all at the same time, in
each step, with 10 mm noise.

- **Range.** The specification gives 0.06 to 10 m on a white target, and
  0.06 to 4 m at 10 % reflectance (30 m maximum). The simulator traces the
  beams to `scan_range` = 10 m (for dark walls, set 4 m). A beam that
  finds no wall in this range reads 65.533 m. A beam that finds a wall
  nearer than 21 mm reads 0.004 m. These are the error outputs of the sensor
  (65533 and 4, in mm), with no noise. The Python code traces to 30 m. A
  beam that finds no wall reads 30 m, with noise.
- **Not in the model:** the ±40 mm accuracy offset and the millimeter
  rounding (the noise is larger than the two), missing returns on dark or
  shiny surfaces before `scan_range`, mixed returns at edges, interference
  from other LiDARs, and sunlight. The ranges come from ray marching on the
  grid of the map. Thus they are correct to approximately one cell.
- **Mounting.** NUC5 has the LiDAR 0.27 m in front of the rear axle
  (`lidar_x`, from `static_transforms.yaml` in the race stack). This is a
  short distance in front of the front axle. The default `lidar_dist`
  (measured from the center of gravity) is `lidar_x - lr` = 0.143 m. In the
  Python code, the default `lidar_dist` is 0. Its docstring says that it
  measures `lidar_dist` from the rear axle. But its code measures it from
  the center of gravity, as this package does.
- **Sweep.** The 270° sweep takes 18.75 ms, and the car moves during the
  sweep. Thus the simulator traces each beam from the pose of the car at
  the time of that beam (to 0.5 ms, from the 1 ms sub-steps). When the car
  turns at 1 rad/s, a scan turns by approximately 1°. With
  `lidar_sweep=false`, the simulator takes the full scan from the pose at
  the first beam.
- **Timing.** The simulator publishes a scan when its sweep ends. The car
  sees that scan until the next scan. `obs.scan_times` gives the time of its
  first beam, as ROS gives the time of a `LaserScan`. Thus you can take
  labels (for example the heading) at that time.
- **Beam directions** come from an angle table with 14400 entries (ten for
  each 0.25° step). They are correct to 0.025°. The 2000 entries of the
  Python code move the beams by up to 0.18°.
- The simulator adds other cars to a scan from the middle of its sweep, at
  their poses at the time of publication.

**Wall collisions** are ground truth. They do not come from the LiDAR.
After each 1 ms sub-step, the simulator compares the body of the car
(`length` × `width`, the same rectangle as for collisions between cars)
with the wall cells of the map. The test is exact for the grid
(`F1Tenth.body_hits_wall`). Away from the walls, one lookup in the distance
map is sufficient. Near the walls, the test examines only the few dozen
cells under the body. After a hit, the car stops where it is, and it keeps
its heading.

The Python code does a time-to-collision test on the noisy LiDAR scan,
one time in each step, with a box around the center of the car that is as
long as the wheelbase. This test has these problems:

- It cannot see the 90° behind the car.
- It uses only the forward speed.
- A wall can go 13 cm into the nose of the car before it finds a crash.
- After a crash, it sets the heading to zero.

## Differences from Python

- The tire model, the actuator chain and the car parameters (see
  [Vehicle model changes](#vehicle-model-changes)). `VehicleParams` has
  different keys. There is no `pid`, `accl_constraints` or
  `vehicle_dynamics_ks`. `vehicle_dynamics_st` takes the drive acceleration
  of the motor. It does not take the list of positional parameters of the
  Python code.
- The LiDAR (1081 beams, 270°, 40 Hz sweep, 30 mm noise, 10 m detection
  range with the error outputs of the sensor, mounted where NUC5 has it,
  and `obs.scan_times`).
- The wall collisions (the body against the map in each sub-step; the
  heading stays the same after a crash; see
  [Vehicle model changes](#vehicle-model-changes)). There is no
  `check_ttc!`. `check_ttc` is available.
- The collisions with walls and other cars use the body sizes that
  `update_params!` sets. The Python code keeps the size from the
  construction of the env.
- `render!(env)` (`:human`) shows the race in real time. The Python code
  waits 5 ms for each frame, which is only approximately real time.
  `:human` draws a maximum of 60 frames for each second of sim time. Thus
  a small `timestep` does not make the window slower than real time.
- The window draws each car at its body size (`length` × `width`). The
  Python code draws all cars at 0.58 × 0.31 m.
- Indices start at 1 (`ego_idx`, `update_params!(…; index)`).
- LiDAR noise has the same distribution, but it comes from the random
  number generator of Julia, not from numpy.
- The env uses the same observation for all steps (see above).
- Some inputs that cause errors in the Python code work: `map="vegas"` by
  name, `"levine"` without `map_ext=".pgm"`, and `map_ext` without the dot.
- The Python code keeps the map and the renderer in class attributes, which
  all envs in a process share. Here, each env has its own map and renderer.
- The render window uses GLMakie, not pyglet. Thus the callbacks draw on
  `renderer.axis`. They do not add to `batch`.
- The window keeps the pixels square. If the camera area does not have the
  shape of the window, the window makes the area wider. It does not stretch
  it.
- On HiDPI monitors, the window becomes larger. To set a different scale,
  use `GLMakie.activate!(scalefactor=...)`. The window draws a maximum of 2
  pixels for each unit (3 is too slow for 60 frames each second). To set a
  different value, use `GLMakie.activate!(px_per_unit=...)`.
- You can use the f1tenth_racetracks maps by name. (The Python code cannot
  do this.)
- When you select `Integrator.RK4`, the package does not show the "different
  from previous versions of the gym" warning of the Python code.

One special behavior of the Python code stays: `obs.ego_idx` is always 1.

## Credits

This package is a port of [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)
by Hongrui Zheng, Joseph Auckley, Matthew O'Kelly and Aman Sinha (MIT
license). Its vehicle models follow the single-track models of
[CommonRoad](https://commonroad.in.tum.de/).

- The tire model comes from ForzaETH's
  [On-Track-SysID](https://github.com/ForzaETH/On-Track-SysID) and their
  [f1tenth_gym fork](https://github.com/ForzaETH/f1tenth_gym) (MIT license).
- The parameters of NUC5 come from the
  [race_stack](https://github.com/ForzaETH/race_stack) configuration of
  ForzaETH. The package uses only the values. It does not copy code.
- The maps in the package come from f1tenth_gym. See
  [maps/LICENSE](maps/LICENSE).
- Pkg downloads the racetrack maps from
  [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)
  (GPL-3.0).

To cite this package, use [CITATION.cff](CITATION.cff).

## Tests

To run the tests, use `julia --project=. -e 'import Pkg; Pkg.test()'`.

- The tests compare each layer that did not change with outputs of the
  Python code (`test/reference/`; `test/reference/generate.py` makes them
  again).
- The tests also do the checks of the unit tests of the Python code.
- Full races no longer agree with the Python code. Thus the tests examine
  the behavior: closed-loop laps finish, the simulator finds crashes, and
  parameter updates apply.
