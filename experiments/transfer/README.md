# Cross-simulator transfer

This experiment tunes (or trains) controllers on one simulator. Then it
races them, with no changes, on the other simulators. The results show if
the realistic car changes which controllers look best.

```sh
julia --project=experiments/transfer -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments/transfer -t auto experiments/transfer/transfer.jl --quick          # approximately 3 minutes
julia --project=experiments/transfer -t auto experiments/transfer/transfer.jl --maps example,Spa,Monza --mu 0.6,0.8
```

The script loads the strict Python port from the `archive/python-compat`
branch. It extracts the `src/` folder of that branch into `.compat/` (with
`git archive`), and loads it as a second module next to the module of this
branch.

## Conditions

| name | simulator | car |
|---|---|---|
| `strict` | strict port | the default car of the Python code (f1tenth_gym) |
| `strict-nuc5` | strict port | the body of NUC5; its tires made linear at zero slip (`C_S = B*C*D`); the peak acceleration of its motor and the top speed of its command limiter; the linear tires, P speed loop and bang-bang steering of the Python code |
| `realistic` | this branch | NUC5 |
| `realistic-mu<x>` | this branch | NUC5 with the friction scale `x` (`--mu`) |

`strict` against `strict-nuc5` shows the effect of the choice of car.
`strict-nuc5` against `realistic` shows the effect of the model. The strict
port also finds crashes in its own way (the time-to-collision check of the
Python code, on the LiDAR scan). This is a part of the strict port.

## Controllers

| key | controller | tuned values |
|---|---|---|
| `pp` | pure pursuit on the path (the planner of the example) | lookahead, speed gain |
| `ftg` | follow the gap, reactive on the LiDAR | top speed, steering gain |
| `mpc` | MPPI (a sampling MPC) on the path, with a kinematic bicycle model: the steering follows its command at once | speed gain, horizon |
| `mpc-lag` | the same MPPI, but its model also has the published actuators of NUC5 (steering slew limit 2.46 rad/s, servo time constant 0.158 s) | speed gain, horizon |
| `rl` | a linear policy on 27 LiDAR sectors and the speed, trained with ARS V2 (augmented random search with state normalization) | trained: 3 policies, 300 iterations |

**The path.** On the example map, `pp`, `mpc` and `mpc-lag` follow the
raceline, which stays 0.40 m or more from the walls. The racelines of
f1tenth_racetracks come to 0.2 m from the walls over a quarter to a half of
the lap (Spa, Monza). This is less than half the diagonal of a car body
(0.31 m or more). Thus a controller that follows them closely hits the
walls, on all simulators. On these tracks, the path is the centerline (1 m
or more from the walls), at the raceline speed of the nearest raceline
point. The races start at the first point of the path.

All controllers use the wheelbase of the car. The MPC also uses the
steering range of the car and the distance to the walls (from the map).
These are properties of the car and the track that you measure. You do
not tune them.

The `mpc` and `mpc-lag` pair shows a model effect directly. On the strict
port, the two can drive. On the realistic car, the servo lag makes `mpc`
weave until it crashes. In the full run (results/full), no `mpc` setting
finishes on the realistic car on the example map (0 of 78) or on Monza (0
of 78). On Spa, 9 of 84 settings finish, all at low speed (speed gain 0.2
to 0.4). `mpc-lag` finishes with 22 of 78, 19 of 78 and 13 of 84 settings.

To add a controller, write a struct with `init` (optional),
`act(ctrl, state, obs, ctx) -> (steer, speed)` and `stochastic`. For grid
tuning, also write `params`, `param_names`, `grid_axes` and
`lower_bounds`, and add it to `GRID_CONTROLLERS`.

## What the script measures

One race is two laps (the `done` of the env), from the first path
point. A race slower than 1 m/s on average is a timeout.

**Grid tuning.** The script races each setting of the grid on each
condition. Controllers with random parts (the LiDAR controllers and MPPI)
race with 3 seeds. The score of a setting is its mean race time if all
seeds finish. If not, the setting failed. If the best setting of a
condition is on an edge of the grid, the script adds a row or column of
settings past that edge, and races them. It does this again until no best
setting is on an edge (maximum 8 times in each direction, and not below the
lower bounds).

- **Rank agreement**: the Spearman correlation between the scores of two
  conditions over the full grid (failed settings are equal and last), and
  the number of settings that finish on one condition and fail on the
  other.
- **Transfer**: the best setting of each condition, raced on each condition
  with `--seeds` new seeds (101, 102, …).

**Training (`rl`).** Each ARS iteration samples 16 directions and races
each direction both ways for 8 s, from random points on the path. The
reward is the progress along the path (m), less 50 for a crash. Each 10
iterations, the script scores the policy with two 2-lap races (seeds 1 and
2; the transfer races use seeds from 101) and 8 episodes from points at
equal intervals along the path. The score is, in this order: the number of
finished races, the mean race time, and the total episode reward. The
script keeps the policy with the best score. The
transfer table then races each of the 3 policies of each condition on
each condition.

## Output

The results go to `results/<time>/` (or `--out`):

- `grid.csv`: each grid race
- `grid_extensions.csv`: how many times each grid extended, in each
  direction
- `transfer.csv`: each transfer race
- `rank_agreement.csv`: the rank agreement of each pair of conditions
- `rl_curves.csv`: the mean reward of each ARS iteration
- `provenance.txt`: one line for each start of the script: the two
  commits, the SHA-256 of transfer.jl, the Julia version, the number of
  threads and the options

Each map and controller has its own folder with the same files. A folder
with a `done` file is complete. If you start the script again with the same
`--out`, it skips the complete folders. At the end, it joins the folders
into the files above.

## The full run (results/full)

The command was:

```sh
julia --project=experiments/transfer -t auto experiments/transfer/transfer.jl --maps example,Spa,Monza --mu 0.6,0.8 --out experiments/transfer/results/full
```

It ran in two starts (see `provenance.txt`). The second start used the
centerline path on the racetracks (Spa, Monza). It skipped the example map,
which the change does not affect.

- `log-run2.log`: the printed tables of the first start. Its example map
  tables are the results. Its Spa and Monza tables are from the raceline
  path, which the second start replaced.
- `log-run3.log`: the printed tables of the second start (Spa, Monza).

Results (5 seeds for each transfer race; trained policies: 3 policies × 5
seeds):

- The best setting on `strict` finishes 0 of 5 races on `realistic`, for all
  four grid-tuned controllers on all three maps. The trained policies
  finish 0 of 15.
- The best setting on `strict-nuc5` (the body and tires of NUC5, on the
  model of the Python code) also finishes 0 of 5 races on `realistic` in all
  12 cases. Thus the model causes the gap, not only the choice of car.
- The rank agreement between `strict` and `realistic` over the grids is
  -0.31 to -0.01 for `ftg`, -0.08 to +0.38 for `pp`, and +0.39 to +0.51 for
  `mpc-lag`. Of the settings that finish on `strict`, 39 % to 70 % crash on
  `realistic` for `pp`, `ftg` and `mpc-lag`, and 76 % to 100 % for `mpc`.
- The best settings on `realistic` finish on `strict` for `pp` and `ftg` on
  all maps, and for `mpc-lag` on Spa and Monza.
- For `pp`, `ftg` and `mpc-lag`, the best settings at friction 1.0 fail at
  friction 0.6 (0 of 5 races), and the best settings at friction 0.6 finish
  at 1.0 and 0.8 (5 of 5), on all maps.
- `rl` is conclusive only on the example map. On Spa and Monza, the linear
  ARS policies seldom finish two laps on any condition, `strict` included.
  These tracks need more training than 300 iterations.

## The ablation (results/ablation)

```sh
julia --project=experiments/transfer -t auto experiments/transfer/transfer.jl --conditions ablation --controllers pp,ftg,mpc --maps example,Spa,Monza --out experiments/transfer/results/ablation
```

`--conditions ablation` races strict-nuc5, realistic, and the realistic car
with one part of the realism made ideal (see `ablation_conditions` in
transfer.jl):

| condition | the part made ideal |
|---|---|
| `realistic-linear-tires` | Magic Formula tires with the same zero-slip stiffness, C = 1 and D = 10: linear to a few percent, with no grip limit |
| `realistic-ideal-steering` | servo time constant 2 ms, no slew limit, the full mechanical range, commands at 1 kHz (the 3.2 rad/s mechanical rate limit stays) |
| `realistic-ideal-speed` | no rate limit on the speed command, a fast VESC setpoint ramp (the PID, current limit and motor stay) |
| `realistic-ideal-lidar` | a scan in each step (100 Hz), no sweep, 10 mm noise, 30 m range |
| `realistic-all-ideal` | all four |

Results (the best strict-nuc5 setting, 5 seeds on each of 3 maps):

- With one part made ideal, the best strict-nuc5 setting finishes 0 of 15
  races, for all three controllers and for each of the four parts. No
  single part explains the gap.
- With all four parts ideal, it finishes 10 of 15 races for `ftg` and
  `mpc` (Spa and Monza, not the example map) and 0 of 15 for `pp`. The
  all-ideal car still differs from strict-nuc5: it detects crashes from the
  body and the map (not from the LiDAR), and it keeps the VESC speed loop.
- The rank agreement with strict-nuc5 (mean over the maps) increases most
  with ideal steering: from +0.46 to +0.75 for `pp`, and from +0.26 (on one
  map; on the other two no setting finishes) to +0.81 for `mpc`. For `ftg`,
  only all four parts together restore it (-0.23 to +0.63).
- The best realistic setting finishes all races on `realistic-all-ideal`
  and on strict-nuc5, for the three controllers.
- `log-run1.log` has the printed tables.
