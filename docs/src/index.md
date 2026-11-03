# F1Tenth.jl

A Julia port of the Python [`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym)
simulator, with the same features. For the same inputs it gives the same
outputs (poses, LiDAR scans, collisions, laps) as the Python code, 10–20×
faster.

## Quick start

```julia
using F1Tenth

env = F110Env(map="berlin", num_agents=2)
obs, reward, done, info = reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])   # rows: x, y, θ
while !done
    action = [0.1 2.0; 0.0 1.5]                 # rows: steering angle, speed
    obs, reward, done, info = step!(env, action)
end
```

For a window, load GLMakie and call [`render!`](@ref):

```julia
using GLMakie
render!(env)
```

## Coming from Python

| Python | Julia |
|---|---|
| `gym.make('f110_gym:f110-v0', map='berlin', num_agents=2)` | `F110Env(map="berlin", num_agents=2)` |
| `env.reset(poses)` | `reset!(env, poses)` |
| `env.step(action)` | `step!(env, action)` |
| `env.render(mode='human_fast')` | `render!(env; mode=:human_fast)` |
| `env.update_params(params, index=0)` | `update_params!(env, params; index=1)` |
| `obs['poses_x'][0]` | `obs[:poses_x][1]` or `obs.poses_x[1]` |
| `Integrator.RK4` | `Integrator.RK4` |

Indices are 1-based. Observations are reused between steps; `copy(obs)`
keeps one. Any ROS map (yaml + image) works, including the tracks in
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks).

```@docs
F1Tenth
```
