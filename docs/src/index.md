# F1Tenth.jl

F1Tenth.jl is an F1TENTH simulator in Julia. It is a port of the Python
[`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym), and it keeps the
API of the Python code. It simulates one real car, ForzaETH's NUC5, from
published values: Magic Formula tires, the actuator chain of the real car
(command limiter, steering servo and VESC speed loop) and its Hokuyo
UST-10LX LiDAR. The `archive/python-compat` branch keeps the strict port.
The outputs of the strict port agree with the outputs of the Python code.

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

## From Python to Julia

| Python | Julia |
|---|---|
| `gym.make('f110_gym:f110-v0', map='berlin', num_agents=2)` | `F110Env(map="berlin", num_agents=2)` |
| `env.reset(poses)` | `reset!(env, poses)` |
| `env.step(action)` | `step!(env, action)` |
| `env.render(mode='human_fast')` | `render!(env; mode=:human_fast)` |
| `env.update_params(params, index=0)` | `update_params!(env, params; index=1)` |
| `obs['poses_x'][0]` | `obs[:poses_x][1]` or `obs.poses_x[1]` |
| `Integrator.RK4` | `Integrator.RK4` |

Indices start at 1. The env uses the same observation for all steps. To
keep one, use `copy(obs)`. You can also use the 23 tracks of
[f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks) by name
(`map="Spa"`; Pkg downloads them when you use one for the first time), and
all ROS maps by path.

```@docs
F1Tenth
```
