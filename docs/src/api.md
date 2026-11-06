# API

The layers mirror the Python package: the gym environment, the simulator
and its cars, then the dynamics, collision and LiDAR models.

## Environment

```@docs
F110Env
step!
reset!
render!
update_map!
update_params!
add_render_callback!
Observation
```

## Rendering

```@docs
EnvRenderer
ScoreLabel
```

## Simulator and cars

```@docs
Simulator
RaceCar
F1Tenth.check_walls!
Integrator
VehicleParams
update_pose!
update_scan!
update_opp_poses!
check_ttc!
ray_cast_agents!
check_collision!
set_map!
```

## Dynamics

```@docs
vehicle_dynamics_st
steering_constraint
drive_force_per_amp
F1Tenth.actuate!
F1Tenth.ActuatorState
```

## Collisions

```@docs
get_vertices
collision
collision_multiple
```

## LiDAR

```@docs
ScanSimulator2D
get_scan!
scan!
get_increment
check_ttc
ray_cast!
get_range
get_blocked_view_indices
```

## Performance settings

```@docs
F1Tenth.TRACER
F1Tenth.THREAD_AGENTS
```
