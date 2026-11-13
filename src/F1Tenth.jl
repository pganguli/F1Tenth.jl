"""
    F1Tenth

An F1TENTH simulator. It is a port of the Python `f1tenth_gym`, and it
keeps the API of the Python code. It has a realistic vehicle model,
actuator chain and LiDAR for NUC5, the car of ForzaETH. For the details,
see the README. (The `archive/python-compat` branch has the strict Python
port.) To start, see [`F110Env`](@ref).
"""
module F1Tenth

import ColorTypes
import FileIO
import ImageIO   # loaded, thus FileIO can read map images from inside this package
using LazyArtifacts   # the lazy-artifact download needs the module binding, not only the macro
import Logging
import Netpbm    # .pgm maps
import YAML
using PrecompileTools: @setup_workload, @compile_workload
using Random: AbstractRNG, Xoshiro, randn, randn!
using SIMD: Vec, vifelse, vgather
using StaticArrays: SVector, SMatrix, MVector, setindex

# The files are the same as the Python modules in f110_gym/envs, and actuators.jl.
include("params.jl")
include("dynamic_models.jl")
include("actuators.jl")
include("collision_models.jl")
include("maps.jl")
include("laser_models.jl")
include("lidar_tracing.jl")
include("base_classes.jl")
include("rendering.jl")
include("f110_env.jl")
include("precompile.jl")

# The tracer choice is different for each machine: do not keep a choice from the precompilation.
__init__() = (_SIMD_FASTER[] = nothing; nothing)

# f110_env
export F110Env, Observation, step!, reset!, render!, update_map!, update_params!, add_render_callback!
# rendering
export EnvRenderer, ScoreLabel
# base_classes
export Integrator, VehicleParams, Simulator, RaceCar, update_pose!, update_scan!, update_opp_poses!,
    ray_cast_agents!, check_collision!, set_map!
# dynamic_models
export steering_constraint, vehicle_dynamics_st, drive_force_per_amp
# collision_models
export get_vertices, collision, collision_multiple
# laser_models
export ScanSimulator2D, get_scan!, scan, scan!, get_increment, check_ttc, ray_cast!, get_range,
    get_blocked_view_indices

end
