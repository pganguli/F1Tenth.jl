# The low-level actuator chain of a real F1TENTH car, as ForzaETH operate
# their cars:
#
#   planner command (steer, speed)
#     -> command limiter (the actuation_manager of ForzaETH, 75 Hz): clips
#        the steering and speed commands, then limits their rates of change
#     -> steering servo: a first-order lag to the steering command
#     -> VESC motor controller (1 kHz): ramps the speed setpoint, does a PID
#        on the motor speed and sets the motor current. Below its minimum
#        speed, it shorts the windings, which brakes the car.
#     -> motor: the current multiplied by the torque constant, through the
#        gearing, is the drive force on the car
#
# This file has the parts of the chain. The actuate! function of RaceCar
# runs them in each step. For the source of each value, see VehicleParams.

"""
    ActuatorState()

The values that the actuator chain of one car keeps from one step to the
next: the latest planner command, the outputs and the timing of the command
limiter, and the setpoint, the PID memory and the motor current of the VESC
speed loop.
"""
Base.@kwdef mutable struct ActuatorState
    steer_cmd::Float64 = 0.0    # latest planner command
    speed_cmd::Float64 = 0.0
    steer_out::Float64 = 0.0    # command limiter outputs
    speed_out::Float64 = 0.0
    tick_wait::Float64 = 0.0    # time to the next tick of the limiter (s)
    erpm_set::Float64 = 0.0     # ramped speed setpoint of the VESC (ERPM)
    i_term::Float64 = 0.0       # memory of the VESC speed PID
    prev_error::Float64 = 0.0
    d_filter::Float64 = 0.0
    current::Float64 = 0.0      # motor current (A)
end

function reset!(a::ActuatorState)
    for f in fieldnames(ActuatorState)
        setfield!(a, f, 0.0)
    end
    return a
end

# Set the speed side of the chain to zero, as when a crash stops the car.
function stop_motor!(a::ActuatorState)
    a.speed_out = a.erpm_set = a.i_term = a.prev_error = a.d_filter = a.current = 0.0
    return a
end

@inline _step_towards(x, target, max_step) = x + clamp(target - x, -max_step, max_step)

# One tick of the ActuationLimiter of ForzaETH (f1tenth_stack/actuation_limiter.py,
# MIT). It clips each command to its range. Then it moves the output to the
# command by a maximum of rate * dt. When the speed command is less than the
# current output, the speed uses the deceleration rate.
function _limiter_tick!(a::ActuatorState, p::VehicleParams)
    dt = 1 / p.command_rate
    speed = clamp(a.speed_cmd, p.speed_cmd_min, p.speed_cmd_max)
    rate = speed < a.speed_out ? p.decel_cmd : p.accel_cmd
    a.speed_out = _step_towards(a.speed_out, speed, rate * dt)
    steer = clamp(a.steer_cmd, p.steer_cmd_min, p.steer_cmd_max)
    a.steer_out = _step_towards(a.steer_out, steer, p.steer_rate * dt)
    return a
end

# One pass of the speed loop of the VESC firmware (the behavior of
# foc_run_pid_control_speed in vedderb/bldc, motor/foc_math.c), with the
# measured motor speed `erpm` and the loop period `dt`:
# - The setpoint moves to the command by a maximum of erpm_ramp * dt.
# - Below vesc_min_erpm, the PID stops (the integral becomes zero), and the
#   duty cycle stays at zero. This shorts the windings: see _shorted_current.
# - Else, a PID on the speed error (the gains apply to error / 20). A
#   low-pass filter smooths the D term. The output is clipped to ±1. The
#   integral changes after the output, and it is clipped to ±1.
# - The motor current is output * current_max (braking is permitted).
function _vesc_step!(a::ActuatorState, erpm::Float64, dt::Float64, p::VehicleParams)
    a.erpm_set = _step_towards(a.erpm_set, p.erpm_gain * a.speed_out, p.erpm_ramp * dt)
    error = a.erpm_set - erpm
    if abs(a.erpm_set) < p.vesc_min_erpm
        a.i_term = 0.0
        a.prev_error = error
        a.current = _shorted_current(erpm, p)
        return a
    end
    p_term = error * p.vesc_kp / 20
    d_term = (error - a.prev_error) * p.vesc_kd / dt / 20
    a.d_filter -= p.vesc_kd_filter * (a.d_filter - d_term)
    a.prev_error = error
    output = clamp(p_term + a.i_term + a.d_filter, -1.0, 1.0)
    a.i_term = clamp(a.i_term + error * p.vesc_ki * dt / 20, -1.0, 1.0)
    a.current = output * p.current_max
    return a
end

# The motor current with the windings shorted. The VESC firmware holds the
# duty cycle at zero below its minimum speed setpoint ("brake when set ERPM
# is below min ERPM" in motor/mcpwm_foc.c). The back-EMF then causes a
# braking current. For a surface-magnet motor at electrical speed ω with zero
# applied voltage, the dq equations in steady state (their time constant L/R
# is 0.14 ms) give i_q = -ω λ R / (R² + (ω L)²).
function _shorted_current(erpm::Float64, p::VehicleParams)
    ω = erpm * 2pi / 60
    R, L = p.motor_resistance, p.motor_inductance
    return -ω * p.flux_linkage * R / (R^2 + (ω * L)^2)
end

"""
    drive_force_per_amp(params) -> Float64

The drive force on the car for each amp of motor current (N/A). A
permanent-magnet motor with field-oriented control makes the torque
`1.5 * pole_pairs * flux_linkage * current`. Through the gearing, the force
multiplied by the car speed is equal to the torque multiplied by the motor
speed. The electrical speed of the motor for each m/s is
`erpm_gain * 2π/60`. Thus the pole pairs cancel, and the force for each amp
is `1.5 * flux_linkage * erpm_gain * 2π/60`. The model has no drivetrain
losses.
"""
drive_force_per_amp(p::VehicleParams) = 1.5 * p.flux_linkage * p.erpm_gain * 2pi / 60
