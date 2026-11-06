# The low-level actuator chain of a real F1TENTH car, as ForzaETH run theirs:
#
#   planner command (steer, speed)
#     -> command limiter (ForzaETH's actuation_manager, 75 Hz): clip, then
#        slew-limit the steering and speed commands
#     -> steering servo: first-order lag towards the steering command
#     -> VESC motor controller (1 kHz): ramp the speed setpoint, PID on motor
#        speed, output motor current
#     -> motor: current times the torque constant, through the gearing, is the
#        drive force on the car
#
# This file holds the chain's pieces; RaceCar's actuate! runs them each step.
# See VehicleParams for every value's source.

"""
    ActuatorState()

What one car's actuator chain remembers between steps: the latest planner
command, the command limiter's outputs and timing, and the VESC speed loop's
setpoint, PID memory and motor current.
"""
Base.@kwdef mutable struct ActuatorState
    steer_cmd::Float64 = 0.0    # latest planner command
    speed_cmd::Float64 = 0.0
    steer_out::Float64 = 0.0    # command limiter outputs
    speed_out::Float64 = 0.0
    tick_wait::Float64 = 0.0    # time until the limiter's next tick (s)
    erpm_set::Float64 = 0.0     # VESC's ramped speed setpoint (ERPM)
    i_term::Float64 = 0.0       # VESC speed PID memory
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

# Zero the speed side of the chain, as when the car is stopped by a crash.
function stop_motor!(a::ActuatorState)
    a.speed_out = a.erpm_set = a.i_term = a.prev_error = a.d_filter = a.current = 0.0
    return a
end

@inline _step_towards(x, target, max_step) = x + clamp(target - x, -max_step, max_step)

# One tick of ForzaETH's ActuationLimiter (f1tenth_stack/actuation_limiter.py,
# MIT): clip each command to its range, then move the output towards it by at
# most rate * dt. The speed command uses the deceleration rate whenever it is
# below the current output.
function _limiter_tick!(a::ActuatorState, p::VehicleParams)
    dt = 1 / p.command_rate
    speed = clamp(a.speed_cmd, p.speed_cmd_min, p.speed_cmd_max)
    rate = speed < a.speed_out ? p.decel_cmd : p.accel_cmd
    a.speed_out = _step_towards(a.speed_out, speed, rate * dt)
    steer = clamp(a.steer_cmd, p.steer_cmd_min, p.steer_cmd_max)
    a.steer_out = _step_towards(a.steer_out, steer, p.steer_rate * dt)
    return a
end

# One pass of the VESC firmware's speed loop (behavior of
# foc_run_pid_control_speed in vedderb/bldc, motor/foc_math.c), given the
# measured motor speed `erpm` and the loop period `dt`:
# - the setpoint moves towards the command by at most erpm_ramp * dt
# - below vesc_min_erpm the motor is released: no current, PID memory cleared
# - otherwise PID on the speed error (gains act on error / 20), the D term
#   low-pass filtered, the output clipped to ±1, and the integral (updated
#   after the output) clipped to ±1
# - motor current = output * current_max (braking allowed)
function _vesc_step!(a::ActuatorState, erpm::Float64, dt::Float64, p::VehicleParams)
    a.erpm_set = _step_towards(a.erpm_set, p.erpm_gain * a.speed_out, p.erpm_ramp * dt)
    error = a.erpm_set - erpm
    if abs(a.erpm_set) < p.vesc_min_erpm
        a.i_term = 0.0
        a.prev_error = error
        a.current = 0.0
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

"""
    drive_force_per_amp(params) -> Float64

Drive force on the car per amp of motor current (N/A). A permanent-magnet
motor under field-oriented control makes torque `1.5 * pole_pairs *
flux_linkage * current`; through the gearing, force times car speed equals
torque times motor speed, and the motor's electrical speed per m/s is
`erpm_gain * 2π/60`, so the pole pairs cancel:
`1.5 * flux_linkage * erpm_gain * 2π/60`. Drivetrain losses are not modeled.
"""
drive_force_per_amp(p::VehicleParams) = 1.5 * p.flux_linkage * p.erpm_gain * 2pi / 60
