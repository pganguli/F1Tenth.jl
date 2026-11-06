# Single-track vehicle dynamics, ported from f110_gym/envs/dynamic_models.py
# (itself following CommonRoad's vehicle models). Arithmetic is written in the
# same order as the Python source so results agree to the last few bits.

"""
    accl_constraints(vel, accl, v_switch, a_max, v_min, v_max) -> Float64

Clamp the requested acceleration `accl` to what the car can do at speed `vel`.
"""
@inline function accl_constraints(vel::Float64, accl::Float64, v_switch::Float64, a_max::Float64,
    v_min::Float64, v_max::Float64)
    # positive acceleration limit
    pos_limit = vel > v_switch ? a_max * v_switch / vel : a_max
    if (vel <= v_min && accl <= 0) || (vel >= v_max && accl >= 0)
        return 0.0
    elseif accl <= -a_max
        return -a_max
    elseif accl >= pos_limit
        return pos_limit
    end
    return accl
end
accl_constraints(vel, accl, v_switch, a_max, v_min, v_max) =
    accl_constraints(map(Float64, (vel, accl, v_switch, a_max, v_min, v_max))...)

"""
    steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max) -> Float64

Clamp the requested steering velocity to the steering angle and rate limits.
"""
@inline function steering_constraint(steering_angle::Float64, steering_velocity::Float64, s_min::Float64,
    s_max::Float64, sv_min::Float64, sv_max::Float64)
    if (steering_angle <= s_min && steering_velocity <= 0) || (steering_angle >= s_max && steering_velocity >= 0)
        return 0.0
    elseif steering_velocity <= sv_min
        return sv_min
    elseif steering_velocity >= sv_max
        return sv_max
    end
    return steering_velocity
end
steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max) =
    steering_constraint(map(Float64, (steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max))...)

@inline _steer_c(δ, sv, p::VehicleParams) = steering_constraint(δ, sv, p.s_min, p.s_max, p.sv_min, p.sv_max)
@inline _accl_c(v, a, p::VehicleParams) = accl_constraints(v, a, p.v_switch, p.a_max, p.v_min, p.v_max)

"""
    vehicle_dynamics_ks(x, u_init, params::VehicleParams) -> SVector{5}

Kinematic single-track model. State `x = (x, y, steering angle, velocity, yaw)`,
input `u_init = (steering velocity, acceleration)`; inputs are clamped first.
"""
@inline function vehicle_dynamics_ks(x::SVector{5,Float64}, u_init::SVector{2,Float64}, p::VehicleParams)
    # wheelbase
    lwb = p.lf + p.lr
    u1 = _steer_c(x[3], u_init[1], p)
    u2 = _accl_c(x[4], u_init[2], p)
    return SVector(x[4] * cos(x[5]),
        x[4] * sin(x[5]),
        u1,
        u2,
        x[4] / lwb * tan(x[3]))
end

"""
    vehicle_dynamics_st(x, u_init, params::VehicleParams) -> SVector{7}

Dynamic single-track model. State `x = (x, y, steering angle, velocity, yaw,
yaw rate, slip angle)`, input `u_init = (steering velocity, acceleration)`.
Below 0.5 m/s it switches to the kinematic model, exactly as the Python code.
"""
@inline function vehicle_dynamics_st(x::SVector{7,Float64}, u_init::SVector{2,Float64}, p::VehicleParams)
    # gravity constant m/s^2
    g = 9.81
    u1 = _steer_c(x[3], u_init[1], p)
    u2 = _accl_c(x[4], u_init[2], p)
    mu, C_Sf, C_Sr, lf, lr, h, m, I = p.mu, p.C_Sf, p.C_Sr, p.lf, p.lr, p.h, p.m, p.I

    # switch to kinematic model for small velocities
    if abs(x[4]) < 0.5
        lwb = lf + lr
        f_ks = vehicle_dynamics_ks(SVector(x[1], x[2], x[3], x[4], x[5]), SVector(u1, u2), p)
        return SVector(f_ks[1], f_ks[2], f_ks[3], f_ks[4], f_ks[5],
            u2 / lwb * tan(x[3]) + x[4] / (lwb * cos(x[3])^2) * u1,
            0.0)
    end
    return SVector(x[4] * cos(x[7] + x[5]),
        x[4] * sin(x[7] + x[5]),
        u1,
        u2,
        x[6],
        -mu * m / (x[4] * I * (lr + lf)) * (lf^2 * C_Sf * (g * lr - u2 * h) + lr^2 * C_Sr * (g * lf + u2 * h)) * x[6] +
        mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u2 * h) - lf * C_Sf * (g * lr - u2 * h)) * x[7] +
        mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u2 * h) * x[3],
        (mu / (x[4]^2 * (lr + lf)) * (C_Sr * (g * lf + u2 * h) * lr - C_Sf * (g * lr - u2 * h) * lf) - 1) * x[6] -
        mu / (x[4] * (lr + lf)) * (C_Sr * (g * lf + u2 * h) + C_Sf * (g * lr - u2 * h)) * x[7] +
        mu / (x[4] * (lr + lf)) * (C_Sf * (g * lr - u2 * h)) * x[3])
end

# Python's positional signature: (x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min,
# s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max).
_params_from_args(args) = VehicleParams(map(Float64, args)..., 0.31, 0.58)

vehicle_dynamics_ks(x::AbstractVector, u::AbstractVector, args::Vararg{Real,16}) =
    vehicle_dynamics_ks(SVector{5,Float64}(x), SVector{2,Float64}(u), _params_from_args(args))
vehicle_dynamics_ks(x::AbstractVector, u::AbstractVector, p::VehicleParams) =
    vehicle_dynamics_ks(SVector{5,Float64}(x), SVector{2,Float64}(u), p)
vehicle_dynamics_st(x::AbstractVector, u::AbstractVector, args::Vararg{Real,16}) =
    vehicle_dynamics_st(SVector{7,Float64}(x), SVector{2,Float64}(u), _params_from_args(args))
vehicle_dynamics_st(x::AbstractVector, u::AbstractVector, p::VehicleParams) =
    vehicle_dynamics_st(SVector{7,Float64}(x), SVector{2,Float64}(u), p)

"""
    pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v, dt) -> (accl, sv)

The env's built-in controller: turns a target speed and steering angle into an
acceleration and a steering velocity.

Steering is rate limited: `sv` is the steering velocity that reaches `steer`
in one step of length `dt`, clipped to `±max_sv`. Large changes slew at
`max_sv`; small ones land on the target without overshoot. This differs from
Python, whose bang-bang steering (always `±max_sv`) overshoots and then
chatters around the target by up to `max_sv * dt` every step.
"""
@inline function pid(speed::Float64, steer::Float64, current_speed::Float64, current_steer::Float64,
    max_sv::Float64, max_a::Float64, max_v::Float64, min_v::Float64, dt::Float64)
    # steering
    sv = clamp((steer - current_steer) / dt, -max_sv, max_sv)
    # acceleration
    vel_diff = speed - current_speed
    if current_speed > 0.0
        # currently forward: accelerate or brake
        kp = vel_diff > 0 ? 10.0 * max_a / max_v : 10.0 * max_a / (-min_v)
    else
        # currently backwards: brake or accelerate
        kp = vel_diff > 0 ? 2.0 * max_a / max_v : 2.0 * max_a / (-min_v)
    end
    accl = kp * vel_diff
    return accl, sv
end
pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v, dt) =
    pid(map(Float64, (speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v, dt))...)
