# Single-track vehicle dynamics. The constraints and the kinematic model are
# ported from f110_gym/envs/dynamic_models.py (itself following CommonRoad's
# vehicle models), in the same arithmetic order so results agree to the last
# few bits. The dynamic model replaces Python's linear tires with Magic
# Formula tires and a friction ellipse.

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

const GRAVITY = 9.81      # m/s^2, as in CommonRoad and f1tenth_gym
const AIR_DENSITY = 1.225  # kg/m^3, ISA sea level

# Magic Formula pure-slip force per unit load and friction scale (Pacejka,
# "Tire and Vehicle Dynamics"), in the form ForzaETH identify their tires with.
@inline _magic_formula(α, B, C, D, E) = D * sin(C * atan(B * α - E * (B * α - atan(B * α))))

# Vertical axle loads with longitudinal load transfer, as in CommonRoad's
# single-track model.
@inline function _axle_loads(a, p::VehicleParams)
    l = p.lf + p.lr
    return max(p.m * (-a * p.h + GRAVITY * p.lr) / l, 0.0), max(p.m * (a * p.h + GRAVITY * p.lf) / l, 0.0)
end

# Longitudinal force on each axle for acceleration `a`. F1TENTH chassis are
# 4WD with an open center differential, which sends equal torque (so equal
# force) to both axles, and can't pass more than the axle with less grip
# holds. Peak grip is `mu * D * Fz` (D is the Magic Formula peak factor);
# with no longitudinal tire data, the peak is assumed the same in every
# direction (a friction circle).
@inline function _axle_drive(a, Fzf, Fzr, p::VehicleParams)
    Ff, Fr = p.mu * p.Df * Fzf, p.mu * p.Dr * Fzr
    cap = min(Ff, Fr)
    return clamp(p.m * a / 2, -cap, cap), Ff, Fr
end

# Lateral grip left once `Fx` of the peak `Fmax` is used (friction ellipse).
@inline _derate(Fx, Fmax) = Fmax > 0 ? sqrt(max(1 - (Fx / Fmax)^2, 0.0)) : 0.0

# Rolling resistance and aerodynamic drag, opposing the velocity `v`.
@inline _resistance(v, p::VehicleParams) =
    sign(v) * (p.C_roll * p.m * GRAVITY + 0.5 * AIR_DENSITY * p.CdA * v^2)

"""
    vehicle_dynamics_st(x, u_init, params::VehicleParams) -> SVector{7}

Dynamic single-track model with Magic Formula tires, a friction ellipse and
an open-differential 4WD drivetrain. State `x = (x, y, steering angle,
velocity, yaw, yaw rate, slip angle)`, input `u_init = (steering velocity,
acceleration)`; inputs are clamped first. Below 0.5 m/s it switches to the
kinematic model.

The lateral force on each axle is
`mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))`, scaled by
`sqrt(1 - (Fx/Fmax)^2)` for the grip the drive force `Fx` uses. The body
accelerations are ForzaETH's single-track model (their EKF and their
f1tenth_gym fork), with the front drive force added:

    v̇x = (Fxr + Fxf cos δ - Fyf sin δ - Fres) / m + vy ψ̇
    v̇y = (Fyr + Fxf sin δ + Fyf cos δ) / m - vx ψ̇
    ψ̈  = (lf (Fyf cos δ + Fxf sin δ) - lr Fyr) / I

where `Fres` is rolling resistance plus aerodynamic drag. See
[`VehicleParams`](@ref) for where the parameters come from.
"""
@inline function vehicle_dynamics_st(x::SVector{7,Float64}, u_init::SVector{2,Float64}, p::VehicleParams)
    u1 = _steer_c(x[3], u_init[1], p)
    u2 = _accl_c(x[4], u_init[2], p)
    lf, lr, m, I = p.lf, p.lr, p.m, p.I
    δ, v, r, β = x[3], x[4], x[6], x[7]

    Fzf, Fzr = _axle_loads(u2, p)
    Fx, Fmax_f, Fmax_r = _axle_drive(u2, Fzf, Fzr, p)

    # switch to kinematic model for small velocities; the speed still feels
    # the grip limit and resistance
    if abs(v) < 0.5
        lwb = lf + lr
        f_ks = vehicle_dynamics_ks(SVector(x[1], x[2], x[3], x[4], x[5]), SVector(u1, u2), p)
        v_dot = (2Fx - _resistance(v, p)) / m
        return SVector(f_ks[1], f_ks[2], f_ks[3], v_dot, f_ks[5],
            v_dot / lwb * tan(δ) + v / (lwb * cos(δ)^2) * u1,
            0.0)
    end

    # body frame velocities and slip angles
    vx, vy = v * cos(β), v * sin(β)
    α_f = atan(-vy - lf * r, vx) + δ
    α_r = atan(-vy + lr * r, vx)

    # lateral tire forces, derated for the grip the drive force uses
    Fyf = _derate(Fx, Fmax_f) * p.mu * Fzf * _magic_formula(α_f, p.Bf, p.Cf, p.Df, p.Ef)
    Fyr = _derate(Fx, Fmax_r) * p.mu * Fzr * _magic_formula(α_r, p.Br, p.Cr, p.Dr, p.Er)

    # body frame accelerations
    sδ, cδ = sincos(δ)
    vx_dot = (Fx + Fx * cδ - Fyf * sδ - _resistance(vx, p)) / m + vy * r
    vy_dot = (Fyr + Fx * sδ + Fyf * cδ) / m - vx * r
    r_dot = (lf * (Fyf * cδ + Fx * sδ) - lr * Fyr) / I

    return SVector(v * cos(β + x[5]),
        v * sin(β + x[5]),
        u1,
        (vx * vx_dot + vy * vy_dot) / v,
        r,
        r_dot,
        (vx * vy_dot - vy * vx_dot) / v^2)
end

# Python's positional signature for the kinematic model: (x, u, mu, C_Sf, C_Sr,
# lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min,
# v_max). The kinematic model has no tires, so mu, C_Sf and C_Sr are unused.
function _params_from_args(args)
    _, _, _, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max = map(Float64, args)
    return VehicleParams(; lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
end

vehicle_dynamics_ks(x::AbstractVector, u::AbstractVector, args::Vararg{Real,16}) =
    vehicle_dynamics_ks(SVector{5,Float64}(x), SVector{2,Float64}(u), _params_from_args(args))
vehicle_dynamics_ks(x::AbstractVector, u::AbstractVector, p::VehicleParams) =
    vehicle_dynamics_ks(SVector{5,Float64}(x), SVector{2,Float64}(u), p)
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
