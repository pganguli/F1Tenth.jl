# Single-track vehicle dynamics with Magic Formula tires and a friction
# ellipse. The steering constraint is ported from f110_gym/envs/dynamic_models.py
# (itself following CommonRoad's vehicle models).

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
drive acceleration)`. The steering velocity is clamped to the steering
limits; the drive acceleration is the motor's force over the mass, before
the tires limit it. Below 0.5 m/s it switches to the kinematic model.

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
    sv = _steer_c(x[3], u_init[1], p)   # steering velocity
    a = u_init[2]                       # drive acceleration
    lf, lr, m, I = p.lf, p.lr, p.m, p.I
    δ, v, r, β = x[3], x[4], x[6], x[7]

    Fzf, Fzr = _axle_loads(a, p)
    Fx, Fmax_f, Fmax_r = _axle_drive(a, Fzf, Fzr, p)

    # switch to kinematic model for small velocities; the speed still feels
    # the grip limit and resistance
    if abs(v) < 0.5
        lwb = lf + lr
        v_dot = (2Fx - _resistance(v, p)) / m
        return SVector(v * cos(x[5]),
            v * sin(x[5]),
            sv,
            v_dot,
            v / lwb * tan(δ),
            v_dot / lwb * tan(δ) + v / (lwb * cos(δ)^2) * sv,
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
        sv,
        (vx * vx_dot + vy * vy_dot) / v,
        r,
        r_dot,
        (vx * vy_dot - vy * vx_dot) / v^2)
end

vehicle_dynamics_st(x::AbstractVector, u::AbstractVector, p::VehicleParams) =
    vehicle_dynamics_st(SVector{7,Float64}(x), SVector{2,Float64}(u), p)
