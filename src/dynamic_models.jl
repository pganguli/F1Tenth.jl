# Single-track vehicle dynamics with Magic Formula tires and a friction
# ellipse. The steering constraint is a port of f110_gym/envs/dynamic_models.py
# (which follows the vehicle models of CommonRoad).

"""
    steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max) -> Float64

Clip the requested steering velocity to the limits of the steering angle
and of the steering rate.
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

# The Magic Formula pure-slip force, for each unit of load and of friction
# scale (Pacejka, "Tire and Vehicle Dynamics"), in the form that ForzaETH
# use to identify their tires.
@inline _magic_formula(α, B, C, D, E) = D * sin(C * atan(B * α - E * (B * α - atan(B * α))))

# The vertical axle loads with longitudinal load transfer, as in the
# single-track model of CommonRoad.
@inline function _axle_loads(a, p::VehicleParams)
    l = p.lf + p.lr
    return max(p.m * (-a * p.h + GRAVITY * p.lr) / l, 0.0), max(p.m * (a * p.h + GRAVITY * p.lf) / l, 0.0)
end

# The longitudinal force on each axle for the acceleration `a`. F1TENTH
# chassis are 4WD with an open center differential. The differential sends
# equal torque (thus equal force) to the two axles. It cannot send more than
# the axle with less grip holds. The peak grip is `mu * D * Fz` (D is the
# Magic Formula peak factor). There is no longitudinal tire data, thus the
# peak is the same in all directions (a friction circle).
@inline function _axle_drive(a, Fzf, Fzr, p::VehicleParams)
    Ff, Fr = p.mu * p.Df * Fzf, p.mu * p.Dr * Fzr
    cap = min(Ff, Fr)
    return clamp(p.m * a / 2, -cap, cap), Ff, Fr
end

# The lateral grip that is available when the drive force uses `Fx` of the
# peak `Fmax` (friction ellipse).
@inline _derate(Fx, Fmax) = Fmax > 0 ? sqrt(max(1 - (Fx / Fmax)^2, 0.0)) : 0.0

# The rolling resistance and the aerodynamic drag, which oppose the velocity `v`.
@inline _resistance(v, p::VehicleParams) =
    sign(v) * (p.C_roll * p.m * GRAVITY + 0.5 * AIR_DENSITY * p.CdA * v^2)

"""
    vehicle_dynamics_st(x, u_init, params::VehicleParams) -> SVector{7}

The dynamic single-track model, with Magic Formula tires, a friction
ellipse and a 4WD drivetrain with an open differential.

- The state is `x = (x, y, steering angle, velocity, yaw, yaw rate, slip
  angle)`.
- The input is `u_init = (steering velocity, drive acceleration)`. The
  function clips the steering velocity to the steering limits. The drive
  acceleration is the force of the motor divided by the mass, before the
  tires limit it.
- Below 0.5 m/s, the function uses the kinematic model.

The lateral force on each axle is
`mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))`, multiplied by
`sqrt(1 - (Fx/Fmax)^2)` for the grip that the drive force `Fx` uses. The
body accelerations are the single-track model of ForzaETH (in their EKF and
in their f1tenth_gym fork), with the front drive force added:

    v̇x = (Fxr + Fxf cos δ - Fyf sin δ - Fres) / m + vy ψ̇
    v̇y = (Fyr + Fxf sin δ + Fyf cos δ) / m - vx ψ̇
    ψ̈  = (lf (Fyf cos δ + Fxf sin δ) - lr Fyr) / I

where `Fres` is the rolling resistance plus the aerodynamic drag. For the
sources of the parameters, see [`VehicleParams`](@ref).
"""
@inline function vehicle_dynamics_st(x::SVector{7,Float64}, u_init::SVector{2,Float64}, p::VehicleParams)
    sv = _steer_c(x[3], u_init[1], p)   # steering velocity
    a = u_init[2]                       # drive acceleration
    lf, lr, m, I = p.lf, p.lr, p.m, p.I
    δ, v, r, β = x[3], x[4], x[6], x[7]

    Fzf, Fzr = _axle_loads(a, p)
    Fx, Fmax_f, Fmax_r = _axle_drive(a, Fzf, Fzr, p)

    # at low velocities, use the kinematic model. The grip limit and the
    # resistance also apply to the speed.
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

    # the velocities in the body frame, and the slip angles
    vx, vy = v * cos(β), v * sin(β)
    α_f = atan(-vy - lf * r, vx) + δ
    α_r = atan(-vy + lr * r, vx)

    # the lateral tire forces, less the grip that the drive force uses
    Fyf = _derate(Fx, Fmax_f) * p.mu * Fzf * _magic_formula(α_f, p.Bf, p.Cf, p.Df, p.Ef)
    Fyr = _derate(Fx, Fmax_r) * p.mu * Fzr * _magic_formula(α_r, p.Br, p.Cr, p.Dr, p.Er)

    # the accelerations in the body frame
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
