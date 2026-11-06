const PARAM_KEYS = (:mu, :Bf, :Cf, :Df, :Ef, :Br, :Cr, :Dr, :Er, :lf, :lr, :h, :m, :I, :s_min, :s_max,
    :sv_min, :sv_max, :v_switch, :a_max, :v_min, :v_max, :C_roll, :CdA, :width, :length)

"""
    VehicleParams(; mu=1.0, Bf=6.4196, Cf=0.9826, Df=0.4044, Ef=0.5073, Br=7.5624, Cr=1.3006, Dr=0.3237,
                    Er=0.6478, lf=0.165, lr=0.155, h=0.02, m=3.5, I=0.0627, s_min=-0.4189, s_max=0.4189,
                    sv_min=-3.2, sv_max=3.2, v_switch=7.319, a_max=9.51, v_min=-5.0, v_max=20.0,
                    C_roll=0.0, CdA=0.0, width=0.31, length=0.58)
    VehicleParams(d::AbstractDict)

Vehicle parameters for the Pacejka single-track model (see [`vehicle_dynamics_st`](@ref)):

- `mu`: friction scale. Multiplies every tire force; `1.0` is the surface the
  tires were identified on
- `Bf`, `Cf`, `Df`, `Ef` / `Br`, `Cr`, `Dr`, `Er`: front / rear Magic Formula
  coefficients (stiffness, shape, peak, curvature). The lateral force on an
  axle is `mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))`, so `D` is
  the peak friction coefficient
- `lf`, `lr`: distance from the center of gravity to the front/rear axle
- `h`: height of the center of gravity
- `m`: total mass of the vehicle
- `I`: moment of inertia of the vehicle about the z axis
- `s_min`, `s_max`: steering angle limits
- `sv_min`, `sv_max`: steering velocity limits
- `v_switch`: velocity above which acceleration can no longer spin the wheels
- `a_max`: maximum longitudinal acceleration the motor can command
- `v_min`, `v_max`: longitudinal velocity limits
- `C_roll`: rolling resistance coefficient (force `C_roll * m * g`)
- `CdA`: drag coefficient times frontal area, in m² (force `½ ρ CdA v²`)
- `width`, `length`: car body size in meters

Where the defaults come from:

- Tires, `lf`, `lr`, `h`, `m`, `I`: ForzaETH's on-track identification of
  their NUC5 car (`models/NUC5/NUC5_pacejka.txt` in
  [ForzaETH/On-Track-SysID](https://github.com/ForzaETH/On-Track-SysID); Dikici
  et al., "Learning-Based On-Track System Identification for Scaled
  Autonomous Racing in Under a Minute", arXiv:2411.17508). Their formula has
  no friction scale, hence `mu = 1.0`.
- Steering, speed and acceleration limits, `v_switch`, body size: f1tenth_gym's
  defaults.
- `C_roll`, `CdA`: zero (off). No identified values for an F1TENTH car have
  been published; set them from a coast-down test of your own car.

A dict (`String` or `Symbol` keys) must hold every key; extra keys are ignored.
"""
Base.@kwdef struct VehicleParams
    mu::Float64 = 1.0
    Bf::Float64 = 6.4196
    Cf::Float64 = 0.9826
    Df::Float64 = 0.4044
    Ef::Float64 = 0.5073
    Br::Float64 = 7.5624
    Cr::Float64 = 1.3006
    Dr::Float64 = 0.3237
    Er::Float64 = 0.6478
    lf::Float64 = 0.165
    lr::Float64 = 0.155
    h::Float64 = 0.02
    m::Float64 = 3.5
    I::Float64 = 0.0627
    s_min::Float64 = -0.4189
    s_max::Float64 = 0.4189
    sv_min::Float64 = -3.2
    sv_max::Float64 = 3.2
    v_switch::Float64 = 7.319
    a_max::Float64 = 9.51
    v_min::Float64 = -5.0
    v_max::Float64 = 20.0
    C_roll::Float64 = 0.0
    CdA::Float64 = 0.0
    width::Float64 = 0.31
    length::Float64 = 0.58
end

function VehicleParams(d::AbstractDict)
    get_key(k) = haskey(d, k) ? d[k] : haskey(d, String(k)) ? d[String(k)] : throw(KeyError(String(k)))
    return VehicleParams(map(get_key, PARAM_KEYS)...)
end

Base.convert(::Type{VehicleParams}, d::AbstractDict) = VehicleParams(d)
