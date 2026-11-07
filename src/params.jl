"""
    VehicleParams(; kwargs...)
    VehicleParams(d::AbstractDict)

Parameters of one car: the vehicle model (see [`vehicle_dynamics_st`](@ref))
and its low-level actuator chain (see [`RaceCar`](@ref)). Every default is a
published value for ForzaETH's NUC5 F1TENTH car, or, where ForzaETH publish
none, a published default of the hardware it runs. "Race stack" below is
[ForzaETH/race_stack](https://github.com/ForzaETH/race_stack), branch
`ros2-humble`, where NUC5's configuration lives.

**Tires** (race stack, `stack_master/config/NUC5/pacejka/dubi/default.yaml`:
ForzaETH's tire identification of NUC5 on their "dubi" floor, with
`friction_coeff: 1.0`)

- `mu = 1.0`: friction scale on every tire force; 1 is the identification floor
- `Bf = 4.142`, `Cf = 0.4375`, `Df = 2.6476`, `Ef = 0.2332` and
  `Br = 4.8721`, `Cr = 2.1651`, `Dr = 1.057`, `Er = 1.692`: front and rear
  Magic Formula coefficients. The lateral force on an axle is
  `mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))`, so `D` bounds
  the friction coefficient. The rear force peaks at 1.00 times the load at
  0.25 rad of slip; with `Er > 1` it then falls, and reverses past 0.63 rad,
  an artifact of the fit (the car is spinning by then). The front builds
  slowly: 0.45, 0.77 and 1.11 times the load at 0.1, 0.2 and 0.4 rad.

**Body** (race stack, `stack_master/config/NUC5/car_model.yaml`)

- `lf = 0.130`, `lr = 0.127`: center of gravity to front/rear axle (m)
- `h = 0.074`: height of the center of gravity (m)
- `m = 2.783`: mass (kg)
- `I = 0.0627`: yaw moment of inertia (kg m²)

**Steering hardware** (race stack, `stack_master/config/NUC5/car_model.yaml`)

- `s_min = -0.4189`, `s_max = 0.4189`: mechanical steering angle limits (rad)
- `sv_min = -3.2`, `sv_max = 3.2`: steering velocity limits (rad/s)
- `tau_steer = 0.15779476`: time constant of the servo, which follows its
  command as a first-order lag (s)

**Command limiter** (ForzaETH's `actuation_manager`, in
[ForzaETH/f1tenth_system](https://github.com/ForzaETH/f1tenth_system), with
NUC5's limits from the race stack, `stack_master/config/NUC5/vesc.yaml`)

- `command_rate = 75.0`: how often it passes a command on (Hz)
- `steer_cmd_min = -0.3271441202`, `steer_cmd_max = 0.2917771883`: steering
  command range (rad)
- `steer_rate = 2.4615384615`: steering command slew limit (rad/s)
- `speed_cmd_min = -14.375`, `speed_cmd_max = 14.375`: speed command range (m/s)
- `accel_cmd = 2.5`, `decel_cmd = 2.5`: how fast the speed command may rise
  and fall (m/s²)

**Motor controller** (a VESC running its firmware's speed control, as
[vedderb/bldc](https://github.com/vedderb/bldc) implements it). ForzaETH
don't publish their VESC settings, so all of these but `erpm_gain` come from
one published configuration of an F1TENTH car's VESC: `vesc1_apr9`
(`vesc/config/vesc1_apr9/vesc_mcconf.xml` in
[TeoIlie/F1TENTH_System](https://github.com/TeoIlie/F1TENTH_System)). It is
the only published one for the stock F1TENTH motor (the motor of the stock
`f1tenth_original` configuration, detected again) on firmware that can run
ForzaETH's commands: braking in speed control and ERPM up to 100,000.
Its speed loop settings are the firmware's defaults but the derivative filter.

- `erpm_gain = 4800.0`: electrical RPM of the motor per m/s (race stack,
  NUC5's `vesc.yaml`, `speed_to_erpm_gain`)
- `erpm_ramp = 25000.0`: speed setpoint ramp (ERPM/s)
- `vesc_kp = 0.004`, `vesc_ki = 0.004`, `vesc_kd = 0.0001`,
  `vesc_kd_filter = 0.5`: speed PID gains and derivative filter
- `vesc_min_erpm = 900.0`: below this setpoint the speed loop stops and the
  firmware holds the duty cycle at zero, shorting the windings (ERPM)
- `vesc_rate = 1000.0`: speed loop rate (Hz)
- `current_max = 65.0`: motor current limit (A)
- `flux_linkage = 0.000755263`, `motor_resistance = 0.006`,
  `motor_inductance = 8.6e-7`: the motor's flux linkage (Wb), winding
  resistance (Ω) and inductance (H), as the VESC's motor detection
  measured them

**LiDAR** (the Hokuyo UST-10LX the stock F1TENTH stack drives with
`urg_node`; Hokuyo's specification C-42-04077)

- `scan_rate = 40.0`: scans per second (25 ms per revolution at 2400 rpm);
  the car sees the latest scan until the next one
- `scan_noise = 0.03`: standard deviation of the range noise (m); the
  datasheet's repeatability bound, σ < 30 mm (white target, factory
  conditions), the only figure it publishes

**Resistance** (off: no identified values for an F1TENTH car have been
published; set them from a coast-down test of your own car)

- `C_roll = 0.0`: rolling resistance coefficient (force `C_roll * m * g`)
- `CdA = 0.0`: drag coefficient times frontal area (m², force `½ ρ CdA v²`)

**Body size** for collisions and drawing

- `width = 0.228`: race stack, NUC5's `car_model.yaml` (`width_m`) (m)
- `length = 0.58`: ForzaETH's
  [f1tenth_gym fork](https://github.com/ForzaETH/f1tenth_gym) default; NUC5's
  configuration gives none (m)

A dict (`String` or `Symbol` keys) must hold every key; extra keys are ignored.
"""
Base.@kwdef struct VehicleParams
    mu::Float64 = 1.0
    Bf::Float64 = 4.142
    Cf::Float64 = 0.4375
    Df::Float64 = 2.6476
    Ef::Float64 = 0.2332
    Br::Float64 = 4.8721
    Cr::Float64 = 2.1651
    Dr::Float64 = 1.057
    Er::Float64 = 1.692
    lf::Float64 = 0.130
    lr::Float64 = 0.127
    h::Float64 = 0.074
    m::Float64 = 2.783
    I::Float64 = 0.0627
    s_min::Float64 = -0.4189
    s_max::Float64 = 0.4189
    sv_min::Float64 = -3.2
    sv_max::Float64 = 3.2
    tau_steer::Float64 = 0.15779476
    command_rate::Float64 = 75.0
    steer_cmd_min::Float64 = -0.3271441202
    steer_cmd_max::Float64 = 0.2917771883
    steer_rate::Float64 = 2.4615384615
    speed_cmd_min::Float64 = -14.375
    speed_cmd_max::Float64 = 14.375
    accel_cmd::Float64 = 2.5
    decel_cmd::Float64 = 2.5
    erpm_gain::Float64 = 4800.0
    erpm_ramp::Float64 = 25000.0
    vesc_kp::Float64 = 0.004
    vesc_ki::Float64 = 0.004
    vesc_kd::Float64 = 0.0001
    vesc_kd_filter::Float64 = 0.5
    vesc_min_erpm::Float64 = 900.0
    vesc_rate::Float64 = 1000.0
    current_max::Float64 = 65.0
    flux_linkage::Float64 = 0.000755263
    motor_resistance::Float64 = 0.006
    motor_inductance::Float64 = 8.6e-7
    scan_rate::Float64 = 40.0
    scan_noise::Float64 = 0.03
    C_roll::Float64 = 0.0
    CdA::Float64 = 0.0
    width::Float64 = 0.228
    length::Float64 = 0.58
end

const PARAM_KEYS = fieldnames(VehicleParams)

function VehicleParams(d::AbstractDict)
    get_key(k) = haskey(d, k) ? d[k] : haskey(d, String(k)) ? d[String(k)] : throw(KeyError(String(k)))
    return VehicleParams(map(get_key, PARAM_KEYS)...)
end

Base.convert(::Type{VehicleParams}, d::AbstractDict) = VehicleParams(d)
