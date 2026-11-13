"""
    VehicleParams(; kwargs...)
    VehicleParams(d::AbstractDict)

The parameters of one car: the vehicle model (see
[`vehicle_dynamics_st`](@ref)) and its low-level actuator chain (see
[`RaceCar`](@ref)). Each default is a published value for NUC5, the F1TENTH
car of ForzaETH. Where ForzaETH publish no value, the default is a published
default of the hardware of the car. Below, "race stack" is
[ForzaETH/race_stack](https://github.com/ForzaETH/race_stack), branch
`ros2-humble`, which has the configuration of NUC5.

**Tires** (race stack, `stack_master/config/NUC5/pacejka/dubi/default.yaml`:
the tire identification of NUC5 that ForzaETH did on their "dubi" floor,
with `friction_coeff: 1.0`)

- `mu = 1.0`: the friction scale on all tire forces. The value 1 is the
  identification floor.
- `Bf = 4.142`, `Cf = 0.4375`, `Df = 2.6476`, `Ef = 0.2332` and
  `Br = 4.8721`, `Cr = 2.1651`, `Dr = 1.057`, `Er = 1.692`: the front and
  rear Magic Formula coefficients. The lateral force on an axle is
  `mu * D * Fz * sin(C * atan(B*α - E*(B*α - atan(B*α))))`. Thus `D` is the
  maximum friction coefficient. The rear force has its peak at 1.00 times
  the load, at 0.25 rad of slip. Because `Er > 1`, the force then decreases,
  and it reverses after 0.63 rad. This is an artifact of the fit (at these
  angles the car already spins). The front force increases slowly: 0.45,
  0.77 and 1.11 times the load at 0.1, 0.2 and 0.4 rad.

**Body** (race stack, `stack_master/config/NUC5/car_model.yaml`)

- `lf = 0.130`, `lr = 0.127`: the distance from the center of gravity to
  the front and rear axles (m)
- `h = 0.074`: the height of the center of gravity (m)
- `m = 2.783`: the mass (kg)
- `I = 0.0627`: the yaw moment of inertia (kg m²)

**Steering hardware** (race stack, `stack_master/config/NUC5/car_model.yaml`)

- `s_min = -0.4189`, `s_max = 0.4189`: the mechanical limits of the steering
  angle (rad)
- `sv_min = -3.2`, `sv_max = 3.2`: the limits of the steering velocity (rad/s)
- `tau_steer = 0.15779476`: the time constant of the servo (s). The servo
  follows its command as a first-order lag.

**Command limiter** (the `actuation_manager` of ForzaETH, in
[ForzaETH/f1tenth_system](https://github.com/ForzaETH/f1tenth_system), with
the limits of NUC5 from the race stack, `stack_master/config/NUC5/vesc.yaml`)

- `command_rate = 75.0`: how frequently the limiter sends a command (Hz)
- `steer_cmd_min = -0.3271441202`, `steer_cmd_max = 0.2917771883`: the range
  of the steering command (rad)
- `steer_rate = 2.4615384615`: the maximum rate of change of the steering
  command (rad/s)
- `speed_cmd_min = -14.375`, `speed_cmd_max = 14.375`: the range of the
  speed command (m/s)
- `accel_cmd = 2.5`, `decel_cmd = 2.5`: the maximum rates at which the speed
  command increases and decreases (m/s²)

**Motor controller** (a VESC with the speed control of its firmware, as
[vedderb/bldc](https://github.com/vedderb/bldc) does it). ForzaETH do not
publish their VESC settings. Thus all these values, but not `erpm_gain`,
come from one published VESC configuration of an F1TENTH car: `vesc1_apr9`
(`vesc/config/vesc1_apr9/vesc_mcconf.xml` in
[TeoIlie/F1TENTH_System](https://github.com/TeoIlie/F1TENTH_System)). It is
the only published configuration for the standard F1TENTH motor (the motor
of the standard `f1tenth_original` configuration, detected again), on
firmware that can run the commands of ForzaETH: braking in speed control,
and up to 100,000 ERPM. Its speed loop uses the default values of the
firmware, but not for the derivative filter.

- `erpm_gain = 4800.0`: the electrical RPM of the motor for each m/s (race
  stack, `vesc.yaml` of NUC5, `speed_to_erpm_gain`)
- `erpm_ramp = 25000.0`: the maximum rate of change of the speed setpoint
  (ERPM/s)
- `vesc_kp = 0.004`, `vesc_ki = 0.004`, `vesc_kd = 0.0001`,
  `vesc_kd_filter = 0.5`: the gains and the derivative filter of the speed PID
- `vesc_min_erpm = 900.0`: below this setpoint (ERPM), the speed loop stops
  and the firmware holds the duty cycle at zero. This shorts the windings.
- `vesc_rate = 1000.0`: the rate of the speed loop (Hz)
- `current_max = 65.0`: the motor current limit (A)
- `flux_linkage = 0.000755263`, `motor_resistance = 0.006`,
  `motor_inductance = 8.6e-7`: the flux linkage (Wb), the winding resistance
  (Ω) and the winding inductance (H) of the motor, as the motor detection of
  the VESC measured them

**LiDAR** (the Hokuyo UST-10LX, which the standard F1TENTH stack operates
with `urg_node`; Hokuyo specification C-42-04077)

- `scan_rate = 40.0`: the scans each second (25 ms for each revolution at
  2400 rpm). The car sees the latest scan until the next scan.
- `scan_noise = 0.03`: the standard deviation of the range noise (m). This
  is the repeatability limit in the datasheet, σ < 30 mm (white target,
  factory conditions). It is the only value that the datasheet gives.
- `scan_range = 10.0`: the detection range (m). The specification gives
  0.06 to 10 m on white Kent sheet (0.06 to 4 m at 10 % diffuse
  reflectance; for dark walls, set 4.0; 30 m maximum). A beam that finds no
  wall in this range reads 65.533 m. A beam that finds a wall nearer than
  21 mm reads 0.004 m. These are the error outputs in the specification
  (65533 and 4, in mm). Only the valid distances get noise.
- `lidar_x = 0.27`: the distance of the LiDAR in front of the rear axle (m)
  (race stack, `static_transforms.yaml` of NUC5, from `base_link` (the
  center of the rear axle) to `laser`). The default `lidar_dist` of the
  environment (measured from the center of gravity) is `lidar_x - lr`.

**Resistance** (off: there are no published identified values for an
F1TENTH car. To use them, measure them with a coast-down test of your car.)

- `C_roll = 0.0`: the rolling resistance coefficient (force `C_roll * m * g`)
- `CdA = 0.0`: the drag coefficient multiplied by the frontal area (m²; force
  `½ ρ CdA v²`)

**Body size**, for collisions and for the window

- `width = 0.228` (m): race stack, `car_model.yaml` of NUC5 (`width_m`)
- `length = 0.58` (m): the default of the
  [f1tenth_gym fork](https://github.com/ForzaETH/f1tenth_gym) of ForzaETH.
  The configuration of NUC5 gives no length.

A dict (with `String` or `Symbol` keys) must have all the keys. The function
ignores other keys.
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
    scan_range::Float64 = 10.0
    lidar_x::Float64 = 0.27
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
