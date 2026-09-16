struct VehicleParams
    μ::Float64         # tire-road friction coefficient
    Cᵦᶠ::Float64         # cornering stiffness, front
    Cᵦʳ::Float64         # cornering stiffness, rear
    lᶠ::Float64         # distance from center of gravity to front axle
    lʳ::Float64         # distance from center of gravity to rear axle
    h::Float64           # center of gravity height
    m::Float64           # vehicle mass
    Iᶻ::Float64         # yaw moment of inertia
    δ⁻::Float64         # minimum steering angle
    δ⁺::Float64         # maximum steering angle
    δ̇⁻::Float64         # minimum steering velocity (rate of change of steering angle)
    δ̇⁺::Float64         # maximum steering velocity
    vₛ::Float64   # velocity above which peak acceleration starts scaling down
    a⁺::Float64         # maximum longitudinal acceleration
    v⁻::Float64         # minimum longitudinal velocity
    v⁺::Float64         # maximum longitudinal velocity
end

function VehicleParams(;
    μ::Float64=1.0489,
    Cᵦᶠ::Float64=4.718,
    Cᵦʳ::Float64=5.4562,
    lᶠ::Float64=0.15875,
    lʳ::Float64=0.17145,
    h::Float64=0.074,
    m::Float64=3.74,
    Iᶻ::Float64=0.04712,
    δ⁻::Float64=-0.4189,
    δ⁺::Float64=0.4189,
    δ̇⁻::Float64=-3.2,
    δ̇⁺::Float64=3.2,
    vₛ::Float64=7.319,
    a⁺::Float64=9.51,
    v⁻::Float64=-5.0,
    v⁺::Float64=20.0,
)
    return VehicleParams(μ, Cᵦᶠ, Cᵦʳ, lᶠ, lʳ, h, m, Iᶻ, δ⁻, δ⁺,
                          δ̇⁻, δ̇⁺, vₛ, a⁺, v⁻, v⁺)
end

function steering_constraint(steering_angle::Float64, steering_velocity::Float64,
                              params::VehicleParams)
    if (steering_angle <= params.δ⁻ && steering_velocity <= 0) ||
       (steering_angle >= params.δ⁺ && steering_velocity >= 0)
        return 0.0
    elseif steering_velocity <= params.δ̇⁻
        return params.δ̇⁻
    elseif steering_velocity >= params.δ̇⁺
        return params.δ̇⁺
    else
        return steering_velocity
    end
end

function accl_constraints(vel::Float64, accl::Float64, params::VehicleParams)
    pos_limit = vel > params.vₛ ? params.a⁺ * params.vₛ / vel : params.a⁺
    if (vel <= params.v⁻ && accl <= 0) || (vel >= params.v⁺ && accl >= 0)
        return 0.0
    elseif accl <= -params.a⁺
        return -params.a⁺
    elseif accl >= pos_limit
        return pos_limit
    else
        return accl
    end
end
