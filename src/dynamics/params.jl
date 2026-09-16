struct VehicleParams
    μ::Float64
    C_Sf::Float64
    C_Sr::Float64
    l_f::Float64
    l_r::Float64
    h::Float64
    m::Float64
    I_z::Float64
    δ_min::Float64
    δ_max::Float64
    δ̇_min::Float64
    δ̇_max::Float64
    v_switch::Float64
    a_max::Float64
    v_min::Float64
    v_max::Float64
end

function VehicleParams(;
    μ::Float64=1.0489,
    C_Sf::Float64=4.718,
    C_Sr::Float64=5.4562,
    l_f::Float64=0.15875,
    l_r::Float64=0.17145,
    h::Float64=0.074,
    m::Float64=3.74,
    I_z::Float64=0.04712,
    δ_min::Float64=-0.4189,
    δ_max::Float64=0.4189,
    δ̇_min::Float64=-3.2,
    δ̇_max::Float64=3.2,
    v_switch::Float64=7.319,
    a_max::Float64=9.51,
    v_min::Float64=-5.0,
    v_max::Float64=20.0,
)
    return VehicleParams(μ, C_Sf, C_Sr, l_f, l_r, h, m, I_z, δ_min, δ_max,
                          δ̇_min, δ̇_max, v_switch, a_max, v_min, v_max)
end

function steering_constraint(steering_angle::Float64, steering_velocity::Float64,
                              params::VehicleParams)
    if (steering_angle <= params.δ_min && steering_velocity <= 0) ||
       (steering_angle >= params.δ_max && steering_velocity >= 0)
        return 0.0
    elseif steering_velocity <= params.δ̇_min
        return params.δ̇_min
    elseif steering_velocity >= params.δ̇_max
        return params.δ̇_max
    else
        return steering_velocity
    end
end

function accl_constraints(vel::Float64, accl::Float64, params::VehicleParams)
    pos_limit = vel > params.v_switch ? params.a_max * params.v_switch / vel : params.a_max
    if (vel <= params.v_min && accl <= 0) || (vel >= params.v_max && accl >= 0)
        return 0.0
    elseif accl <= -params.a_max
        return -params.a_max
    elseif accl >= pos_limit
        return pos_limit
    else
        return accl
    end
end
