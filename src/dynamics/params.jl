struct VehicleParams
    mu::Float64
    C_Sf::Float64
    C_Sr::Float64
    lf::Float64
    lr::Float64
    h::Float64
    m::Float64
    I::Float64
    s_min::Float64
    s_max::Float64
    sv_min::Float64
    sv_max::Float64
    v_switch::Float64
    a_max::Float64
    v_min::Float64
    v_max::Float64
end

function VehicleParams(;
    mu::Float64=1.0489,
    C_Sf::Float64=4.718,
    C_Sr::Float64=5.4562,
    lf::Float64=0.15875,
    lr::Float64=0.17145,
    h::Float64=0.074,
    m::Float64=3.74,
    I::Float64=0.04712,
    s_min::Float64=-0.4189,
    s_max::Float64=0.4189,
    sv_min::Float64=-3.2,
    sv_max::Float64=3.2,
    v_switch::Float64=7.319,
    a_max::Float64=9.51,
    v_min::Float64=-5.0,
    v_max::Float64=20.0,
)
    return VehicleParams(mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max,
                          sv_min, sv_max, v_switch, a_max, v_min, v_max)
end

function steering_constraint(steering_angle::Float64, steering_velocity::Float64,
                              params::VehicleParams)
    if (steering_angle <= params.s_min && steering_velocity <= 0) ||
       (steering_angle >= params.s_max && steering_velocity >= 0)
        return 0.0
    elseif steering_velocity <= params.sv_min
        return params.sv_min
    elseif steering_velocity >= params.sv_max
        return params.sv_max
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
