const PARAM_KEYS = (:mu, :C_Sf, :C_Sr, :lf, :lr, :h, :m, :I, :s_min, :s_max, :sv_min, :sv_max,
    :v_switch, :a_max, :v_min, :v_max, :width, :length)

"""
    VehicleParams(; mu=1.0489, C_Sf=4.718, C_Sr=5.4562, lf=0.15875, lr=0.17145, h=0.074,
                    m=3.74, I=0.04712, s_min=-0.4189, s_max=0.4189, sv_min=-3.2, sv_max=3.2,
                    v_switch=7.319, a_max=9.51, v_min=-5.0, v_max=20.0, width=0.31, length=0.58)
    VehicleParams(d::AbstractDict)

Vehicle parameters. Same keys and defaults as the Python env's `params` dict:

- `mu`: surface friction coefficient
- `C_Sf`, `C_Sr`: front/rear cornering stiffness coefficients
- `lf`, `lr`: distance from the center of gravity to the front/rear axle
- `h`: height of the center of gravity
- `m`: total mass of the vehicle
- `I`: moment of inertia of the vehicle about the z axis
- `s_min`, `s_max`: steering angle limits
- `sv_min`, `sv_max`: steering velocity limits
- `v_switch`: velocity above which acceleration can no longer spin the wheels
- `a_max`: maximum longitudinal acceleration
- `v_min`, `v_max`: longitudinal velocity limits
- `width`, `length`: car body size in meters

A dict (`String` or `Symbol` keys) must hold every key, as in Python; extra
keys are ignored.
"""
Base.@kwdef struct VehicleParams
    mu::Float64 = 1.0489
    C_Sf::Float64 = 4.718
    C_Sr::Float64 = 5.4562
    lf::Float64 = 0.15875
    lr::Float64 = 0.17145
    h::Float64 = 0.074
    m::Float64 = 3.74
    I::Float64 = 0.04712
    s_min::Float64 = -0.4189
    s_max::Float64 = 0.4189
    sv_min::Float64 = -3.2
    sv_max::Float64 = 3.2
    v_switch::Float64 = 7.319
    a_max::Float64 = 9.51
    v_min::Float64 = -5.0
    v_max::Float64 = 20.0
    width::Float64 = 0.31
    length::Float64 = 0.58
end

function VehicleParams(d::AbstractDict)
    get_key(k) = haskey(d, k) ? d[k] : haskey(d, String(k)) ? d[String(k)] : throw(KeyError(String(k)))
    return VehicleParams(map(get_key, PARAM_KEYS)...)
end

Base.convert(::Type{VehicleParams}, d::AbstractDict) = VehicleParams(d)
