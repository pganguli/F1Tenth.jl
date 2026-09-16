using StaticArrays

const State = SVector{7,Float64}
const Action = SVector{2,Float64}

@inline function dynamics_rhs(state::State, action::Action, params::VehicleParams)
    x, y, δ, v, ψ, ψ̇, β = state
    sv = steering_constraint(δ, action[1], params)
    accl = accl_constraints(v, action[2], params)
    p = (params.μ, params.C_Sf, params.C_Sr, params.lᶠ, params.lʳ,
         params.h, params.m, params.Iᶻ, sv, accl)
    if abs(v) < 0.5
        # Python's vehicle_dynamics_st, when it internally switches to the
        # kinematic model for |v| < 0.5, does *not* zero ψ̈ (yaw angular
        # acceleration, d/dt of the yaw-rate state) like the standalone
        # vehicle_dynamics_ks does -- it computes the derivative of the KS
        # yaw-rate expression w.r.t. time (only β_dot is zeroed).
        # KS_RHS/KS_MODE (Task 2) deliberately set both to zero for a
        # cleaner standalone HybridSystem-mode equation set, so that
        # simplification is corrected here, at the single per-stage
        # evaluation point, rather than in KS_RHS itself.
        ks = KS_RHS(state, p)
        lwb = params.lᶠ + params.lʳ
        ψ̈ = accl / lwb * tan(δ) + v / (lwb * cos(δ)^2) * sv
        return State(ks[1], ks[2], ks[3], ks[4], ks[5], ψ̈, 0.0)
    else
        return ST_RHS(state, p)
    end
end

# k₁..k₄: classical RK4 stage derivatives (dynamics_rhs evaluated at the
# start, two midpoint estimates, and the endpoint of the step).
function step_dynamics(state::State, action::Action, params::VehicleParams, dt::Float64)
    k₁ = dynamics_rhs(state, action, params)
    k₂ = dynamics_rhs(state + dt * (k₁ / 2), action, params)
    k₃ = dynamics_rhs(state + dt * (k₂ / 2), action, params)
    k₄ = dynamics_rhs(state + dt * k₃, action, params)
    new_state = state + dt * (1 / 6) * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

    x, y, δ, v, ψ, ψ̇, β = new_state
    ψ = mod(ψ, 2 * pi)

    return State(x, y, δ, v, ψ, ψ̇, β)
end
