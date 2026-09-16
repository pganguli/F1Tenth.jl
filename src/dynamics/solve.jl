using DifferentialEquations

function simulate(state0::State, control_fn, params::VehicleParams,
                   tspan::Tuple{Float64,Float64}; solver=Tsit5())
    f = (u, _p, t) -> dynamics_rhs(State(u), control_fn(t), params)
    prob = ODEProblem(f, state0, tspan)
    return solve(prob, solver)
end
