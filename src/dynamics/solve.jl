using DifferentialEquations

function simulate(𝐱₀::State, control_fn, params::VehicleParams,
                   tspan::Tuple{Float64,Float64}; solver=Tsit5())
    f = (u, _p, t) -> dynamics_rhs(State(u), control_fn(t), params)
    prob = ODEProblem(f, 𝐱₀, tspan)
    return solve(prob, solver)
end
