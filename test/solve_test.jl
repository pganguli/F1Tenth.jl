using F1Tenth
using Test

@testset "simulate matches ground truth on a mild scenario" begin
    p = F1Tenth.VehicleParams(;
        μ=1.0489,
        Cᵦᶠ=21.92 / 1.0489,
        Cᵦʳ=21.92 / 1.0489,
        lᶠ=0.3048 * 3.793293,
        lʳ=0.3048 * 4.667707,
        h=0.3048 * 2.01355,
        m=4.4482216152605 / 0.3048 * 74.91452,
        Iᶻ=4.4482216152605 * 0.3048 * 1321.416,
        δ⁻=-1.066,
        δ⁺=1.066,
        δ̇⁻=-0.4,
        δ̇⁺=0.4,
        v⁻=-13.6,
        v⁺=50.8,
        vₛ=7.319,
        a⁺=11.5,
    )
    u = F1Tenth.Action(0.15, 0.63 * 9.81)
    state0 = F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gt = F1Tenth.State(3.0731976046859715, 0.2869835398304389,
                        0.1500000000000000, 6.1802999999999999,
                        0.1097747074946325, 0.3248268063223301,
                        0.0697547542798040)

    sol = F1Tenth.simulate(state0, _ -> u, p, (0.0, 1.0))
    final = F1Tenth.State(sol.u[end])

    @test isapprox(final, gt; atol=1e-2)
end
