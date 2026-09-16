using F1Tenth
using StaticArrays
using Test

@testset "KS_RHS / ST_RHS ground truth derivatives" begin
    # CommonRoad reference vehicle parameters used by the Python test suite
    mu = 1.0489
    C_Sf = 21.92 / 1.0489
    C_Sr = 21.92 / 1.0489
    lf = 0.3048 * 3.793293
    lr = 0.3048 * 4.667707
    h = 0.3048 * 2.01355
    m = 4.4482216152605 / 0.3048 * 74.91452
    I = 4.4482216152605 * 0.3048 * 1321.416

    g = 9.81
    sv = 0.15
    accl_cmd = 0.63 * g

    # f1tenth_gym's vehicle_dynamics_ks/st (the Python reference this test
    # is ported from) internally clamp the commanded acceleration via
    # accl_constraints, using the CommonRoad reference vehicle's own
    # longitudinal bounds (v_switch = 7.319, a_max = 11.5), *before*
    # evaluating the derivative. F1Tenth.jl's KS_RHS/ST_RHS deliberately
    # take the control input already clamped (see mtk_models.jl), so the
    # ground-truth derivatives below are only reproduced when `accl` is
    # pre-clamped the same way -- once per test state, since the clamp
    # depends on the current velocity.
    accl_ref_clamp(v, accl; v_switch=7.319, a_max=11.5) =
        v > v_switch ? min(accl, a_max * v_switch / v) : min(accl, a_max)

    # x_ks from Python test, padded with psi_dot=0, beta=0 to fill the
    # shared 7-element state space
    x_ks = SVector{7,Float64}(3.9579422297936526, 0.0391650102771405,
                               0.0378491427211811, 16.3546957860883566,
                               0.0294717351052816, 0.0, 0.0)
    f_ks_gt = SVector{7,Float64}(16.3475935934250209, 0.4819314886013121,
                                  0.1500000000000000, 5.1464424102339752,
                                  0.2401426578627629, 0.0, 0.0)
    p_ks = (mu, C_Sf, C_Sr, lf, lr, h, m, I, sv, accl_ref_clamp(x_ks[4], accl_cmd))

    x_st = SVector{7,Float64}(2.0233348142065677, 0.0041907137716636,
                               0.0197545248559617, 15.7216236334290116,
                               0.0025857914776859, 0.0529001056654038,
                               0.0033012170610298)
    f_st_gt = SVector{7,Float64}(15.7213512030862397, 0.0925527979719355,
                                  0.1500000000000000, 5.3536773276413925,
                                  0.0529001056654038, 0.6435589397748606,
                                  0.0313297971641291)
    p_st = (mu, C_Sf, C_Sr, lf, lr, h, m, I, sv, accl_ref_clamp(x_st[4], accl_cmd))

    @test isapprox(F1Tenth.KS_RHS(x_ks, p_ks), f_ks_gt; atol=1e-6)
    @test isapprox(F1Tenth.ST_RHS(x_st, p_st), f_st_gt; atol=1e-6)
end

@testset "KS_RHS / ST_RHS are allocation-free" begin
    mu, C_Sf, C_Sr, lf, lr, h, m, I = 1.0489, 4.718, 5.4562, 0.15875, 0.17145, 0.074, 3.74, 0.04712
    p = (mu, C_Sf, C_Sr, lf, lr, h, m, I, 0.1, 1.0)
    state = SVector{7,Float64}(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    F1Tenth.KS_RHS(state, p) # warmup
    F1Tenth.ST_RHS(state, p) # warmup
    @test (@allocated F1Tenth.KS_RHS(state, p)) == 0
    @test (@allocated F1Tenth.ST_RHS(state, p)) == 0
end
