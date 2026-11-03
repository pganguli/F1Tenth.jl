using StaticArrays: SVector

# Python dynamics_test.py parameters (CommonRoad vehicle 1)
const DYN_TEST_PARAMS = VehicleParams(; mu=1.0489, C_Sf=21.92 / 1.0489, C_Sr=21.92 / 1.0489, lf=0.3048 * 3.793293,
    lr=0.3048 * 4.667707, h=0.3048 * 2.01355, m=4.4482216152605 / 0.3048 * 74.91452,
    I=4.4482216152605 * 0.3048 * 1321.416, s_min=-1.066, s_max=1.066, sv_min=-0.4, sv_max=0.4, v_min=-13.6,
    v_max=50.8, v_switch=7.319, a_max=11.5)

# Stand-in for scipy's odeint in the Python tests: fine fixed-step RK4 over
# numpy.arange(0, 1, 1e-4), returning the state at the last sample.
function integrate(f, x0, u, p)
    x = x0
    h = 1e-4
    for _ in 1:9999
        k1 = f(x, u, p)
        k2 = f(x + h / 2 * k1, u, p)
        k3 = f(x + h / 2 * k2, u, p)
        k4 = f(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return x
end

@testset "dynamic_models (Python unit tests)" begin
    p = DYN_TEST_PARAMS
    g = 9.81

    @testset "derivatives" begin
        f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752,
            0.2401426578627629]
        f_st_gt = [15.7213512030862397, 0.0925527979719355, 0.1500000000000000, 5.3536773276413925,
            0.0529001056654038, 0.6435589397748606, 0.0313297971641291]
        x_ks = SVector(3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566,
            0.0294717351052816)
        x_st = SVector(2.0233348142065677, 0.0041907137716636, 0.0197545248559617, 15.7216236334290116,
            0.0025857914776859, 0.0529001056654038, 0.0033012170610298)
        u = SVector(0.15, 0.63 * g)
        @test maximum(abs.(f_ks_gt - vehicle_dynamics_ks(x_ks, u, p))) ≈ 0 atol = 1e-7
        @test maximum(abs.(f_st_gt - vehicle_dynamics_st(x_st, u, p))) ≈ 0 atol = 1e-7
        # Python's positional-argument form gives the same result
        args = (p.mu, p.C_Sf, p.C_Sr, p.lf, p.lr, p.h, p.m, p.I, p.s_min, p.s_max, p.sv_min, p.sv_max, p.v_switch,
            p.a_max, p.v_min, p.v_max)
        @test vehicle_dynamics_st(collect(x_st), collect(u), args...) == vehicle_dynamics_st(x_st, u, p)
        @test vehicle_dynamics_ks(collect(x_ks), collect(u), args...) == vehicle_dynamics_ks(x_ks, u, p)
        # Python asserts > 5000 evaluations/s
        n = 100_000
        t = @elapsed for _ in 1:n
            x_st = vehicle_dynamics_st(x_st, u, p) * 1e-9 + x_st
        end
        @test n / t > 5000
    end

    x0_st = zero(SVector{7,Float64})
    x0_ks = zero(SVector{5,Float64})
    @testset "zero init, rolling" begin
        u = SVector(0.0, 0.0)
        @test integrate(vehicle_dynamics_st, x0_st, u, p) == x0_st
        @test integrate(vehicle_dynamics_ks, x0_ks, u, p) == x0_ks
    end
    @testset "zero init, decelerating" begin
        u = SVector(0.0, -0.7 * g)
        st_gt = [-3.4335000000000013, 0.0, 0.0, -6.8670000000000018, 0.0, 0.0, 0.0]
        ks_gt = [-3.4335000000000013, 0.0, 0.0, -6.8670000000000018, 0.0]
        @test all(abs.(integrate(vehicle_dynamics_st, x0_st, u, p) - st_gt) .< 1e-2)
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
    @testset "zero init, accelerating and steering left" begin
        u = SVector(0.15, 0.63 * g)
        st_gt = [3.0731976046859715, 0.2869835398304389, 0.15, 6.1802999999999999, 0.1097747074946325,
            0.3248268063223301, 0.0697547542798040]
        ks_gt = [3.0845676868494927, 0.1484249221523042, 0.15, 6.1803000000000017, 0.1203664469224163]
        @test all(abs.(integrate(vehicle_dynamics_st, x0_st, u, p) - st_gt) .< 1e-2)
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
    @testset "zero init, rolling and steering left" begin
        u = SVector(0.15, 0.0)
        st_gt = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0]
        ks_gt = [0.0, 0.0, 0.15, 0.0, 0.0]
        @test all(abs.(integrate(vehicle_dynamics_st, x0_st, u, p) - st_gt) .< 1e-2)
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
end

@testset "dynamic_models (match Python outputs)" begin
    p = GYM_PARAMS
    x, u = ref("dyn_x"), ref("dyn_u")
    st, ks = ref("dyn_st"), ref("dyn_ks")
    for i in axes(x, 1)
        xi = SVector{7}(x[i, :])
        ui = SVector{2}(u[i, :])
        @test vehicle_dynamics_st(xi, ui, p) ≈ st[i, :] rtol = 1e-12 atol = 1e-12
        @test vehicle_dynamics_ks(SVector{5}(x[i, 1:5]), ui, p) ≈ ks[i, :] rtol = 1e-12 atol = 1e-12
    end

    pin, pout = ref("pid_in"), ref("pid_out")
    for i in axes(pin, 1)
        accl, sv = pid(pin[i, 1], pin[i, 2], pin[i, 3], pin[i, 4], p.sv_max, p.a_max, p.v_max, p.v_min)
        @test accl ≈ pout[i, 1] rtol = 1e-14 atol = 1e-14
        @test sv == pout[i, 2]
    end

    cin, cout = ref("constraint_in"), ref("constraint_out")
    for i in axes(cin, 1)
        @test steering_constraint(cin[i, 1], cin[i, 2], p.s_min, p.s_max, p.sv_min, p.sv_max) == cout[i, 1]
        @test accl_constraints(cin[i, 3], cin[i, 4], p.v_switch, p.a_max, p.v_min, p.v_max) == cout[i, 2]
    end
end
