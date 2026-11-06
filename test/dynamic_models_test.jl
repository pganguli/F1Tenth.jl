using StaticArrays: SVector

# Python dynamics_test.py parameters (CommonRoad vehicle 1); the kinematic
# model doesn't use the tire parameters
const DYN_TEST_PARAMS = VehicleParams(; lf=0.3048 * 3.793293,
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

@testset "dynamic_models (Python unit tests, kinematic model)" begin
    p = DYN_TEST_PARAMS
    g = 9.81

    @testset "derivatives" begin
        f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752,
            0.2401426578627629]
        x_ks = SVector(3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566,
            0.0294717351052816)
        u = SVector(0.15, 0.63 * g)
        @test maximum(abs.(f_ks_gt - vehicle_dynamics_ks(x_ks, u, p))) ≈ 0 atol = 1e-7
        # Python's positional-argument form gives the same result (its tire
        # arguments are unused by the kinematic model)
        args = (1.0489, 20.9, 20.9, p.lf, p.lr, p.h, p.m, p.I, p.s_min, p.s_max, p.sv_min, p.sv_max, p.v_switch,
            p.a_max, p.v_min, p.v_max)
        @test vehicle_dynamics_ks(collect(x_ks), collect(u), args...) == vehicle_dynamics_ks(x_ks, u, p)
    end

    x0_ks = zero(SVector{5,Float64})
    @testset "zero init, rolling" begin
        u = SVector(0.0, 0.0)
        @test integrate(vehicle_dynamics_ks, x0_ks, u, p) == x0_ks
    end
    @testset "zero init, decelerating" begin
        u = SVector(0.0, -0.7 * g)
        ks_gt = [-3.4335000000000013, 0.0, 0.0, -6.8670000000000018, 0.0]
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
    @testset "zero init, accelerating and steering left" begin
        u = SVector(0.15, 0.63 * g)
        ks_gt = [3.0845676868494927, 0.1484249221523042, 0.15, 6.1803000000000017, 0.1203664469224163]
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
    @testset "zero init, rolling and steering left" begin
        u = SVector(0.15, 0.0)
        ks_gt = [0.0, 0.0, 0.15, 0.0, 0.0]
        @test all(abs.(integrate(vehicle_dynamics_ks, x0_ks, u, p) - ks_gt) .< 1e-2)
    end
end

@testset "dynamic_models (match Python outputs)" begin
    p = PY_PARAMS
    x, u = ref("dyn_x"), ref("dyn_u")
    ks = ref("dyn_ks")
    for i in axes(x, 1)
        ui = SVector{2}(u[i, :])
        @test vehicle_dynamics_ks(SVector{5}(x[i, 1:5]), ui, p) ≈ ks[i, :] rtol = 1e-12 atol = 1e-12
    end

    # The speed loop still matches Python; the steering loop deliberately doesn't
    # (see "rate-limited steering" below).
    pin, pout = ref("pid_in"), ref("pid_out")
    for i in axes(pin, 1)
        accl, _ = pid(pin[i, 1], pin[i, 2], pin[i, 3], pin[i, 4], p.sv_max, p.a_max, p.v_max, p.v_min, 0.01)
        @test accl ≈ pout[i, 1] rtol = 1e-14 atol = 1e-14
    end

    @testset "rate-limited steering" begin
        dt = 0.01
        sv(target, current) = pid(0.0, target, 0.0, current, p.sv_max, p.a_max, p.v_max, p.v_min, dt)[2]
        # large errors slew at the limit, small ones are closed in one step
        @test sv(0.3, 0.0) == p.sv_max
        @test sv(-0.3, 0.0) == -p.sv_max
        @test sv(0.01, 0.0) ≈ 0.01 / dt
        @test sv(0.2, 0.2) == 0.0
        # stepping the angle reaches the target without overshoot, then holds still
        δ, target, path = 0.0, 0.1, Float64[]
        for _ in 1:20
            δ += sv(target, δ) * dt
            push!(path, δ)
        end
        @test all(path .<= target + 1e-12)
        @test all(abs.(path[5:end] .- target) .< 1e-12)   # 0.1 rad at 0.032 rad/step: 4 steps
    end

    cin, cout = ref("constraint_in"), ref("constraint_out")
    for i in axes(cin, 1)
        @test steering_constraint(cin[i, 1], cin[i, 2], p.s_min, p.s_max, p.sv_min, p.sv_max) == cout[i, 1]
        @test accl_constraints(cin[i, 3], cin[i, 4], p.v_switch, p.a_max, p.v_min, p.v_max) == cout[i, 2]
    end
end

@testset "Pacejka single-track model" begin
    p = GYM_PARAMS
    g = 9.81
    @testset "matches ForzaETH's model with no drive force" begin
        # their vehicle_dynamics_st_pacejka, same tires and car (reference/forza_ref.py)
        for line in eachline(joinpath(@__DIR__, "reference", "data", "forza_st_pacejka.txt"))
            row = parse.(Float64, split(line))
            x, u, mu, f = SVector{7}(row[1:7]), SVector{2}(row[8:9]), row[10], row[11:17]
            @test vehicle_dynamics_st(x, u, VehicleParams(; mu)) ≈ f rtol = 1e-12 atol = 1e-12
        end
    end

    @testset "small slip: linear bicycle steady state" begin
        # ForzaETH's own check: with the Magic Formula slopes as cornering
        # stiffness, the yaw rate settles where the linear bicycle model says
        x = SVector(0.0, 0.0, 0.02, 3.0, 0.0, 0.0, 0.0)
        u = SVector(0.0, 0.0)
        h = 1e-3
        for _ in 1:3000
            k1 = vehicle_dynamics_st(x, u, p)
            k2 = vehicle_dynamics_st(x + h / 2 * k1, u, p)
            k3 = vehicle_dynamics_st(x + h / 2 * k2, u, p)
            k4 = vehicle_dynamics_st(x + h * k3, u, p)
            x = x + h / 6 * (k1 + 2k2 + 2k3 + k4)
        end
        L = p.lf + p.lr
        Cαf = p.mu * p.Df * p.Bf * p.Cf * p.m * g * p.lr / L
        Cαr = p.mu * p.Dr * p.Br * p.Cr * p.m * g * p.lf / L
        K_us = p.m / L * (p.lr / Cαf - p.lf / Cαr)
        @test x[6] / (x[4] * x[3] / (L + K_us * x[4]^2)) ≈ 1 atol = 0.02
    end

    @testset "tire forces peak at mu * D * Fz" begin
        αs = range(-pi / 2, pi / 2; length=2001)
        @test maximum(abs, F1Tenth._magic_formula.(αs, p.Bf, p.Cf, p.Df, p.Ef)) <= p.Df
        @test maximum(abs, F1Tenth._magic_formula.(αs, p.Br, p.Cr, p.Dr, p.Er)) <= p.Dr
    end

    @testset "drive force is grip limited" begin
        # straight line, full throttle: both axles push equally (open center
        # differential) up to the grip of the weaker axle
        for v in (0.3, 5.0)   # kinematic and dynamic regimes
            x = SVector(0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0)
            f = vehicle_dynamics_st(x, SVector(0.0, p.a_max), p)
            Fzf, Fzr = F1Tenth._axle_loads(p.a_max, p)
            @test f[4] ≈ 2 * min(p.mu * p.Df * Fzf, p.mu * p.Dr * Fzr) / p.m
            @test f[4] < p.a_max
        end
        # well inside the grip limit the car accelerates as asked
        x = SVector(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        @test vehicle_dynamics_st(x, SVector(0.0, 0.5), p)[4] ≈ 0.5
    end

    @testset "friction ellipse: drive force costs lateral grip" begin
        x = SVector(0.0, 0.0, 0.15, 4.0, 0.0, 0.5, -0.05)
        lateral(a) = (f = vehicle_dynamics_st(x, SVector(0.0, a), p);
            v = x[4]; β = x[7]; f[4] * sin(β) + v * cos(β) * f[7] + v * cos(β) * x[6])
        @test abs(lateral(p.a_max)) < abs(lateral(1.0)) < abs(lateral(0.0))
        @test abs(lateral(-p.a_max)) < abs(lateral(0.0))
    end

    @testset "rolling resistance and drag" begin
        q = VehicleParams(; C_roll=0.02, CdA=0.03)
        for v in (0.3, 6.0)
            x = SVector(0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0)
            @test vehicle_dynamics_st(x, SVector(0.0, 0.0), q)[4] ≈ -(0.02 * g + 0.5 * 1.225 * 0.03 * v^2 / q.m)
            @test vehicle_dynamics_st(x, SVector(0.0, 0.0), p)[4] == 0.0   # off by default
        end
        # standing still stays still
        @test vehicle_dynamics_st(zero(SVector{7,Float64}), SVector(0.0, 0.0), q) == zero(SVector{7,Float64})
    end

    # Python asserts > 5000 evaluations/s
    x = SVector(2.0, 0.0, 0.02, 15.0, 0.0, 0.05, 0.003)
    u = SVector(0.15, 0.63 * g)
    n = 100_000
    t = @elapsed for _ in 1:n
        x = vehicle_dynamics_st(x, u, p) * 1e-9 + x
    end
    @test n / t > 5000
end
