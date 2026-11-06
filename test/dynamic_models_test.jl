using StaticArrays: SVector

@testset "steering constraint (match Python outputs)" begin
    p = PY_PARAMS
    cin, cout = ref("constraint_in"), ref("constraint_out")
    for i in axes(cin, 1)
        @test steering_constraint(cin[i, 1], cin[i, 2], p.s_min, p.s_max, p.sv_min, p.sv_max) == cout[i, 1]
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
        # straight line, full motor current: both axles push equally (open
        # center differential) up to the grip of the weaker axle
        a_full = p.current_max * drive_force_per_amp(p) / p.m
        for v in (0.3, 5.0)   # kinematic and dynamic regimes
            x = SVector(0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0)
            f = vehicle_dynamics_st(x, SVector(0.0, a_full), p)
            Fzf, Fzr = F1Tenth._axle_loads(a_full, p)
            @test f[4] ≈ 2 * min(p.mu * p.Df * Fzf, p.mu * p.Dr * Fzr) / p.m
            @test f[4] < a_full
        end
        # well inside the grip limit the car accelerates as asked
        x = SVector(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
        @test vehicle_dynamics_st(x, SVector(0.0, 0.5), p)[4] ≈ 0.5
    end

    @testset "friction ellipse: drive force costs lateral grip" begin
        x = SVector(0.0, 0.0, 0.15, 4.0, 0.0, 0.5, -0.05)
        lateral(a) = (f = vehicle_dynamics_st(x, SVector(0.0, a), p);
            v = x[4]; β = x[7]; f[4] * sin(β) + v * cos(β) * f[7] + v * cos(β) * x[6])
        @test abs(lateral(9.0)) < abs(lateral(1.0)) < abs(lateral(0.0))
        @test abs(lateral(-9.0)) < abs(lateral(0.0))
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
