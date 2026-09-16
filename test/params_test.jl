using F1Tenth
using Test

@testset "VehicleParams" begin
    p = F1Tenth.VehicleParams()
    @test p.mu == 1.0489
    @test p.C_Sf == 4.718
    @test p.C_Sr == 5.4562
    @test p.lf == 0.15875
    @test p.lr == 0.17145
    @test p.h == 0.074
    @test p.m == 3.74
    @test p.I == 0.04712
    @test p.s_min == -0.4189
    @test p.s_max == 0.4189
    @test p.sv_min == -3.2
    @test p.sv_max == 3.2
    @test p.v_switch == 7.319
    @test p.a_max == 9.51
    @test p.v_min == -5.0
    @test p.v_max == 20.0
end

@testset "steering_constraint" begin
    p = F1Tenth.VehicleParams()
    # within bounds: passes through unchanged
    @test F1Tenth.steering_constraint(0.0, 1.0, p) == 1.0
    # at upper angle limit, positive velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.s_max, 1.0, p) == 0.0
    # at lower angle limit, negative velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.s_min, -1.0, p) == 0.0
    # requested velocity below sv_min -> clamped
    @test F1Tenth.steering_constraint(0.0, -10.0, p) == p.sv_min
    # requested velocity above sv_max -> clamped
    @test F1Tenth.steering_constraint(0.0, 10.0, p) == p.sv_max
end

@testset "accl_constraints" begin
    p = F1Tenth.VehicleParams()
    # within bounds, below v_switch: passes through unchanged
    @test F1Tenth.accl_constraints(1.0, 2.0, p) == 2.0
    # at v_max, positive accel requested -> zeroed
    @test F1Tenth.accl_constraints(p.v_max, 1.0, p) == 0.0
    # at v_min, negative accel requested -> zeroed
    @test F1Tenth.accl_constraints(p.v_min, -1.0, p) == 0.0
    # requested accel below -a_max -> clamped
    @test F1Tenth.accl_constraints(1.0, -100.0, p) == -p.a_max
    # above v_switch, positive-limit scales down: pos_limit = a_max*v_switch/vel
    vel = p.v_switch * 2
    pos_limit = p.a_max * p.v_switch / vel
    @test F1Tenth.accl_constraints(vel, 100.0, p) == pos_limit
end
