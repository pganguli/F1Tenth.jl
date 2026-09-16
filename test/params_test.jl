using F1Tenth
using Test

@testset "VehicleParams" begin
    p = F1Tenth.VehicleParams()
    @test p.μ == 1.0489
    @test p.Cᵦᶠ == 4.718
    @test p.Cᵦʳ == 5.4562
    @test p.lᶠ == 0.15875
    @test p.lʳ == 0.17145
    @test p.h == 0.074
    @test p.m == 3.74
    @test p.Iᶻ == 0.04712
    @test p.δ⁻ == -0.4189
    @test p.δ⁺ == 0.4189
    @test p.δ̇⁻ == -3.2
    @test p.δ̇⁺ == 3.2
    @test p.vₛ == 7.319
    @test p.a⁺ == 9.51
    @test p.v⁻ == -5.0
    @test p.v⁺ == 20.0
end

@testset "steering_constraint" begin
    p = F1Tenth.VehicleParams()
    # within bounds: passes through unchanged
    @test F1Tenth.steering_constraint(0.0, 1.0, p) == 1.0
    # at upper angle limit, positive velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.δ⁺, 1.0, p) == 0.0
    # at lower angle limit, negative velocity requested -> zeroed
    @test F1Tenth.steering_constraint(p.δ⁻, -1.0, p) == 0.0
    # requested velocity below δ̇⁻ -> clamped
    @test F1Tenth.steering_constraint(0.0, -10.0, p) == p.δ̇⁻
    # requested velocity above δ̇⁺ -> clamped
    @test F1Tenth.steering_constraint(0.0, 10.0, p) == p.δ̇⁺
end

@testset "accl_constraint" begin
    p = F1Tenth.VehicleParams()
    # within bounds, below vₛ: passes through unchanged
    @test F1Tenth.accl_constraint(1.0, 2.0, p) == 2.0
    # at v⁺, positive accel requested -> zeroed
    @test F1Tenth.accl_constraint(p.v⁺, 1.0, p) == 0.0
    # at v⁻, negative accel requested -> zeroed
    @test F1Tenth.accl_constraint(p.v⁻, -1.0, p) == 0.0
    # requested accel below -a⁺ -> clamped
    @test F1Tenth.accl_constraint(1.0, -100.0, p) == -p.a⁺
    # above vₛ, positive-limit scales down: pos_limit = a⁺*vₛ/v
    v = p.vₛ * 2
    pos_limit = p.a⁺ * p.vₛ / v
    @test F1Tenth.accl_constraint(v, 100.0, p) == pos_limit
end
