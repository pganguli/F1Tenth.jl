using StaticArrays: SVector

function chain_car(p=VehicleParams())
    car = F1Tenth.RaceCar(p, 1; time_step=0.01, integrator=Integrator.RK4)
    return F1Tenth.reset!(car, (0.0, 0.0, 0.0))
end

# Run `steps` 10 ms steps under a fixed command; returns speed and steering histories.
function hold(car, steer, speed, steps)
    v, δ = Float64[], Float64[]
    for _ in 1:steps
        F1Tenth.actuate!(car, steer, speed)
        push!(v, car.state[4])
        push!(δ, car.state[3])
    end
    return v, δ
end

@testset "actuator chain" begin
    p = VehicleParams()

    @testset "motor force constant" begin
        # 1.5 * flux linkage * motor electrical speed per m/s
        @test drive_force_per_amp(p) ≈ 1.5 * 0.000776 * 4800 * 2pi / 60
        @test 60 * drive_force_per_amp(p) ≈ 35.105 atol = 1e-3   # N at the current limit
    end

    @testset "command limiter (ForzaETH's ActuationLimiter)" begin
        a = F1Tenth.ActuatorState(; speed_cmd=10.0, steer_cmd=1.0)
        F1Tenth._limiter_tick!(a, p)
        @test a.speed_out ≈ p.accel_cmd / p.command_rate
        @test a.steer_out ≈ p.steer_rate / p.command_rate
        for _ in 1:1000
            F1Tenth._limiter_tick!(a, p)
        end
        @test a.speed_out == 10.0 && a.steer_out == p.steer_cmd_max   # steering clipped to its range
        a.speed_cmd, a.steer_cmd = 100.0, -1.0
        for _ in 1:1000
            F1Tenth._limiter_tick!(a, p)
        end
        @test a.speed_out == p.speed_cmd_max && a.steer_out == p.steer_cmd_min
        # slowing down uses the deceleration limit
        a.speed_cmd = 0.0
        F1Tenth._limiter_tick!(a, p)
        @test a.speed_out ≈ p.speed_cmd_max - p.decel_cmd / p.command_rate
    end

    @testset "limiter ticks at command_rate" begin
        # ramping towards 10 m/s, each tick adds accel_cmd / command_rate
        car = chain_car()
        hold(car, 0.0, 10.0, 100)   # 1 s
        @test car.actuators.speed_out / (p.accel_cmd / p.command_rate) ≈ 75
    end

    @testset "VESC speed loop" begin
        a = F1Tenth.ActuatorState()
        # below the minimum speed setpoint the PID stops and the windings are
        # shorted: no current at rest, a braking current when turning
        a.speed_out = 0.9 * p.vesc_min_erpm / p.erpm_gain
        for _ in 1:100
            F1Tenth._vesc_step!(a, 0.0, 0.001, p)
        end
        @test a.current == 0.0 && a.i_term == 0.0
        F1Tenth._vesc_step!(a, 1000.0, 0.001, p)
        ω = 1000 * 2pi / 60   # back-EMF over the winding resistance, less a little for its inductance
        @test 0.99 < -a.current / (ω * p.flux_linkage / p.motor_resistance) < 1
        F1Tenth._vesc_step!(a, 0.0, 0.001, p)
        # the setpoint ramps at erpm_ramp, and a large error saturates the current
        a.speed_out = 5.0
        F1Tenth._vesc_step!(a, 0.0, 0.001, p)
        @test a.erpm_set ≈ 0.9 * p.vesc_min_erpm + p.erpm_ramp * 0.001
        for _ in 1:2000
            F1Tenth._vesc_step!(a, 0.0, 0.001, p)
        end
        @test a.current == p.current_max
        F1Tenth._vesc_step!(a, 2 * p.erpm_gain * 5.0, 0.001, p)   # far too fast: brakes
        @test a.current < 0
    end

    @testset "launch follows the command ramp without wheelspin" begin
        car = chain_car()
        v, _ = hold(car, 0.0, 3.0, 800)
        accel = diff([0.0; v]) ./ 0.01
        @test maximum(accel) < 3.0   # the tires could give about 3.4 m/s²
        @test abs(v[100] - 2.5) < 0.2   # 2.5 m/s² ramp, slightly behind
        @test abs(v[end] - 3.0) < 0.01
    end

    @testset "a stop command stops the car" begin
        # the speed loop brakes down to its minimum setpoint, then the shorted
        # windings brake the rest of the way (no creeping on)
        car = chain_car()
        hold(car, 0.0, 3.0, 300)
        v, _ = hold(car, 0.0, 0.0, 200)
        @test v[end] < 1e-3
    end

    @testset "integral action holds speed in a turn" begin
        car = chain_car()
        v, _ = hold(car, 0.15, 2.0, 3000)
        @test abs(v[end] - 2.0) < 1e-3
    end

    @testset "steering: slew, servo lag, command range" begin
        car = chain_car()
        _, δ = hold(car, 0.2, 0.0, 200)
        @test all(diff(δ) .>= -1e-12) && maximum(δ) <= 0.2 + 1e-9   # no overshoot
        @test abs(δ[end] - 0.2) < 1e-3
        @test maximum(abs, diff(δ)) / 0.01 <= p.steer_rate + 1e-9
        _, δ = hold(car, 1.0, 0.0, 300)
        @test abs(δ[end] - p.steer_cmd_max) < 1e-3
    end

    @testset "reset clears the chain" begin
        car = chain_car()
        hold(car, 0.2, 3.0, 50)
        F1Tenth.reset!(car, (0.0, 0.0, 0.0))
        a = car.actuators
        @test (a.speed_out, a.steer_out, a.erpm_set, a.i_term, a.current) == (0.0, 0.0, 0.0, 0.0, 0.0)
    end
end
