using F1Tenth
using Test

include("testutils.jl")

@testset "AsyncSim: constructor requires one action per agent" begin
    setup = _async_test_setup(; n_agents=2)
    @test_throws ArgumentError F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01,
        setup.cfg, setup.states, [F1Tenth.Action(0.0, 0.0)])
end

@testset "AsyncSim: start! enforces the minimum thread count" begin
    setup = _async_test_setup()
    sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
        setup.states, setup.actions)
    if Threads.nthreads() < 2
        @test_throws ErrorException F1Tenth.start!(sim)
    else
        @warn "Skipping AsyncSim thread-guard test: needs Threads.nthreads() < 2 to exercise the failure path; got $(Threads.nthreads())"
    end
end

if Threads.nthreads() >= 2
    @testset "AsyncSim: start!/stop! lifecycle advances physics and produces output" begin
        setup = _async_test_setup()
        # Uses a nonzero initial velocity, not setup.states/setup.actions
        # (both all-zero): step_dynamics on an all-zero state with a
        # zero-order-held all-zero action is an exact fixed point (the
        # derivative is identically zero), so a "did the state change"
        # assertion against the default setup would always fail
        # regardless of whether the physics tier is ticking correctly.
        moving_states = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            moving_states, setup.actions)

        F1Tenth.start!(sim)
        local frame = missing
        for _ in 1:200
            frame = F1Tenth.take_latest!(sim.output)
            frame === missing || break
            sleep(0.01)
        end
        F1Tenth.stop!(sim)

        @test frame !== missing
        new_states, scans, collisions = frame
        @test new_states[1][1] > moving_states[1][1]  # x position advanced under constant forward velocity
        @test length(scans) == 1
        @test length(collisions) == 1
    end

    @testset "AsyncSim: input actions reach the physics tier (zero-order hold)" begin
        setup = _async_test_setup()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions)

        F1Tenth.start!(sim)
        F1Tenth.put_latest!(sim.input[1], F1Tenth.Action(0.0, 5.0))
        sleep(0.2)
        F1Tenth.stop!(sim)

        @test sim.last_actions[1] == F1Tenth.Action(0.0, 5.0)
        @test sim.states[1][4] > 0.0  # velocity increased under sustained acceleration
    end

    @testset "AsyncSim: tick_hz real-time mode runs without error" begin
        setup = _async_test_setup()
        sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
            setup.states, setup.actions; tick_hz=20.0)

        F1Tenth.start!(sim)
        sleep(0.3)
        F1Tenth.stop!(sim)

        @test F1Tenth.take_latest!(sim.output) !== missing
    end
else
    @warn "Skipping AsyncSim lifecycle tests: needs Threads.nthreads() >= 2; got $(Threads.nthreads())"
end
