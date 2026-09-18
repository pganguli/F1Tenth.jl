using F1Tenth
using Arrow
using StaticArrays
using Test

include(joinpath(@__DIR__, "..", "async", "testutils.jl"))

function _small_test_track()
    n = 8
    R = 5.0
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    return F1Tenth.build_track(centerline, fill(1.0, n), fill(1.0, n))
end

@testset "TelemetrySink: logs ticks across episodes and round-trips via Arrow" begin
    track = _small_test_track()
    mktempdir() do dir
        path = joinpath(dir, "telemetry.arrow")
        sink = F1Tenth.TelemetrySink(path)

        F1Tenth.init!(sink, track)
        states1 = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        scans1 = [[1.0, 2.0, 3.0]]
        F1Tenth.render!(sink, states1, scans1, [false])
        states2 = [F1Tenth.State(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(sink, states2, scans1, [true])

        F1Tenth.reset!(sink, track)
        states3 = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(sink, states3, scans1, [false])

        F1Tenth.close!(sink)

        tbl = Arrow.Table(path)
        @test length(tbl.tick_index) == 3
        @test collect(tbl.episode_index) == [0, 0, 1]
        @test collect(tbl.tick_index) == [0, 1, 0]
        @test collect(tbl.agent_index) == [1, 1, 1]
        @test collect(tbl.collided) == [false, true, false]
        @test Vector(tbl.state[2]) ≈ [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        @test Vector(tbl.scan[1]) ≈ [1.0, 2.0, 3.0]
    end
end

@testset "TelemetrySink: render! before init! throws" begin
    sink = F1Tenth.TelemetrySink(tempname())
    @test_throws ErrorException F1Tenth.render!(sink, F1Tenth.State[], Vector{Float64}[], Bool[])
end

@testset "TelemetrySink: close! is idempotent" begin
    track = _small_test_track()
    mktempdir() do dir
        sink = F1Tenth.TelemetrySink(joinpath(dir, "telemetry2.arrow"))
        F1Tenth.init!(sink, track)
        F1Tenth.close!(sink)
        F1Tenth.close!(sink)
        @test true
    end
end

if Threads.nthreads() >= 3
    @testset "TelemetrySink: attaches to a real AsyncSim end-to-end" begin
        setup = _async_test_setup()
        mktempdir() do dir
            path = joinpath(dir, "async_telemetry.arrow")
            sink = F1Tenth.TelemetrySink(path)
            # Per TelemetrySink's docstring: callers must call init! themselves
            # before attaching it to an AsyncSim -- the constructor alone does
            # not open the underlying Arrow writer.
            F1Tenth.init!(sink, setup.track)

            moving_states = [F1Tenth.State(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
            sim = F1Tenth.AsyncSim(setup.bvh, setup.params, setup.car, 0.01, setup.cfg,
                moving_states, setup.actions;
                renderers=[F1Tenth.RendererEntry(sink, F1Tenth.DropLatest())])

            F1Tenth.start!(sim)
            sleep(1.0)
            F1Tenth.stop!(sim)
            F1Tenth.close!(sink)

            tbl = Arrow.Table(path)
            @test length(tbl.tick_index) > 0
        end
    end
else
    @warn "Skipping TelemetrySink/AsyncSim attachment test: needs Threads.nthreads() >= 3; got $(Threads.nthreads())"
end
