using F1Tenth
using GLMakie
using CairoMakie
using Test

include("testutils.jl")

function _glmakie_usable()
    try
        GLMakie.activate!(; visible=false)
        fig = Figure()
        display(fig)
        return true
    catch
        return false
    end
end

if _glmakie_usable()
    @testset "GLMakieRenderer: render! updates per-car Observables" begin
        track = small_test_track()
        renderer = F1Tenth.GLMakieRenderer(track;
            car=F1Tenth.CarFootprint(), cfg=F1Tenth.LiDARConfig(; num_beams=3, fov=1.0))

        states1 = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        scans1 = [[1.0, 2.0, 3.0]]
        F1Tenth.render!(renderer, states1, scans1, [false])

        @test length(renderer.car_polys) == 1
        @test length(renderer.car_polys[1][]) == 4
        @test renderer.trail_points[1] == [Point2f(0.0, 0.0)]
        @test renderer.car_colors[1][] == :blue

        states2 = [F1Tenth.State(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        F1Tenth.render!(renderer, states2, scans1, [true])

        @test renderer.trail_points[1] == [Point2f(0.0, 0.0), Point2f(1.0, 0.0)]
        @test renderer.car_colors[1][] == :red

        F1Tenth.reset!(renderer, track)
        @test isempty(renderer.car_polys)

        F1Tenth.close!(renderer)
    end

    @testset "GLMakieRenderer: render! before init! throws" begin
        track = small_test_track()
        renderer = F1Tenth.GLMakieRenderer(track)
        renderer.initialized = false
        @test_throws ErrorException F1Tenth.render!(renderer, F1Tenth.State[], Vector{Float64}[], Bool[])
        F1Tenth.close!(renderer)
    end

    @testset "GLMakieRenderer: init! activates GL even when CairoMakie last activated" begin
        track = small_test_track()
        # Simulate CairoMakie having last claimed the process-global Makie backend
        # (as its __init__ does), which is exactly the scenario under `using F1Tenth`
        # once GLMakie and CairoMakie are both regular dependencies.
        CairoMakie.activate!()
        @test GLMakie.Makie.current_backend() == CairoMakie

        renderer = F1Tenth.GLMakieRenderer(track)
        @test GLMakie.Makie.current_backend() == GLMakie
        @test renderer.screen isa GLMakie.Screen

        F1Tenth.close!(renderer)
        @test renderer.fig === nothing
        @test renderer.screen === nothing
        @test renderer.initialized == false
        # close! must be idempotent-safe against a second call.
        F1Tenth.close!(renderer)
    end
else
    @warn "Skipping GLMakieRenderer tests: no usable OpenGL context in this environment"
end
