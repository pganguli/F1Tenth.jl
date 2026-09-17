using F1Tenth
using CairoMakie
using Test

include("testutils.jl")

@testset "CairoMakieRenderer: full lifecycle produces summary and frame SVGs" begin
    track = small_test_track()
    mktempdir() do dir
        summary_path = joinpath(dir, "summary.svg")
        frames_dir = joinpath(dir, "frames")
        renderer = F1Tenth.CairoMakieRenderer(track; output_dir=frames_dir, summary_path=summary_path,
            car=F1Tenth.CarFootprint(), cfg=F1Tenth.LiDARConfig(; num_beams=3, fov=1.0))

        for tick in 1:3
            states = [F1Tenth.State(Float64(tick), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            scans = [[1.0, 2.0, 3.0]]
            F1Tenth.render!(renderer, states, scans, [tick == 3])
        end
        F1Tenth.close!(renderer)

        @test isfile(summary_path)
        @test filesize(summary_path) > 0
        content = read(summary_path, String)
        @test startswith(content, "<?xml") || startswith(content, "<svg")

        frame_files = readdir(frames_dir)
        @test length(frame_files) == 3
    end
end

@testset "CairoMakieRenderer: close! with zero ticks still emits a track-only summary" begin
    track = small_test_track()
    mktempdir() do dir
        summary_path = joinpath(dir, "summary.svg")
        renderer = F1Tenth.CairoMakieRenderer(track; summary_path=summary_path)
        F1Tenth.close!(renderer)
        @test isfile(summary_path)
    end
end

@testset "CairoMakieRenderer: render! before init! throws" begin
    track = small_test_track()
    renderer = F1Tenth.CairoMakieRenderer(track)
    renderer.initialized = false
    @test_throws ErrorException F1Tenth.render!(renderer, F1Tenth.State[], Vector{Float64}[], Bool[])
end
