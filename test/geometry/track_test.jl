using F1Tenth
using StaticArrays
using LinearAlgebra: norm
using Test

@testset "build_track: segment count and boundary offsets on a large circle" begin
    # Large radius so polygon-tangent curvature error is negligible —
    # isolates the offset math from discretization error.
    R = 100.0
    n = 360
    w = 1.5
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    w_left = fill(w, n)
    w_right = fill(w, n)

    track = F1Tenth.build_track(centerline, w_left, w_right)

    @test length(track.segments) == 2n
    @test length(track.inner) == n
    @test length(track.outer) == n

    # Every outer point should sit ~w further from the origin than the
    # centerline point at the same index; every inner point ~w closer.
    for i in 1:n
        @test isapprox(norm(track.outer[i]), R + w; atol=1e-2)
        @test isapprox(norm(track.inner[i]), R - w; atol=1e-2)
    end
end

@testset "build_track: variable width is respected per-point" begin
    R = 100.0
    n = 360
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    w_left = [i <= n ÷ 2 ? 1.0 : 3.0 for i in 1:n]
    w_right = fill(1.0, n)

    track = F1Tenth.build_track(centerline, w_left, w_right)

    @test isapprox(norm(track.outer[1]), R + 1.0; atol=1e-2)
    @test isapprox(norm(track.outer[n ÷ 2 + 1]), R + 3.0; atol=1e-2)
end

@testset "load_track: reads CSV with header" begin
    mktempdir() do dir
        path = joinpath(dir, "track.csv")
        open(path, "w") do io
            println(io, "x,y,w_left,w_right")
            println(io, "0.0,0.0,1.0,1.0")
            println(io, "1.0,0.0,1.0,1.0")
            println(io, "1.0,1.0,1.0,1.0")
            println(io, "0.0,1.0,1.0,1.0")
        end
        track = F1Tenth.load_track(path)
        @test length(track.centerline) == 4
        @test track.centerline[1] == SVector(0.0, 0.0)
        @test length(track.segments) == 8
    end
end
