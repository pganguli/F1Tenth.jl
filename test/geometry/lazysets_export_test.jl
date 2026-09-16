using F1Tenth
using StaticArrays
using LazySets: VPolygon, an_element, ∈
using Test

@testset "to_lazysets: one convex polygon per centerline slice" begin
    R = 100.0
    n = 360
    w = 1.5
    centerline = [SVector(R * cos(θ), R * sin(θ)) for θ in range(0.0, 2π; length=n + 1)[1:end-1]]
    track = F1Tenth.build_track(centerline, fill(w, n), fill(w, n))

    polys = F1Tenth.to_lazysets(track)

    @test length(polys) == n
    @test all(p -> p isa VPolygon, polys)

    # The centerline waypoint itself must lie inside (or on the boundary
    # of) its own slice polygon.
    @test track.centerline[1] ∈ polys[1] || track.centerline[1] ∈ polys[n]
end
