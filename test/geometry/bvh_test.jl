using F1Tenth
using StaticArrays
using NearestNeighbors: inrange
using Test

@testset "build_bvh: inrange query finds the expected segment" begin
    segments = [
        F1Tenth.Segment(SVector(0.0, 0.0), SVector(1.0, 0.0)),
        F1Tenth.Segment(SVector(10.0, 0.0), SVector(11.0, 0.0)),
    ]
    bvh = F1Tenth.build_bvh(segments)

    @test bvh.max_half_length == 0.5

    near_first = inrange(bvh.tree, SVector(0.5, 0.0), 1.0)
    @test near_first == [1]

    near_second = inrange(bvh.tree, SVector(10.5, 0.0), 1.0)
    @test near_second == [2]

    near_neither = inrange(bvh.tree, SVector(5.0, 0.0), 1.0)
    @test isempty(near_neither)
end
