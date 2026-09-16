using StaticArrays
using LinearAlgebra: norm
using NearestNeighbors: KDTree

struct TrackBVH
    tree::KDTree
    segments::Vector{Segment}
    max_half_length::Float64
end

function build_bvh(segments::Vector{Segment})
    midpoints = [(seg.p1 + seg.p2) / 2 for seg in segments]
    tree = KDTree(midpoints)
    max_half_length = maximum(norm(seg.p2 - seg.p1) / 2 for seg in segments)
    return TrackBVH(tree, segments, max_half_length)
end
