using StaticArrays
using LinearAlgebra: norm
using DelimitedFiles: readdlm

struct Segment
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
end

struct Track
    centerline::Vector{SVector{2,Float64}}
    w_left::Vector{Float64}
    w_right::Vector{Float64}
    inner::Vector{SVector{2,Float64}}
    outer::Vector{SVector{2,Float64}}
    segments::Vector{Segment}
end

function build_track(centerline::Vector{SVector{2,Float64}},
    w_left::Vector{Float64}, w_right::Vector{Float64})
    n = length(centerline)
    inner = Vector{SVector{2,Float64}}(undef, n)
    outer = Vector{SVector{2,Float64}}(undef, n)
    for i in 1:n
        prev = centerline[mod1(i - 1, n)]
        next = centerline[mod1(i + 1, n)]
        tangent = next - prev
        tangent = tangent / norm(tangent)
        # This assumes the centerline is wound counter-clockwise (CCW): for a
        # CCW-wound loop, rotating the tangent by -90 degrees points outward,
        # so `outer`/`w_left` correctly land on the geometrically-outward
        # boundary and `inner`/`w_right` on the inward one. For a
        # clockwise-wound centerline, inner and outer swap.
        normal_outward = SVector(tangent[2], -tangent[1])
        outer[i] = centerline[i] + normal_outward * w_left[i]
        inner[i] = centerline[i] - normal_outward * w_right[i]
    end
    segments = Segment[]
    for i in 1:n
        j = mod1(i + 1, n)
        push!(segments, Segment(outer[i], outer[j]))
        push!(segments, Segment(inner[i], inner[j]))
    end
    return Track(centerline, w_left, w_right, inner, outer, segments)
end

function load_track(path::AbstractString)
    raw = readdlm(path, ',')
    start_row = all(x -> x isa Number, raw[1, 1:4]) ? 1 : 2
    data = Float64.(raw[start_row:end, 1:4])
    n = size(data, 1)
    centerline = [SVector(data[i, 1], data[i, 2]) for i in 1:n]
    return build_track(centerline, data[:, 3], data[:, 4])
end
