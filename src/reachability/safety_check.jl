using ReachabilityAnalysis

function check_flowpipe_safety(sol, track::Track)
    isempty(sol.F) &&
        throw(ArgumentError("check_flowpipe_safety: sol.F is empty -- no flowpipe segments to check"))
    lazy_corridor = to_lazysets(track)
    isempty(lazy_corridor) &&
        throw(ArgumentError("check_flowpipe_safety: to_lazysets(track) is empty -- no corridor to check against"))
    corridor = UnionSetArray(lazy_corridor)
    violating = Int[]
    idx = 0
    for fp in sol.F
        for rs in fp
            idx += 1
            box = overapproximate(set(rs), Hyperrectangle)
            xy_box = Hyperrectangle(low=low(box)[1:2], high=high(box)[1:2])
            issubset(xy_box, corridor) || push!(violating, idx)
        end
    end
    return (isempty(violating), violating)
end
