using ReachabilityAnalysis

function check_flowpipe_safety(sol, track::Track)
    corridor = UnionSetArray(to_lazysets(track))
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
