using ReachabilityAnalysis

function reachability_analyze(hs, initial_set::Zonotope, input_set::Zonotope,
    tspan::Tuple{Float64,Float64}; alg=TMJets())
    v_lo = low(initial_set)[4]
    v_hi = high(initial_set)[4]
    if v_hi <= 0.5
        location = 1  # :ks
    elseif v_lo >= 0.5
        location = 2  # :st
    else
        throw(ArgumentError(
            "reachability_analyze: initial_set straddles the v = 0.5 KS/ST mode " *
            "boundary (v ∈ [$v_lo, $v_hi]); supply an initial_set fully within one mode"))
    end

    augmented = convert(Zonotope, initial_set × input_set)
    prob = InitialValueProblem(hs, [(location, augmented)])
    return solve(prob; tspan=tspan, alg=alg, max_jumps=4)
end
