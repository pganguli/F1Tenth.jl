using ReachabilityAnalysis

# Augmented state order shared by both modes: (x, y, δ, v, ψ, ψ̇, β, a, δ̇).
# `a`/`δ̇` (Phase 1's control inputs) are appended as extra
# zero-derivative dimensions -- TMJets has no separate time-varying
# input channel, so bounding them via the initial set is the standard
# state-augmentation technique for this kind of Taylor-model
# reachability (spec Decision 1/2). This means `a`/`δ̇` stay within
# their initial box for the whole analysis horizon rather than varying
# independently within it at every instant -- an accepted
# over-approximation, not an under-approximation, so still sound.
const _AUGMENTED_DIM = 9

# These two functions are hand-written mirrors of `_ks_eqs`/`_st_eqs`
# in src/dynamics/mtk_models.jl -- NOT calls into KS_RHS/ST_RHS/
# KS_MODE/ST_MODE. See this file's F1Tenth.jl-level comment header (and
# the spec's amended Decision 1) for why: ReachabilityAnalysis.jl's
# `@taylorize` is a macro operating on function *source code*, and
# KS_RHS/ST_RHS are `@eval`'d into existence at runtime from a
# build_function-generated Expr -- there is no source form for
# `@taylorize` to macroexpand. Keeping these in sync with
# mtk_models.jl's equations by hand is a real, accepted maintenance
# cost; this task's own tests (above) cross-check both mirrors against
# KS_RHS/ST_RHS at sample points to catch silent divergence.
function _ks_taylor_expr(fname::Symbol, params::VehicleParams)
    lf = params.lᶠ
    lr = params.lʳ
    return quote
        @taylorize function $(fname)(du, u, p, t)
            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
            du[1] = v * cos(ψ)
            du[2] = v * sin(ψ)
            du[3] = δ̇
            du[4] = a
            du[5] = v / ($lf + $lr) * tan(δ)
            # Matches step.jl's corrected KS ψ̈ term (dynamics_rhs's fallback
            # branch), NOT KS_RHS's own zeroed simplification -- see the
            # Fix 1 discussion above: a flowpipe that starts in :ks and
            # transitions into :st must not carry a frozen, stale ψ̇, since
            # ψ̇ is a live state in :st (feeds its du[6]/du[7]).
            du[6] = a / ($lf + $lr) * tan(δ) + v / (($lf + $lr) * cos(δ)^2) * δ̇
            du[7] = zero(v)
            du[8] = zero(v)
            du[9] = zero(v)
            return du
        end
    end
end

function _st_taylor_expr(fname::Symbol, params::VehicleParams)
    μ = params.μ
    Cbf = params.Cᵦᶠ
    Cbr = params.Cᵦʳ
    lf = params.lᶠ
    lr = params.lʳ
    h = params.h
    m = params.m
    Iz = params.Iᶻ
    g = _MTK_G  # reuse mtk_models.jl's gravity constant rather than a second hardcoded copy
    return quote
        @taylorize function $(fname)(du, u, p, t)
            x, y, δ, v, ψ, ψ̇, β, a, δ̇ = u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9]
            du[1] = v * cos(β + ψ)
            du[2] = v * sin(β + ψ)
            du[3] = δ̇
            du[4] = a
            du[5] = ψ̇
            du[6] = (-$μ * $m / (v * $Iz * ($lr + $lf)) *
                        ($lf^2 * $Cbf * ($g * $lr - a * $h) +
                         $lr^2 * $Cbr * ($g * $lf + a * $h)) * ψ̇ +
                     $μ * $m / ($Iz * ($lr + $lf)) *
                        ($lr * $Cbr * ($g * $lf + a * $h) -
                         $lf * $Cbf * ($g * $lr - a * $h)) * β +
                     $μ * $m / ($Iz * ($lr + $lf)) *
                        $lf * $Cbf * ($g * $lr - a * $h) * δ)
            du[7] = (($μ / (v^2 * ($lr + $lf)) *
                         ($Cbr * ($g * $lf + a * $h) * $lr -
                          $Cbf * ($g * $lr - a * $h) * $lf) - 1) * ψ̇ -
                     $μ / (v * ($lr + $lf)) *
                         ($Cbr * ($g * $lf + a * $h) +
                          $Cbf * ($g * $lr - a * $h)) * β +
                     $μ / (v * ($lr + $lf)) *
                         ($Cbf * ($g * $lr - a * $h)) * δ)
            du[8] = zero(v)
            du[9] = zero(v)
            return du
        end
    end
end

# NOTE on repeated eval: `_build_mode` (via `build_hybrid_system`) `gensym`s
# and `Core.eval`s a brand-new `@taylorize`d function into this module on
# every call, permanently retaining a uniquely-named global function each
# time. This is intentionally uncached: `build_hybrid_system` is an offline,
# standalone reachability-analysis entry point per the spec's
# Decision/Non-goals -- it is explicitly NOT wired into `AsyncSim`'s
# per-tick loop (out of scope for this phase) -- so it is expected to be
# called once, or a handful of times, per offline analysis session rather
# than in a hot loop. If a future phase calls this in a loop or performance
# becomes a concern, memoizing the built `HybridSystem` on `params` (e.g.
# via a `Dict`) would be a straightforward fix at that point.
function _build_mode(fname::Symbol, taylor_expr::Expr, invariant, dim::Int)
    Core.eval(@__MODULE__, taylor_expr)
    sys_expr = :(@system(u' = $(fname)(u), dim: $dim, u ∈ $invariant))
    return Core.eval(@__MODULE__, sys_expr)
end

"""
    build_hybrid_system(params::VehicleParams) -> ReachabilityAnalysis.HybridSystem

Build a 2-location hybrid automaton over the 9-D augmented state
`(x, y, δ, v, ψ, ψ̇, β, a, δ̇)`, switching on the longitudinal velocity `v`
(index 4) at the KS/ST threshold `v = 0.5`:

- location `1` = `:ks` (kinematic single-track, active for `v <= 0.5`)
- location `2` = `:st` (dynamic single-track, active for `v >= 0.5`)

`a`/`δ̇` are carried as extra zero-derivative dimensions (state-augmented
control inputs); see the `_AUGMENTED_DIM` comment above for why.
"""
function build_hybrid_system(params::VehicleParams)
    ks_name = gensym(:phase5_ks)
    st_name = gensym(:phase5_st)

    # v is index 4 of the 9-D augmented state.
    v_ks_invariant = HalfSpace([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5)   # v <= 0.5
    v_st_invariant = HalfSpace([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.5) # v >= 0.5

    ks_mode = _build_mode(ks_name, _ks_taylor_expr(ks_name, params), v_ks_invariant, _AUGMENTED_DIM)
    st_mode = _build_mode(st_name, _st_taylor_expr(st_name, params), v_st_invariant, _AUGMENTED_DIM)

    automaton = GraphAutomaton(2)
    add_transition!(automaton, 1, 2, 1)  # :ks -> :st, v crosses 0.5 upward
    add_transition!(automaton, 2, 1, 2)  # :st -> :ks, v crosses 0.5 downward

    reset_ks_to_st = ConstrainedIdentityMap(_AUGMENTED_DIM, v_st_invariant)
    reset_st_to_ks = ConstrainedIdentityMap(_AUGMENTED_DIM, v_ks_invariant)

    return HybridSystem(automaton, [ks_mode, st_mode],
        [reset_ks_to_st, reset_st_to_ks],
        [AutonomousSwitching(), AutonomousSwitching()])
end
