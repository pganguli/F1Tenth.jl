using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Symbolics: build_function
using StaticArrays: SVector

@parameters μ C_Sf C_Sr l_f l_r h m I_z δ̇ a
@variables x(t) y(t) δ(t) v(t) ψ(t) ψ̇(t) β(t)

const _MTK_G = 9.81

const _ks_eqs = [
    D(x) ~ v * cos(ψ),
    D(y) ~ v * sin(ψ),
    D(δ) ~ δ̇,
    D(v) ~ a,
    D(ψ) ~ v / (l_f + l_r) * tan(δ),
    D(ψ̇) ~ 0.0,
    D(β) ~ 0.0,
]

const _st_eqs = [
    D(x) ~ v * cos(β + ψ),
    D(y) ~ v * sin(β + ψ),
    D(δ) ~ δ̇,
    D(v) ~ a,
    D(ψ) ~ ψ̇,
    D(ψ̇) ~
        -μ * m / (v * I_z * (l_r + l_f)) *
            (l_f^2 * C_Sf * (_MTK_G * l_r - a * h) +
             l_r^2 * C_Sr * (_MTK_G * l_f + a * h)) * ψ̇ +
        μ * m / (I_z * (l_r + l_f)) *
            (l_r * C_Sr * (_MTK_G * l_f + a * h) -
             l_f * C_Sf * (_MTK_G * l_r - a * h)) * β +
        μ * m / (I_z * (l_r + l_f)) *
            l_f * C_Sf * (_MTK_G * l_r - a * h) * δ,
    D(β) ~
        (μ / (v^2 * (l_r + l_f)) *
            (C_Sr * (_MTK_G * l_f + a * h) * l_r -
             C_Sf * (_MTK_G * l_r - a * h) * l_f) - 1) * ψ̇ -
        μ / (v * (l_r + l_f)) *
            (C_Sr * (_MTK_G * l_f + a * h) +
             C_Sf * (_MTK_G * l_r - a * h)) * β +
        μ / (v * (l_r + l_f)) *
            (C_Sf * (_MTK_G * l_r - a * h)) * δ,
]

const KS_MODE = System(_ks_eqs, t; name=:ks_mode)
const ST_MODE = System(_st_eqs, t; name=:st_mode)

const _state_syms = [x, y, δ, v, ψ, ψ̇, β]
const _param_syms = [μ, C_Sf, C_Sr, l_f, l_r, h, m, I_z, δ̇, a]

function _rhs_expr(eqs, state_syms, param_syms)
    # Passing the RHS list as an SVector (rather than a plain Vector) tells
    # build_function's array codegen to emit a static-array constructor, so
    # the generated function returns an SVector{7,Float64} with zero heap
    # allocations when called with SVector state -- passing a plain Vector
    # here makes it emit a heap-allocating Vector-returning closure instead.
    rhss = SVector{length(eqs)}([eq.rhs for eq in eqs])
    # expression=Val{false} returns a RuntimeGeneratedFunction (RGF), whose
    # call signature is fully dynamic (`args::Vararg{Any,N}`) and therefore
    # allocates on every call regardless of the static-array output type
    # fix above (verified empirically: still 224 bytes/call). Generating
    # the Expr instead (expression=Val{true}) gives us an ordinary,
    # fully-specializable compiled closure once eval'd, which is
    # allocation-free when called with SVector state.
    expr_oop, _ = build_function(rhss, state_syms, param_syms; expression=Val{true})
    return expr_oop
end

# expression=Val{false} (RuntimeGeneratedFunction) constructed from a
# top-level `const` binding is created at *this module's* precompile time;
# its backing Dict cache isn't preserved across precompilation, so calling
# it in a fresh process raises a KeyError. Deferring construction to
# __init__() (which Julia re-runs in every fresh process regardless of
# precompilation) sidesteps this.
#
# We also avoid routing calls through a `Ref{Function}`/wrapper-function
# indirection: `Function` is an abstract type, so a call dispatched through
# a `Ref{Function}` field boxes its arguments and allocates on every call
# (verified empirically: 224 bytes/call), even though the underlying
# generated closure itself is allocation-free when called directly.
# Instead, __mtk_models_init__ `@eval`s the compiled closure straight into
# a top-level `const KS_RHS`/`const ST_RHS` binding -- evaluated fresh in
# every process via __init__, so it never hits the precompile-cache
# problem, and calls go directly through the concrete closure type with no
# extra indirection.
function __mtk_models_init__()
    ks_expr = _rhs_expr(_ks_eqs, _state_syms, _param_syms)
    st_expr = _rhs_expr(_st_eqs, _state_syms, _param_syms)
    @eval const KS_RHS = $ks_expr
    @eval const ST_RHS = $st_expr
    return nothing
end

# KS_RHS(state, p) -> SVector{7,Float64}
#
# Kinematic single-track model RHS. `state` is `(x, y, δ, v, ψ, ψ̇, β)` and
# `p` is `(μ, C_Sf, C_Sr, l_f, l_r, h, m, I_z, δ̇, a)` -- i.e. the 8 physics
# parameters followed by the control input `(steering_velocity,
# acceleration)`. The control input must already be clamped by the caller
# (e.g. via `steering_constraint`/`accl_constraints`) -- this function does
# not re-apply vehicle-specific control-input bounds, since those bounds
# are not part of `p`.
#
# ST_RHS(state, p) -> SVector{7,Float64}
#
# Single-track (dynamic) model RHS. Same `state`/`p` contract as `KS_RHS`.
#
# Both are defined by __mtk_models_init__ at module __init__ time (see
# above); they do not exist as callable bindings until F1Tenth has been
# loaded (which happens automatically -- __init__ runs once per process
# immediately after the module is loaded/precompiled).
