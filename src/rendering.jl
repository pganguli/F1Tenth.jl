# Render window types, ported from f110_gym/envs/rendering.py. The drawing
# itself is in ext/F1TenthGLMakieExt.jl, loaded by `using GLMakie`, so
# headless use never pays GLMakie's load time.

const CAR_LENGTH = 0.58      # car shape drawn by the renderer (fixed, as in Python)
const CAR_WIDTH = 0.31
const RENDER_SCALE = 50.0    # renderer units per meter, as in Python
const ZOOM_IN_FACTOR = 1.2
const ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

"""
    ScoreLabel

The text label in the render window, like Python's pyglet `Label`: set
`text`, `x`, `y` (renderer units) from a render callback.
"""
mutable struct ScoreLabel
    text::String
    x::Float64
    y::Float64
end

"""
    EnvRenderer

The render window, passed to render callbacks. Like Python's renderer it
works in renderer units (50 × meters). Callbacks may set the camera (`left`,
`right`, `bottom`, `top`) and `score_label`, read `cars[i][]` (corner points
of car `i`), `poses` (`num_agents × 3`, meters) and `ego_idx`, and draw on
`axis` with GLMakie (where Python callbacks add to `batch`).
"""
mutable struct EnvRenderer
    width::Int
    height::Int
    left::Float64
    right::Float64
    bottom::Float64
    top::Float64
    zoom_level::Float64
    zoomed_width::Float64
    zoomed_height::Float64
    score_label::ScoreLabel
    ego_idx::Int
    poses::Union{Nothing,Matrix{Float64}}
    cars::Vector{Any}
    figure::Any
    axis::Any
    screen::Any
    state::Any   # backend-private
end

# Implemented by the GLMakie extension.
const _NEED_GLMAKIE = "Rendering needs GLMakie: run `using GLMakie` (after `import Pkg; Pkg.add(\"GLMakie\")`)."
_make_renderer(args...) = error(_NEED_GLMAKIE)
_renderer_update_map!(args...) = error(_NEED_GLMAKIE)
_renderer_update_obs!(args...) = error(_NEED_GLMAKIE)
_renderer_draw!(args...) = error(_NEED_GLMAKIE)
