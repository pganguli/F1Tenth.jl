# The render window types, a port of f110_gym/envs/rendering.py. The drawing
# code is in ext/F1TenthGLMakieExt.jl, which `using GLMakie` loads. Thus use
# without a display does not need the load time of GLMakie.

const RENDER_SCALE = 50.0    # renderer units for each meter, as in the Python code
const ZOOM_IN_FACTOR = 1.2
const ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

"""
    ScoreLabel

The text label in the render window, as the pyglet `Label` of the Python
code. To change it, set `text`, `x` and `y` (renderer units) in a render
callback.
"""
mutable struct ScoreLabel
    text::String
    x::Float64
    y::Float64
end

"""
    EnvRenderer

The render window, which the render callbacks get. As the renderer of the
Python code, it uses renderer units (50 × meters). The callbacks can:

- set the camera (`left`, `right`, `bottom` and `top`) and `score_label`
- read `cars[i][]` (the corner points of car `i`), `poses`
  (`num_agents × 3`, meters) and `ego_idx`
- draw on `axis` with GLMakie (the callbacks of the Python code add to
  `batch`)
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
    state::Any   # private to the backend
end

# The GLMakie extension gives these functions.
const _NEED_GLMAKIE = "Rendering needs GLMakie: run `using GLMakie` (after `import Pkg; Pkg.add(\"GLMakie\")`)."
_make_renderer(args...) = error(_NEED_GLMAKIE)
_renderer_update_map!(args...) = error(_NEED_GLMAKIE)
_renderer_update_obs!(args...) = error(_NEED_GLMAKIE)
_renderer_draw!(args...) = error(_NEED_GLMAKIE)
