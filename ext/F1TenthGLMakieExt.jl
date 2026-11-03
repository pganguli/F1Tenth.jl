# The render window, ported from f110_gym/envs/rendering.py (pyglet) to
# GLMakie: the same camera (left/right/bottom/top in renderer units, drag to
# pan, scroll to zoom about the mouse), colors, cars, label and FPS readout.
module F1TenthGLMakieExt

using F1Tenth
using F1Tenth: EnvRenderer, ScoreLabel, MapData, Observation, get_vertices, RENDER_SCALE, CAR_LENGTH,
    CAR_WIDTH, ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR
using GLMakie
using Printf: @sprintf

const BG_COLOR = RGBf(9 / 255, 32 / 255, 87 / 255)
const MAP_COLOR = RGBAf(183 / 255, 193 / 255, 222 / 255, 1.0)
const EGO_COLOR = RGBf(172 / 255, 97 / 255, 185 / 255)
const OPP_COLOR = RGBf(99 / 255, 52 / 255, 94 / 255)

mutable struct GLState
    map_plot::Any
    label_pos::Observable{Point2f}
    label_text::Observable{String}
    fps_text::Observable{String}
    limits::NTuple{4,Float64}   # last camera applied to the axis
    frames::Int
    fps_t0::Float64
    dragging::Bool
    last_mouse::Point2f
end

function F1Tenth._make_renderer(width::Int, height::Int)
    fig = Figure(; size=(width, height), backgroundcolor=BG_COLOR, figure_padding=0)
    # autolimitaspect: keep pixels square, widening a camera area that doesn't
    # match the window's shape instead of stretching it (Python stretches)
    ax = Axis(fig[1, 1]; backgroundcolor=BG_COLOR, autolimitaspect=1)
    hidedecorations!(ax)
    hidespines!(ax)
    foreach(name -> deregister_interaction!(ax, name), collect(keys(interactions(ax))))

    label = ScoreLabel(@sprintf("Lap Time: %.2f, Ego Lap Count: %.0f", 0.0, 0.0), 0.0, -800.0)
    st = GLState(nothing, Observable(Point2f(label.x, label.y)), Observable(label.text), Observable(""),
        (NaN, NaN, NaN, NaN), 0, time(), false, Point2f(0, 0))
    label_plot = text!(ax, st.label_pos; text=st.label_text, fontsize=48, markerspace=:data,
        align=(:center, :center), color=:white)
    translate!(label_plot, 0, 0, 2)
    text!(fig.scene, Point2f(8, 8); text=st.fps_text, space=:pixel, fontsize=24, color=RGBAf(1, 1, 1, 0.5),
        align=(:left, :bottom))

    screen = GLMakie.Screen(; title="F1Tenth Gym", vsync=false, framerate=1000.0, focus_on_show=false,
        scalefactor=_scalefactor())
    display(screen, fig)
    r = EnvRenderer(width, height, -width / 2, width / 2, -height / 2, height / 2, 1.2, width, height, label, 1,
        nothing, Any[], fig, ax, screen, st)
    _register_events!(r)
    # the window manager may not honor the requested size; start from the real one
    w, h = widths(events(fig).window_area[])
    w > 0 && h > 0 ? _on_resize!(r, w, h) : _on_resize!(r, width, height)
    return r
end

# HiDPI scale: GLMakie's own (the system's, or one set with
# GLMakie.activate!(scalefactor=...)), else, when the system reports none (as
# X11 often does), the monitor's DPI over 96, rounded.
function _scalefactor()
    configured = Makie.current_default_theme()[:GLMakie][:scalefactor][]
    configured isa Makie.Automatic || return configured
    monitor = GLMakie.GLFW.GetPrimaryMonitor()
    xscale, _ = GLMakie.GLFW.GetMonitorContentScale(monitor)
    xscale > 1 && return Float64(xscale)
    width_mm = GLMakie.GLFW.GetMonitorPhysicalSize(monitor).width
    width_mm > 0 || return 1.0
    dpi = GLMakie.GLFW.GetVideoMode(monitor).width / (width_mm / 25.4)
    return max(1.0, round(dpi / 96))
end

# Window resize resets the camera to the origin at the current zoom, as in Python.
function _on_resize!(r::EnvRenderer, width, height)
    r.width, r.height = width, height
    r.left, r.right = -r.zoom_level * width / 2, r.zoom_level * width / 2
    r.bottom, r.top = -r.zoom_level * height / 2, r.zoom_level * height / 2
    r.zoomed_width, r.zoomed_height = r.zoom_level * width, r.zoom_level * height
    return _apply_camera!(r)
end

function _on_mouse_drag!(r::EnvRenderer, dx, dy)
    r.left -= dx * r.zoom_level
    r.right -= dx * r.zoom_level
    r.bottom -= dy * r.zoom_level
    r.top -= dy * r.zoom_level
    return _apply_camera!(r)
end

function _on_mouse_scroll!(r::EnvRenderer, x, y, dy)
    f = dy > 0 ? ZOOM_IN_FACTOR : dy < 0 ? ZOOM_OUT_FACTOR : 1.0
    0.01 < r.zoom_level * f < 10 || return r
    r.zoom_level *= f
    mouse_x, mouse_y = x / r.width, y / r.height
    mouse_x_in_world = r.left + mouse_x * r.zoomed_width
    mouse_y_in_world = r.bottom + mouse_y * r.zoomed_height
    r.zoomed_width *= f
    r.zoomed_height *= f
    r.left = mouse_x_in_world - mouse_x * r.zoomed_width
    r.right = mouse_x_in_world + (1 - mouse_x) * r.zoomed_width
    r.bottom = mouse_y_in_world - mouse_y * r.zoomed_height
    r.top = mouse_y_in_world + (1 - mouse_y) * r.zoomed_height
    return _apply_camera!(r)
end

function _register_events!(r::EnvRenderer)
    ev, st = events(r.figure), r.state
    on(ev.window_area) do area
        w, h = widths(area)
        w > 0 && h > 0 && (w, h) != (r.width, r.height) && _on_resize!(r, w, h)
    end
    on(ev.mousebutton) do e
        if e.action == Mouse.press
            st.dragging = true
            st.last_mouse = Point2f(ev.mouseposition[])
        elseif isempty(ev.mousebuttonstate)
            st.dragging = false
        end
    end
    on(ev.mouseposition) do pos
        st.dragging || return
        dx, dy = Point2f(pos) - st.last_mouse
        st.last_mouse = Point2f(pos)
        _on_mouse_drag!(r, dx, dy)
    end
    on(ev.scroll) do (_, dy)
        _on_mouse_scroll!(r, ev.mouseposition[]..., dy)
    end
    return r
end

function _apply_camera!(r::EnvRenderer)
    camera = (r.left, r.right, r.bottom, r.top)
    if camera != r.state.limits
        r.state.limits = camera
        limits!(r.axis, camera...)
    end
    return r
end

# Obstacle pixels (exactly 0 in the image, as Python draws them) as one
# texture, each pixel centered where Python puts its GL point.
function F1Tenth._renderer_update_map!(r::EnvRenderer, m::MapData)
    h, w = size(m.raw)
    colors = [m.raw[row, col] == 0x00 ? MAP_COLOR : RGBAf(0, 0, 0, 0) for col in 1:w, row in 1:h]
    res, ox, oy = m.resolution, m.origin[1], m.origin[2]
    xs = RENDER_SCALE * (ox - res / 2) .. RENDER_SCALE * (ox + (w - 0.5) * res)
    ys = RENDER_SCALE * (oy - res / 2) .. RENDER_SCALE * (oy + (h - 0.5) * res)
    r.state.map_plot = image!(r.axis, xs, ys, colors; interpolate=false)
    translate!(r.state.map_plot, 0, 0, -1)
    return r
end

function _car_points(x, y, th)
    v = get_vertices((x, y, th), CAR_LENGTH, CAR_WIDTH)
    # a plain Vector: Makie rejects static vectors of points
    return Point2f[Point2f(RENDER_SCALE * v[k][1], RENDER_SCALE * v[k][2]) for k in 1:4]::Vector{Point2f}
end

function F1Tenth._renderer_update_obs!(r::EnvRenderer, obs::Observation)
    r.ego_idx = obs.ego_idx
    n = length(obs.poses_x)
    if r.poses === nothing   # first frame: one polygon per car
        for i in 1:n
            pts = Observable(_car_points(0.0, 0.0, 0.0))
            poly!(r.axis, pts; color=i == r.ego_idx ? EGO_COLOR : OPP_COLOR)
            push!(r.cars, pts)
        end
        r.poses = zeros(n, 3)
    end
    for j in 1:n
        r.poses[j, :] .= (obs.poses_x[j], obs.poses_y[j], obs.poses_theta[j])
        r.cars[j][] = _car_points(obs.poses_x[j], obs.poses_y[j], obs.poses_theta[j])
    end
    return r
end

function F1Tenth._renderer_draw!(r::EnvRenderer)
    isopen(r.screen) || error("Rendering window was closed.")
    st = r.state
    _apply_camera!(r)
    label = r.score_label
    Point2f(label.x, label.y) == st.label_pos[] || (st.label_pos[] = Point2f(label.x, label.y))
    label.text == st.label_text[] || (st.label_text[] = label.text)
    st.frames += 1
    t = time()
    if t - st.fps_t0 >= 0.25
        st.fps_text[] = @sprintf("%.2f", st.frames / (t - st.fps_t0))
        st.frames, st.fps_t0 = 0, t
    end
    return r
end

end
