using GLMakie

mutable struct GLMakieRendererImpl <: AbstractRenderer
    camera_mode::Symbol
    ego_index::Int
    car::CarFootprint
    cfg::LiDARConfig
    fig::Union{Nothing,Figure}
    ax::Union{Nothing,Axis}
    screen::Union{Nothing,Any}
    half_extent::Float64
    car_polys::Vector{Observable{Vector{Point2f}}}
    trails::Vector{Observable{Vector{Point2f}}}
    trail_points::Vector{Vector{Point2f}}
    scan_lines::Vector{Observable{Vector{Point2f}}}
    car_colors::Vector{Observable{Symbol}}
    initialized::Bool
end

function GLMakieRenderer(track::Track; camera_mode::Symbol=:fixed,
    ego_index::Int=1, car::CarFootprint=CarFootprint(),
    cfg::LiDARConfig=LiDARConfig())
    camera_mode in (:fixed, :follow) ||
        throw(ArgumentError("camera_mode must be :fixed or :follow, got $camera_mode"))
    renderer = GLMakieRendererImpl(camera_mode, ego_index, car, cfg, nothing, nothing, nothing, 0.0,
        Observable{Vector{Point2f}}[], Observable{Vector{Point2f}}[], Vector{Point2f}[],
        Observable{Vector{Point2f}}[], Observable{Symbol}[], false)
    init!(renderer, track)
    return renderer
end

function _plot_glmakie_track!(ax, track::Track)
    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
    lines!(ax, outer_pts; color=:black)
    lines!(ax, inner_pts; color=:black)
end

function _glmakie_track_bounds(track::Track)
    pts = vcat(track.outer, track.inner)
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    return (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
end

function init!(renderer::GLMakieRendererImpl, track::Track)
    GLMakie.activate!()
    if renderer.fig === nothing
        renderer.fig = Figure()
        renderer.ax = Axis(renderer.fig[1, 1]; aspect=DataAspect())
        renderer.screen = display(renderer.fig)
    else
        empty!(renderer.ax)
    end
    _plot_glmakie_track!(renderer.ax, track)
    xmin, xmax, ymin, ymax = _glmakie_track_bounds(track)
    renderer.half_extent = max(xmax - xmin, ymax - ymin) / 2
    if renderer.camera_mode == :fixed
        limits!(renderer.ax, xmin, xmax, ymin, ymax)
    end
    empty!(renderer.car_polys)
    empty!(renderer.trails)
    empty!(renderer.trail_points)
    empty!(renderer.scan_lines)
    empty!(renderer.car_colors)
    renderer.initialized = true
    return renderer
end

reset!(renderer::GLMakieRendererImpl, track::Track) = init!(renderer, track)

function _ensure_glmakie_car_plots!(renderer::GLMakieRendererImpl, n::Int)
    isempty(renderer.car_polys) || return
    for _ in 1:n
        car_obs = Observable(Point2f[])
        trail_obs = Observable(Point2f[])
        scan_obs = Observable(Point2f[])
        color_obs = Observable(:blue)
        poly!(renderer.ax, car_obs; color=color_obs)
        lines!(renderer.ax, trail_obs; color=:gray, linestyle=:dot)
        lines!(renderer.ax, scan_obs; color=:red, linewidth=0.5)
        push!(renderer.car_polys, car_obs)
        push!(renderer.trails, trail_obs)
        push!(renderer.trail_points, Point2f[])
        push!(renderer.scan_lines, scan_obs)
        push!(renderer.car_colors, color_obs)
    end
end

function _glmakie_scan_ray_points(state::State, beam_ranges::Vector{Float64}, cfg::LiDARConfig)
    x, y, _, _, ψ, _, _ = state
    angles = cfg.num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=cfg.num_beams)
    points = Point2f[]
    for (angle, r) in zip(angles, beam_ranges)
        push!(points, Point2f(x, y))
        push!(points, Point2f(x + r * cos(angle), y + r * sin(angle)))
        push!(points, Point2f(NaN, NaN))
    end
    return points
end

function render!(renderer::GLMakieRendererImpl, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    renderer.initialized || error("GLMakieRenderer: render! called before init!/reset!")
    n = length(states)
    _ensure_glmakie_car_plots!(renderer, n)
    for i in 1:n
        verts = car_vertices(states[i], renderer.car)
        renderer.car_polys[i][] = [Point2f(v[1], v[2]) for v in verts]
        push!(renderer.trail_points[i], Point2f(states[i][1], states[i][2]))
        renderer.trails[i][] = renderer.trail_points[i]
        renderer.car_colors[i][] = collisions[i] ? :red : :blue
        renderer.scan_lines[i][] = _glmakie_scan_ray_points(states[i], scans[i], renderer.cfg)
    end
    if renderer.camera_mode == :follow
        1 <= renderer.ego_index <= length(states) ||
            error("GLMakieRenderer: ego_index=$(renderer.ego_index) out of range for $(length(states)) agents")
        ex, ey = states[renderer.ego_index][1], states[renderer.ego_index][2]
        h = renderer.half_extent
        limits!(renderer.ax, ex - h, ex + h, ey - h, ey + h)
    end
    return nothing
end

function close!(renderer::GLMakieRendererImpl)
    renderer.screen === nothing || close(renderer.screen)
    renderer.fig = nothing
    renderer.screen = nothing
    renderer.initialized = false
    return nothing
end
