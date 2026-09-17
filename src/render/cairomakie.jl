using CairoMakie

mutable struct CairoMakieRendererImpl <: AbstractRenderer
    output_dir::Union{Nothing,String}
    summary_path::String
    car::CarFootprint
    cfg::LiDARConfig
    track::Union{Nothing,Track}
    history_states::Vector{Vector{State}}
    history_scans::Vector{Vector{Vector{Float64}}}
    history_collisions::Vector{Vector{Bool}}
    initialized::Bool
end

function CairoMakieRenderer(track::Track;
    output_dir::Union{Nothing,String}=nothing, summary_path::String="summary.svg",
    car::CarFootprint=CarFootprint(), cfg::LiDARConfig=LiDARConfig())
    renderer = CairoMakieRendererImpl(output_dir, summary_path, car, cfg, nothing,
        Vector{State}[], Vector{Vector{Float64}}[], Vector{Bool}[], false)
    init!(renderer, track)
    return renderer
end

function init!(renderer::CairoMakieRendererImpl, track::Track)
    renderer.track = track
    empty!(renderer.history_states)
    empty!(renderer.history_scans)
    empty!(renderer.history_collisions)
    renderer.output_dir === nothing || mkpath(renderer.output_dir)
    renderer.initialized = true
    return renderer
end

reset!(renderer::CairoMakieRendererImpl, track::Track) = init!(renderer, track)

function render!(renderer::CairoMakieRendererImpl, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    renderer.initialized || error("CairoMakieRenderer: render! called before init!/reset!")
    push!(renderer.history_states, copy(states))
    push!(renderer.history_scans, [copy(s) for s in scans])
    push!(renderer.history_collisions, copy(collisions))
    return nothing
end

function _plot_cairomakie_track!(ax, track::Track)
    outer_pts = [Point2f(p[1], p[2]) for p in vcat(track.outer, [track.outer[1]])]
    inner_pts = [Point2f(p[1], p[2]) for p in vcat(track.inner, [track.inner[1]])]
    lines!(ax, outer_pts; color=:black)
    lines!(ax, inner_pts; color=:black)
end

function _draw_cairomakie_frame(track::Track, trails::Vector{Vector{Point2f}},
    states::Vector{State}, scans::Vector{Vector{Float64}}, collisions::Vector{Bool},
    car::CarFootprint, cfg::LiDARConfig)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    _plot_cairomakie_track!(ax, track)
    for i in eachindex(states)
        isempty(trails[i]) || lines!(ax, trails[i]; color=:gray, linestyle=:dot)
        verts = car_vertices(states[i], car)
        poly!(ax, [Point2f(v[1], v[2]) for v in verts]; color=collisions[i] ? :red : :blue)
        x, y, _, _, ψ, _, _ = states[i]
        num_beams = length(scans[i])
        angles = num_beams == 1 ? [ψ] : range(ψ - cfg.fov / 2, ψ + cfg.fov / 2; length=num_beams)
        for (angle, r) in zip(angles, scans[i])
            lines!(ax, [Point2f(x, y), Point2f(x + r * cos(angle), y + r * sin(angle))];
                color=:red, linewidth=0.5)
        end
    end
    return fig
end

function close!(renderer::CairoMakieRendererImpl)
    track = renderer.track
    n_ticks = length(renderer.history_states)
    n_cars = n_ticks == 0 ? 0 : length(renderer.history_states[1])

    if n_ticks == 0
        summary_fig = Figure()
        ax = Axis(summary_fig[1, 1]; aspect=DataAspect())
        _plot_cairomakie_track!(ax, track)
    else
        full_trails = [Point2f[] for _ in 1:n_cars]
        for tick in 1:n_ticks, i in 1:n_cars
            s = renderer.history_states[tick][i]
            push!(full_trails[i], Point2f(s[1], s[2]))
        end
        summary_fig = _draw_cairomakie_frame(track, full_trails, renderer.history_states[end],
            renderer.history_scans[end], renderer.history_collisions[end], renderer.car, renderer.cfg)
    end
    save(renderer.summary_path, summary_fig; backend=CairoMakie)

    if renderer.output_dir !== nothing
        running_trails = [Point2f[] for _ in 1:n_cars]
        for tick in 1:n_ticks
            for i in 1:n_cars
                s = renderer.history_states[tick][i]
                push!(running_trails[i], Point2f(s[1], s[2]))
            end
            frame_fig = _draw_cairomakie_frame(track, running_trails, renderer.history_states[tick],
                renderer.history_scans[tick], renderer.history_collisions[tick], renderer.car, renderer.cfg)
            path = joinpath(renderer.output_dir, "frame_" * lpad(tick, 4, '0') * ".svg")
            save(path, frame_fig; backend=CairoMakie)
        end
    end
    return nothing
end
