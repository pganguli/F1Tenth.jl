using Arrow

"""
    TelemetrySink(path::String) <: AbstractRenderer

An Arrow-backed telemetry sink: each `render!` call appends one row per
agent to an Arrow IPC stream file at `path`.

Unlike `GLMakieRenderer`/`CairoMakieRenderer`, whose constructors take a
`track` and call `init!` eagerly, `TelemetrySink(path)` alone does NOT open
the underlying Arrow writer -- `initialized` starts `false`. Callers MUST
call `init!(sink, track)` themselves before attaching the sink to an
`AsyncSim` (e.g. via `RendererEntry(sink, ...)`) and calling `start!`. If
`init!` is skipped, the first `render!` call throws an `ErrorException`,
which -- once attached to an `AsyncSim` -- propagates and tears down the
whole sim (physics tier included). See `test/telemetry/sink_test.jl` for an
end-to-end example of the required `init!`-then-attach sequence.
"""
mutable struct TelemetrySink <: AbstractRenderer
    path::String
    writer::Union{Nothing,Arrow.Writer}
    tick_index::Int
    episode_index::Int
    initialized::Bool
end

TelemetrySink(path::String) = TelemetrySink(path, nothing, 0, 0, false)

function init!(sink::TelemetrySink, track::Track)
    sink.writer = open(Arrow.Writer, sink.path)
    sink.tick_index = 0
    sink.episode_index = 0
    sink.initialized = true
    return sink
end

function reset!(sink::TelemetrySink, track::Track)
    sink.initialized || error("TelemetrySink: reset! called before init!")
    sink.episode_index += 1
    sink.tick_index = 0
    return sink
end

function render!(sink::TelemetrySink, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool})
    sink.initialized || error("TelemetrySink: render! called before init!/reset!")
    n = length(states)
    batch = (
        episode_index=fill(sink.episode_index, n),
        tick_index=fill(sink.tick_index, n),
        agent_index=collect(1:n),
        state=[Vector(s) for s in states],
        scan=scans,
        collided=collisions,
    )
    Arrow.write(sink.writer, batch)
    sink.tick_index += 1
    return nothing
end

function close!(sink::TelemetrySink)
    sink.writer === nothing && return nothing
    close(sink.writer)
    sink.writer = nothing
    sink.initialized = false
    return nothing
end
