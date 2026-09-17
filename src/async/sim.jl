using Random: AbstractRNG, default_rng
using Base.Threads: Atomic, @spawn, nthreads

const RenderFrame = Tuple{Vector{State},Vector{Vector{Float64}},Vector{Bool}}

abstract type RenderBackpressure end
struct DropLatest <: RenderBackpressure end
struct Blocking <: RenderBackpressure end

struct RendererEntry
    renderer::AbstractRenderer
    policy::RenderBackpressure
    channel::Union{LatestChannel{RenderFrame},Channel{RenderFrame}}
end

RendererEntry(renderer::AbstractRenderer, ::DropLatest) =
    RendererEntry(renderer, DropLatest(), LatestChannel{RenderFrame}())
RendererEntry(renderer::AbstractRenderer, ::Blocking; buffer::Int=8) =
    RendererEntry(renderer, Blocking(), Channel{RenderFrame}(buffer))

mutable struct AsyncSim
    bvh::TrackBVH
    params::VehicleParams
    car::CarFootprint
    dt::Float64
    cfg::LiDARConfig
    tick_hz::Union{Float64,Nothing}
    renderers::Vector{RendererEntry}
    input::Vector{LatestChannel{Action}}
    last_actions::Vector{Action}
    states::Vector{State}
    output::LatestChannel{RenderFrame}
    running::Atomic{Bool}
    physics_task::Union{Task,Nothing}
    renderer_tasks::Vector{Task}
    error::Ref{Union{Nothing,Exception}}
    error_lock::ReentrantLock
    rng::AbstractRNG
end

function AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64,
    cfg::LiDARConfig, initial_states::Vector{State}, initial_actions::Vector{Action};
    tick_hz::Union{Float64,Nothing}=nothing, renderers::Vector{RendererEntry}=RendererEntry[],
    rng::AbstractRNG=default_rng())
    n = length(initial_states)
    length(initial_actions) == n || throw(ArgumentError(
        "initial_actions must have one entry per agent (got $(length(initial_actions)) for $n agents)"))
    input = [LatestChannel{Action}() for _ in 1:n]
    return AsyncSim(bvh, params, car, dt, cfg, tick_hz, renderers, input, copy(initial_actions),
        copy(initial_states), LatestChannel{RenderFrame}(), Atomic{Bool}(false),
        nothing, Task[], Ref{Union{Nothing,Exception}}(nothing), ReentrantLock(), rng)
end

_required_threads(sim::AsyncSim) = 2 + length(sim.renderers)

function _record_error!(sim::AsyncSim, e::Exception)
    lock(sim.error_lock) do
        sim.error[] === nothing && (sim.error[] = e)
    end
    return nothing
end

function _close_all_channels!(sim::AsyncSim)
    for lc in sim.input
        close(lc.ch)
    end
    close(sim.output.ch)
    for entry in sim.renderers
        close(entry.channel isa LatestChannel ? entry.channel.ch : entry.channel)
    end
    return nothing
end

function _physics_tick!(sim::AsyncSim)
    n = length(sim.states)
    actions = Vector{Action}(undef, n)
    for i in 1:n
        a = take_latest!(sim.input[i])
        a === missing || (sim.last_actions[i] = a)
        actions[i] = sim.last_actions[i]
    end
    new_states, scans, collisions = step_multi_agent!(sim.states, actions, sim.bvh,
        sim.params, sim.car, sim.dt, sim.cfg; rng=sim.rng)
    sim.states = new_states
    frame = (new_states, scans, collisions)
    put_latest!(sim.output, frame)
    for entry in sim.renderers
        if entry.channel isa LatestChannel
            put_latest!(entry.channel, frame)
        else
            put!(entry.channel, frame)
        end
    end
    return nothing
end

function _physics_loop(sim::AsyncSim)
    deadline = time() + (sim.tick_hz === nothing ? 0.0 : 1 / sim.tick_hz)
    try
        while sim.running[]
            _physics_tick!(sim)
            if sim.tick_hz !== nothing
                sleep(max(0.0, deadline - time()))
                deadline += 1 / sim.tick_hz
            end
        end
    catch e
        if !(e isa InvalidStateException)
            _record_error!(sim, e)
            sim.running[] = false
            _close_all_channels!(sim)
            rethrow()
        end
    end
    return nothing
end

function _renderer_loop(sim::AsyncSim, entry::RendererEntry)
    try
        while true
            frame = entry.channel isa LatestChannel ? wait_latest!(entry.channel) : take!(entry.channel)
            render!(entry.renderer, frame...)
        end
    catch e
        if !(e isa InvalidStateException)
            _record_error!(sim, e)
            sim.running[] = false
            _close_all_channels!(sim)
            rethrow()
        end
    end
    return nothing
end

function start!(sim::AsyncSim)
    sim.running[] && error("AsyncSim: already running")
    required = _required_threads(sim)
    nthreads() >= required || error(
        "AsyncSim: needs Threads.nthreads() >= $required (1 physics thread + " *
        "$(length(sim.renderers)) renderer thread(s)); got $(nthreads()). " *
        "Start Julia with `julia --threads=$required` or set JULIA_NUM_THREADS=$required.")
    sim.error[] = nothing
    sim.running[] = true
    empty!(sim.renderer_tasks)
    for entry in sim.renderers
        push!(sim.renderer_tasks, @spawn _renderer_loop(sim, entry))
    end
    sim.physics_task = @spawn _physics_loop(sim)
    return sim
end

function stop!(sim::AsyncSim)
    sim.running[] = false
    _close_all_channels!(sim)
    tasks = sim.physics_task === nothing ? sim.renderer_tasks : [sim.physics_task; sim.renderer_tasks]
    for t in tasks
        try
            fetch(t)
        catch
            # real cause, if any, is captured in sim.error and rethrown below
        end
    end
    sim.physics_task = nothing
    empty!(sim.renderer_tasks)
    if sim.error[] !== nothing
        err = sim.error[]
        sim.error[] = nothing
        throw(err)
    end
    return sim
end

function reset!(sim::AsyncSim, track::Track, initial_states::Vector{State})
    sim.running[] && error("AsyncSim: reset! requires the sim to be stopped first (call stop! first)")
    n = length(sim.last_actions)
    length(initial_states) == n || throw(ArgumentError(
        "reset! requires the same number of agents ($n); got $(length(initial_states))"))
    for entry in sim.renderers
        reset!(entry.renderer, track)
    end
    sim.states = copy(initial_states)
    fill!(sim.last_actions, zero(Action))
    take_latest!(sim.output)
    for lc in sim.input
        take_latest!(lc)
    end
    sim.error[] = nothing
    return sim
end
