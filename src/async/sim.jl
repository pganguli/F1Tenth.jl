using Random: AbstractRNG, default_rng
using Base.Threads: Atomic, @spawn, nthreads

const RenderFrame = Tuple{Vector{State},Vector{Vector{Float64}},Vector{Bool}}

mutable struct AsyncSim
    bvh::TrackBVH
    params::VehicleParams
    car::CarFootprint
    dt::Float64
    cfg::LiDARConfig
    tick_hz::Union{Float64,Nothing}
    input::Vector{LatestChannel{Action}}
    last_actions::Vector{Action}
    states::Vector{State}
    output::LatestChannel{RenderFrame}
    running::Atomic{Bool}
    physics_task::Union{Task,Nothing}
    error::Ref{Union{Nothing,Exception}}
    error_lock::ReentrantLock
    rng::AbstractRNG
end

function AsyncSim(bvh::TrackBVH, params::VehicleParams, car::CarFootprint, dt::Float64,
    cfg::LiDARConfig, initial_states::Vector{State}, initial_actions::Vector{Action};
    tick_hz::Union{Float64,Nothing}=nothing, rng::AbstractRNG=default_rng())
    n = length(initial_states)
    length(initial_actions) == n || throw(ArgumentError(
        "initial_actions must have one entry per agent (got $(length(initial_actions)) for $n agents)"))
    input = [LatestChannel{Action}() for _ in 1:n]
    return AsyncSim(bvh, params, car, dt, cfg, tick_hz, input, copy(initial_actions),
        copy(initial_states), LatestChannel{RenderFrame}(), Atomic{Bool}(false),
        nothing, Ref{Union{Nothing,Exception}}(nothing), ReentrantLock(), rng)
end

_required_threads(sim::AsyncSim) = 2

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
    put_latest!(sim.output, (new_states, scans, collisions))
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

function start!(sim::AsyncSim)
    sim.running[] && error("AsyncSim: already running")
    required = _required_threads(sim)
    nthreads() >= required || error(
        "AsyncSim: needs Threads.nthreads() >= $required; got $(nthreads()). " *
        "Start Julia with `julia --threads=$required` or set JULIA_NUM_THREADS=$required.")
    sim.error[] = nothing
    sim.running[] = true
    sim.physics_task = @spawn _physics_loop(sim)
    return sim
end

function stop!(sim::AsyncSim)
    sim.running[] = false
    _close_all_channels!(sim)
    if sim.physics_task !== nothing
        try
            fetch(sim.physics_task)
        catch
            # the real cause, if any, is captured in sim.error and rethrown below
        end
    end
    sim.physics_task = nothing
    if sim.error[] !== nothing
        err = sim.error[]
        sim.error[] = nothing
        throw(err)
    end
    return sim
end
