abstract type AbstractRenderer end

function init! end
function render! end
function close! end
function reset! end

# Implemented by src/render/glmakie.jl and src/render/cairomakie.jl
# respectively (Tasks 2/3) -- these have no methods until then.
function GLMakieRenderer end
function CairoMakieRenderer end

struct NullRenderer <: AbstractRenderer end

init!(renderer::NullRenderer, track::Track) = renderer
render!(renderer::NullRenderer, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool}) = nothing
close!(renderer::NullRenderer) = nothing
reset!(renderer::NullRenderer, track::Track) = renderer
