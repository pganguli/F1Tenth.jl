abstract type AbstractRenderer end

function init! end
function render! end
function close! end
function reset! end

# Implemented by ext/F1TenthGLMakieExt.jl and ext/F1TenthCairoMakieExt.jl
# respectively -- these have no methods until the caller also loads
# GLMakie/CairoMakie. See Project.toml's [weakdeps]/[extensions].
function GLMakieRenderer end
function CairoMakieRenderer end

struct NullRenderer <: AbstractRenderer end

init!(renderer::NullRenderer, track::Track) = renderer
render!(renderer::NullRenderer, states::Vector{State},
    scans::Vector{Vector{Float64}}, collisions::Vector{Bool}) = nothing
close!(renderer::NullRenderer) = nothing
reset!(renderer::NullRenderer, track::Track) = renderer
