using F1Tenth
using Logging
using Statistics: std, mean
using Test

include(joinpath(@__DIR__, "reference", "reference.jl"))

quiet(f) = Logging.with_logger(f, Logging.NullLogger())   # hides the one-time RK4 warning

@testset "F1Tenth.jl" begin
    include("dynamic_models_test.jl")
    include("actuators_test.jl")
    include("collision_models_test.jl")
    include("laser_models_test.jl")
    include("f110_env_test.jl")
    include("examples_test.jl")
    include("rendering_test.jl")
    include("aqua_test.jl")
end
