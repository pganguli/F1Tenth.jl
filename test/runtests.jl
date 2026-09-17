using F1Tenth
using Test

@testset "F1Tenth.jl" begin
    include("params_test.jl")
    include("mtk_models_test.jl")
    include("step_test.jl")
    include("solve_test.jl")
    include("geometry/track_test.jl")
    include("geometry/bvh_test.jl")
    include("geometry/raycast_test.jl")
    include("geometry/collision_test.jl")
    include("geometry/step2_test.jl")
    include("geometry/lazysets_export_test.jl")
    include("render/null_test.jl")
    include("render/glmakie_test.jl")
    include("render/cairomakie_test.jl")
    include("exports_test.jl")
end
