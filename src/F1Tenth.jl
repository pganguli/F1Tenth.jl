module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")

function __init__()
    __mtk_models_init__()
end

end
