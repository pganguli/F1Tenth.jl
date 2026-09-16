module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")

function __init__()
    __mtk_models_init__()
end

end
