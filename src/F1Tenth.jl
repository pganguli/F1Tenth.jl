module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")

function __init__()
    __mtk_models_init__()
end

end
