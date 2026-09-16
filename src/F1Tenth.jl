module F1Tenth

include("dynamics/params.jl")
include("dynamics/mtk_models.jl")
include("dynamics/step.jl")
include("dynamics/solve.jl")
include("geometry/track.jl")
include("geometry/bvh.jl")
include("geometry/raycast.jl")
include("geometry/collision.jl")
include("geometry/step2.jl")
include("geometry/lazysets_export.jl")

export VehicleParams, State, Action, step_dynamics, simulate, KS_MODE, ST_MODE
export Track, Segment, build_track, load_track
export TrackBVH, build_bvh
export LiDARConfig, CarFootprint, car_vertices, cast_ray, scan
export collides, check_car_collisions, check_track_collision
export step_multi_agent!
export to_lazysets

function __init__()
    __mtk_models_init__()
end

end
