using F1Tenth
using Test

@testset "F1Tenth: public API is exported" begin
    exported_names = [
        :VehicleParams, :State, :Action, :step_dynamics, :simulate, :KS_MODE, :ST_MODE,
        :Track, :Segment, :build_track, :load_track,
        :TrackBVH, :build_bvh,
        :LiDARConfig, :CarFootprint, :car_vertices, :cast_ray, :scan,
        :collides, :check_car_collisions, :check_track_collision,
        :step_multi_agent!,
        :to_lazysets,
        :AbstractRenderer, :NullRenderer, :GLMakieRenderer, :CairoMakieRenderer,
        :init!, :render!, :close!, :reset!,
        :LatestChannel, :put_latest!, :take_latest!, :wait_latest!,
        :AsyncSim, :start!, :stop!,
        :RenderBackpressure, :DropLatest, :Blocking, :RendererEntry,
        :build_hybrid_system, :reachability_analyze, :check_flowpipe_safety,
        :TelemetrySink,
    ]
    for name in exported_names
        @test name in names(F1Tenth)
    end
end
