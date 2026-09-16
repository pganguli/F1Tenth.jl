using F1Tenth
using Test

include("testutils.jl")

@testset "NullRenderer: full lifecycle is a safe no-op" begin
    track = small_test_track()
    renderer = F1Tenth.NullRenderer()

    F1Tenth.init!(renderer, track)
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    scans = [[1.0, 2.0, 3.0]]
    collisions = [false]
    F1Tenth.render!(renderer, states, scans, collisions)
    F1Tenth.reset!(renderer, track)
    F1Tenth.render!(renderer, states, scans, collisions)
    F1Tenth.close!(renderer)

    @test true  # reaching here without throwing is the assertion
end

@testset "NullRenderer: render! before init! does not throw" begin
    renderer = F1Tenth.NullRenderer()
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    F1Tenth.render!(renderer, states, [[1.0]], [false])
    @test true
end

@testset "NullRenderer: render! allocates zero bytes" begin
    renderer = F1Tenth.NullRenderer()
    states = [F1Tenth.State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    scans = [[1.0]]
    collisions = [false]
    F1Tenth.render!(renderer, states, scans, collisions)  # compile once
    allocs = @allocated F1Tenth.render!(renderer, states, scans, collisions)
    @test allocs == 0
end
