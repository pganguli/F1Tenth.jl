using Random: Xoshiro, randn

@testset "collision_models (Python unit tests)" begin
    rng = Xoshiro(1234)
    vertices1 = [4 11.0; 5 5; 9 9; 10 10]
    len, wid = 0.32, 0.22

    @test length(get_vertices((2.3, 6.7, 0.8), len, wid)) == 4
    perturb() = vertices1 + randn(rng, size(vertices1)) / 100
    # perturbed copies of one body always overlap
    @test all(collision(perturb(), perturb()) for _ in 1:1000)

    a, b, c, d, e, f = (perturb() for _ in 1:6)
    g = vertices1 .+ 10.0
    collisions, collision_idx = collision_multiple([a, b, c, d, e, f, g])
    @test collisions == [1.0, 1, 1, 1, 1, 1, 0]
    @test collision_idx == [6, 6, 6, 6, 6, 5, -1]   # Python: [5, 5, 5, 5, 5, 4, -1], 0-based

    # Python asserts > 500 checks/s for both
    t = @elapsed for _ in 1:10_000
        get_vertices((2.3, 6.7, 0.8), len, wid)
    end
    @test 10_000 / t > 500
    pairs = [(perturb(), perturb()) for _ in 1:1000]
    t = @elapsed foreach(p -> collision(p...), pairs)
    @test 1000 / t > 500
end

@testset "collision_models (match Python outputs)" begin
    poses, verts = ref("vert_poses"), ref("vert_out")
    for i in axes(poses, 1)
        v = get_vertices(poses[i, :], 0.58, 0.31)
        @test [v[k][j] for k in 1:4, j in 1:2] ≈ verts[i, :, :] rtol = 1e-14 atol = 1e-14
    end

    a, b, hit = ref("gjk_a"), ref("gjk_b"), ref("gjk_hit")
    results = [collision(get_vertices(a[i, :], 0.58, 0.31), get_vertices(b[i, :], 0.58, 0.31)) for i in axes(a, 1)]
    @test results == (hit .== 1)
    @test 0.2 < count(results) / length(results) < 0.8   # the data covers hits and misses

    groups, cols, idxs = ref("multi_poses"), ref("multi_collisions"), ref("multi_idx")
    for k in axes(groups, 1)
        c, ix = collision_multiple([get_vertices(groups[k, i, :], 0.58, 0.31) for i in 1:5])
        @test c == cols[k, :]
        @test ix == [v < 0 ? -1 : Int(v) + 1 for v in idxs[k, :]]
    end
end
