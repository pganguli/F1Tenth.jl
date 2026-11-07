using Random: Xoshiro
using Statistics: std, mean

@testset "map loading and distance transform" begin
    for i in 0:2
        bmp = ref("edt_small_$(i)_in")
        raw = UInt8.(bmp)
        @test F1Tenth.distance_transform(raw, 0.05) ≈ ref("edt_small_$(i)_out") rtol = 1e-15
    end
    for (name, (path, ext)) in REF_MAPS
        m = F1Tenth.load_map(path, ext)
        h, w, dtsum, dtmax = ref("map_$(name)_dt_stats")
        @test size(m.dt) == (h, w)
        idx = ref("map_$(name)_dt_idx")
        vals = ref("map_$(name)_dt_val")
        @test [m.dt[Int(idx[k, 1])+1, Int(idx[k, 2])+1] for k in axes(idx, 1)] == vals
        @test maximum(m.dt) == dtmax
        @test sum(m.dt) ≈ dtsum rtol = 1e-12
    end
    # cached: the same map file gives the same object
    p, e = REF_MAPS["berlin"]
    @test F1Tenth.load_map(p, e) === F1Tenth.load_map(p, e)
end

# berlin with a rotated origin, built the same way as in generate.py
function rotated_berlin(dir)
    cp(joinpath(F1Tenth.MAPS_DIR, "berlin.png"), joinpath(dir, "berlin_rot.png"))
    write(joinpath(dir, "berlin_rot.yaml"), "image: berlin_rot.png\nresolution: 0.050000\n" *
        "origin: [-11.606540, -26.520793, 0.300000]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    return joinpath(dir, "berlin_rot.yaml")
end

@testset "scans (match Python bit for bit, noise free)" begin
    mktempdir() do dir
        maps = merge(REF_MAPS, Dict("berlin_rot" => (rotated_berlin(dir), ".png")))
        # both tracers, whichever one :auto picks on this machine
        for tracer in (:scalar, :simd)
            F1Tenth.TRACER[] = tracer
            @testset "$tracer" begin
                for (name, (path, ext)) in maps
                    sim = ScanSimulator2D(1080, 4.7)
                    set_map!(sim, path, ext)
                    poses, scans = ref("map_$(name)_scan_poses"), ref("map_$(name)_scans")
                    @test all(scan(sim, poses[k, :], nothing) == scans[k, :] for k in axes(poses, 1))
                end
            end
        end
        F1Tenth.TRACER[] = :auto
    end
    sim = ScanSimulator2D(1080, 4.7)
    @test_throws ArgumentError scan(sim, (0.0, 0.0, 0.0), nothing)
end

@testset "scans (Python unit tests)" begin
    sim = ScanSimulator2D(1080, 4.7)
    set_map!(sim, REF_MAPS["berlin"]...)
    # test_rng: reseeding reproduces the noise, successive scans differ
    rng = Xoshiro(12345)
    scan1 = scan(sim, (0.0, 0.0, 0.0), rng)
    scan2 = scan(sim, (0.0, 0.0, 0.0), rng)
    for _ in 1:100
        scan(sim, (0.0, 0.0, 0.0), rng)
    end
    scan4 = scan(sim, (0.0, 0.0, 0.0), rng)
    rng = Xoshiro(12345)
    scan5 = scan(sim, (0.0, 0.0, 0.0), rng)
    scan(sim, (0.0, 0.0, 0.0), rng)
    for _ in 1:100
        scan(sim, (0.0, 0.0, 0.0), rng)
    end
    scan6 = scan(sim, (0.0, 0.0, 0.0), rng)
    @test scan1 ≈ scan5
    @test !(scan1 ≈ scan2)
    @test scan4 ≈ scan6
    # noise is N(0, 0.01) per beam, as rng.normal(0, 0.01) in Python
    clean = scan(sim, (0.0, 0.0, 0.0), nothing)
    @test std(scan1 - clean) ≈ 0.01 rtol = 0.1
    @test abs(mean(scan1 - clean)) < 0.002
    # test_fps: Python asserts > 500 scans/s
    out = zeros(1080)
    t = @elapsed for i in 0:9999
        scan!(out, sim, (i / 10000, 0.0, 0.0), rng)
    end
    @test 10_000 / t > 500
end

@testset "LiDAR geometry, iTTC and car occlusion (match Python)" begin
    # Python's LiDAR layout (1080 beams over 4.7 rad); the iTTC check is fed
    # Python's car outline, since this package measures to the real body
    g = F1Tenth.LidarGeometry(1080, 4.7, get_increment(ScanSimulator2D(1080, 4.7)))
    @test g.scan_angles ≈ ref("scan_angles") rtol = 1e-14 atol = 1e-15
    py_sides = ref("side_distances")

    poses, opp, base, out = ref("raycast_poses"), ref("raycast_opp"), ref("raycast_base"), ref("raycast_out")
    for i in axes(poses, 1)
        s = ray_cast!(poses[i, :], base[i, :], g.scan_angles, get_vertices(opp[i, :], 0.58, 0.31))
        @test s ≈ out[i, :] rtol = 1e-12
    end
    @test any(ref("raycast_out") .!= ref("raycast_base"))   # the data has occlusions

    scans, vel, ttc = ref("ttc_scans"), ref("ttc_vel"), ref("ttc_out")
    cosines = cos.(g.scan_angles)
    results = [check_ttc(scans[i, :], vel[i], g.scan_angles, cosines, py_sides, 0.005) for i in axes(scans, 1)]
    @test results == (ttc .== 1)
    @test any(results) && !all(results)
end
