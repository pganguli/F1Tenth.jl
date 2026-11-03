# Opens real windows, so it needs a display (CI provides one with Xvfb).
@testset "rendering" begin
    if !(haskey(ENV, "DISPLAY") || haskey(ENV, "WAYLAND_DISPLAY"))
        @info "No display; skipping render window tests."
        return
    end
    @eval Main using GLMakie
    GLMakie = Main.GLMakie

    env = F110Env(map="berlin", num_agents=2)
    calls = Ref(0)
    add_render_callback!(env, function (e)
        calls[] += 1
        e.score_label.x = 123.0
    end)
    reset!(env, [0.0 0.0 0.0; 2.0 0.0 0.0])
    Base.invokelatest() do
        render!(env)
        for _ in 1:5
            step!(env, [0.1 2.0; 0.0 1.0])
            render!(env; mode=:human_fast)
        end
    end
    r = env.renderer
    # let the window manager finish resizing the window (handled on GLMakie's
    # render task) before checking the camera
    area() = GLMakie.widths(GLMakie.events(r.figure).window_area[])
    stable = 0
    for _ in 1:100   # up to 5 s; done once the size has held for 0.3 s
        last = area()
        sleep(0.05)
        stable = area() == last == (r.width, r.height) ? stable + 1 : 0
        stable >= 6 && break
    end
    @test r isa EnvRenderer
    @test calls[] == 6
    @test length(r.cars) == 2 && r.poses[:, 1] == env.obs.poses_x
    # car corners are 50 × get_vertices with the fixed renderer car size
    v = get_vertices((env.obs.poses_x[1], env.obs.poses_y[1], env.obs.poses_theta[1]), 0.58, 0.31)
    @test r.cars[1][][1][1] ≈ 50 * v[1][1] rtol = 1e-5
    @test r.score_label.x == 123.0
    @test r.score_label.text == "Lap Time: 0.00, Ego Lap Count: 0"   # never updated, as in Python
    # initial camera, as Python's on_resize at zoom 1.2 (1000 × 800 unless the
    # window manager resized the window)
    @test (r.left, r.right) == (-0.6r.width, 0.6r.width) && (r.bottom, r.top) == (-0.6r.height, 0.6r.height)

    Base.invokelatest() do
        ev = GLMakie.events(r.figure)
        # scroll up at the window center zooms in about it
        ev.mouseposition[] = (r.width / 2, r.height / 2)
        ev.scroll[] = (0.0, 1.0)
        @test r.zoom_level ≈ 1.44
        @test r.right - r.left ≈ 1.44 * r.width
        # drag pans by dx * zoom
        left = r.left
        ev.mousebutton[] = GLMakie.MouseButtonEvent(GLMakie.Mouse.left, GLMakie.Mouse.press)
        ev.mouseposition[] = (r.width / 2 + 10, r.height / 2)
        ev.mousebutton[] = GLMakie.MouseButtonEvent(GLMakie.Mouse.left, GLMakie.Mouse.release)
        @test r.left ≈ left - 10 * r.zoom_level
        # update_map! leaves the window as it is
        update_map!(env, joinpath(F1Tenth.MAPS_DIR, "skirk.yaml"), ".png")
        render!(env; mode=:human_fast)
    end

    # closing the window makes the next render! fail, as in Python
    close(r.screen)
    @test_throws ErrorException Base.invokelatest(render!, env)
end
