"""Freeze gold-standard outputs of the Python f1tenth_gym for the Julia tests.

Run from anywhere (needs numpy, scipy, numba, pillow, pyyaml):

    python test/reference/generate.py [path/to/f1tenth_gym]

The f1tenth_gym checkout defaults to ../f1tenth_gym next to this repository.
`gym` and `pyglet` are replaced by the stubs in ./stubs, since the reference
code only needs them to import.

Every dataset is written as raw little-endian float64 in C order to
data/<name>.bin, with its shape recorded in data/manifest.txt.
"""

import importlib.util
import os
import sys
import tempfile

sys.dont_write_bytecode = True  # leave no __pycache__ in the f1tenth_gym checkout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GYM_ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "..", "f1tenth_gym"))
OUT = os.path.join(HERE, "data")
# keep numba's on-disk cache out of the f1tenth_gym checkout
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "f1tenth_reference_numba"))

sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, os.path.join(GYM_ROOT, "gym"))

from f110_gym.envs import laser_models
from f110_gym.envs.base_classes import Integrator, RaceCar
from f110_gym.envs.collision_models import collision, collision_multiple, get_vertices
from f110_gym.envs.dynamic_models import (
    accl_constraints,
    pid,
    steering_constraint,
    vehicle_dynamics_ks,
    vehicle_dynamics_st,
)
from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, get_dt, ray_cast

spec = importlib.util.spec_from_file_location(
    "waypoint_follow", os.path.join(GYM_ROOT, "examples", "waypoint_follow.py")
)
waypoint_follow = importlib.util.module_from_spec(spec)
sys.modules["waypoint_follow"] = waypoint_follow  # numba's cache looks the module up by name
spec.loader.exec_module(waypoint_follow)

MAPS_DIR = os.path.join(GYM_ROOT, "gym", "f110_gym", "envs", "maps")
EXAMPLES = os.path.join(GYM_ROOT, "examples")
MAPS = {
    "berlin": (os.path.join(MAPS_DIR, "berlin.yaml"), ".png"),
    "levine": (os.path.join(MAPS_DIR, "levine.yaml"), ".pgm"),
    "skirk": (os.path.join(MAPS_DIR, "skirk.yaml"), ".png"),
    "stata_basement": (os.path.join(MAPS_DIR, "stata_basement.yaml"), ".png"),
    "vegas": (os.path.join(MAPS_DIR, "vegas.yaml"), ".png"),
    "example_map": (os.path.join(EXAMPLES, "example_map.yaml"), ".png"),
}
PARAMS = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I": 0.04712,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 20.0,
    "width": 0.31,
    "length": 0.58,
}
PKEYS = [
    "mu",
    "C_Sf",
    "C_Sr",
    "lf",
    "lr",
    "h",
    "m",
    "I",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_switch",
    "a_max",
    "v_min",
    "v_max",
]

manifest = []


def save(name, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    arr.astype("<f8").tofile(os.path.join(OUT, name + ".bin"))
    manifest.append(name + " " + " ".join(str(d) for d in arr.shape))


def pargs(p):
    return [p[k] for k in PKEYS]


def gen_dynamics(rng):
    n = 3000
    x = np.empty((n, 7))
    x[:, 0:2] = rng.uniform(-10, 10, (n, 2))
    x[:, 2] = rng.uniform(-0.5, 0.5, n)  # steering, sometimes past the limits
    x[:, 3] = rng.uniform(-6, 22, n)  # velocity, sometimes past the limits
    x[: n // 4, 3] = rng.uniform(-0.6, 0.6, n // 4)  # low-speed (kinematic) branch
    x[:, 4] = rng.uniform(-7, 7, n)
    x[:, 5] = rng.uniform(-3, 3, n)
    x[:, 6] = rng.uniform(-0.5, 0.5, n)
    u = np.column_stack([rng.uniform(-5, 5, n), rng.uniform(-12, 12, n)])
    st = np.array([vehicle_dynamics_st(x[i], u[i], *pargs(PARAMS)) for i in range(n)])
    ks = np.array([vehicle_dynamics_ks(x[i, 0:5], u[i], *pargs(PARAMS)) for i in range(n)])
    save("dyn_x", x)
    save("dyn_u", u)
    save("dyn_st", st)
    save("dyn_ks", ks)

    pin = np.column_stack(
        [rng.uniform(-6, 22, n), rng.uniform(-0.5, 0.5, n), rng.uniform(-6, 22, n), rng.uniform(-0.5, 0.5, n)]
    )
    pin[:50, 1] = pin[:50, 3] + rng.uniform(-1e-4, 1e-4, 50)  # steering dead band
    pin[50:100, 2] = 0.0  # standing still
    pout = np.array(
        [
            pid(
                pin[i, 0],
                pin[i, 1],
                pin[i, 2],
                pin[i, 3],
                PARAMS["sv_max"],
                PARAMS["a_max"],
                PARAMS["v_max"],
                PARAMS["v_min"],
            )
            for i in range(n)
        ]
    )
    save("pid_in", pin)
    save("pid_out", pout)

    cin = np.column_stack(
        [rng.uniform(-0.6, 0.6, n), rng.uniform(-5, 5, n), rng.uniform(-7, 22, n), rng.uniform(-12, 12, n)]
    )
    cout = np.array(
        [
            [
                steering_constraint(
                    cin[i, 0], cin[i, 1], PARAMS["s_min"], PARAMS["s_max"], PARAMS["sv_min"], PARAMS["sv_max"]
                ),
                accl_constraints(
                    cin[i, 2], cin[i, 3], PARAMS["v_switch"], PARAMS["a_max"], PARAMS["v_min"], PARAMS["v_max"]
                ),
            ]
            for i in range(n)
        ]
    )
    save("constraint_in", cin)
    save("constraint_out", cout)


def gen_collision(rng):
    n = 2000
    poses = np.column_stack([rng.uniform(-5, 5, (n, 2)), rng.uniform(-7, 7, n)])
    verts = np.array([get_vertices(poses[i], 0.58, 0.31) for i in range(n)])
    save("vert_poses", poses)
    save("vert_out", verts)

    # pairs of cars whose centers are 0..1 m apart: a mix of hits and misses
    a = np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(-4, 4, n)])
    off = rng.normal(size=(n, 2))
    off = off / np.linalg.norm(off, axis=1, keepdims=True) * rng.uniform(0, 1.0, (n, 1))
    b = np.column_stack([a[:, 0:2] + off, rng.uniform(-4, 4, n)])
    hits = np.array([collision(get_vertices(a[i], 0.58, 0.31), get_vertices(b[i], 0.58, 0.31)) for i in range(n)])
    save("gjk_a", a)
    save("gjk_b", b)
    save("gjk_hit", hits.astype(np.float64))

    m = 300
    groups = np.column_stack([rng.uniform(-1.2, 1.2, (m * 5, 2)), rng.uniform(-4, 4, m * 5)]).reshape(m, 5, 3)
    cols, idxs = [], []
    for g in groups:
        c, ix = collision_multiple(np.array([get_vertices(p, 0.58, 0.31) for p in g]))
        cols.append(c)
        idxs.append(ix)
    save("multi_poses", groups)
    save("multi_collisions", cols)
    save("multi_idx", idxs)


def gen_maps(rng):
    for name, (yaml_path, ext) in MAPS.items():
        sim = ScanSimulator2D(1080, 4.7)
        sim.set_map(yaml_path, ext)
        dt = sim.dt
        h, w = dt.shape
        k = 4000
        idx = np.column_stack([rng.integers(0, h, k), rng.integers(0, w, k)])
        save(f"map_{name}_dt_idx", idx)
        save(f"map_{name}_dt_val", dt[idx[:, 0], idx[:, 1]])
        save(f"map_{name}_dt_stats", [h, w, dt.sum(), dt.max()])

        # poses in free space, plus two outside the map (exercises dt[-1, -1])
        free = np.argwhere(dt > 0.4)
        pick = free[rng.integers(0, len(free), 24)]
        poses = np.column_stack(
            [
                pick[:, 1] * sim.map_resolution + sim.orig_x + 0.01,
                pick[:, 0] * sim.map_resolution + sim.orig_y + 0.01,
                rng.uniform(-7, 7, 24),
            ]
        )
        poses = np.vstack(
            [
                poses,
                [sim.orig_x - 3.0, sim.orig_y - 2.0, 0.3],
                [sim.orig_x + w * sim.map_resolution + 1, sim.orig_y, 2.0],
            ]
        )
        scans = np.array([sim.scan(p, None) for p in poses])
        save(f"map_{name}_scan_poses", poses)
        save(f"map_{name}_scans", scans)

    # a map whose origin is rotated (every bundled map has zero yaw)
    import shutil

    rot_dir = tempfile.mkdtemp()
    shutil.copy(os.path.join(MAPS_DIR, "berlin.png"), os.path.join(rot_dir, "berlin_rot.png"))
    with open(os.path.join(rot_dir, "berlin_rot.yaml"), "w") as f:
        f.write(
            "image: berlin_rot.png\nresolution: 0.050000\norigin: [-11.606540, -26.520793, 0.300000]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
        )
    sim = ScanSimulator2D(1080, 4.7)
    sim.set_map(os.path.join(rot_dir, "berlin_rot.yaml"), ".png")
    free = np.argwhere(sim.dt > 0.4)
    pick = free[rng.integers(0, len(free), 24)]
    lx, ly = pick[:, 1] * sim.map_resolution + 0.01, pick[:, 0] * sim.map_resolution + 0.01
    poses = np.column_stack(
        [
            sim.orig_x + lx * sim.orig_c - ly * sim.orig_s,
            sim.orig_y + lx * sim.orig_s + ly * sim.orig_c,
            rng.uniform(-7, 7, 24),
        ]
    )
    save("map_berlin_rot_scan_poses", poses)
    save("map_berlin_rot_scans", np.array([sim.scan(p, None) for p in poses]))
    shutil.rmtree(rot_dir)

    for i, (h, w) in enumerate([(37, 53), (64, 41), (20, 90)]):
        bmp = np.where(rng.uniform(size=(h, w)) < 0.15, 0.0, 255.0)
        save(f"edt_small_{i}_in", bmp)
        save(f"edt_small_{i}_out", get_dt(bmp, 0.05))


def gen_lidar_geometry(rng):
    RaceCar(PARAMS, 12345, num_beams=1080, fov=4.7)  # fills in the RaceCar class attributes
    save("side_distances", RaceCar.side_distances)
    save("scan_angles", RaceCar.scan_angles)

    n = 120
    poses = np.column_stack([rng.uniform(-2, 2, (n, 2)), rng.uniform(-7, 7, n)])
    opp = np.column_stack([poses[:, 0:2] + rng.uniform(-3, 3, (n, 2)), rng.uniform(-7, 7, n)])
    base = rng.uniform(0.5, 10.0, (n, 1080))
    out = np.array(
        [ray_cast(poses[i], base[i].copy(), RaceCar.scan_angles, get_vertices(opp[i], 0.58, 0.31)) for i in range(n)]
    )
    save("raycast_poses", poses)
    save("raycast_opp", opp)
    save("raycast_base", base)
    save("raycast_out", out)

    vel = rng.uniform(-6, 12, n)
    vel[:5] = 0.0
    scans = rng.uniform(0.0, 3.0, (n, 1080)) + rng.uniform(0.0, 1.0, (n, 1))
    ttc = np.array(
        [
            check_ttc_jit(scans[i], vel[i], RaceCar.scan_angles, RaceCar.cosines, RaceCar.side_distances, 0.005)
            for i in range(n)
        ]
    )
    save("ttc_scans", scans)
    save("ttc_vel", vel)
    save("ttc_out", ttc.astype(np.float64))


def disable_scan_noise():
    orig = laser_models.ScanSimulator2D.scan

    def scan(self, pose, rng, std_dev=0.01):
        return orig(self, pose, None)

    laser_models.ScanSimulator2D.scan = scan


def record(name, env, poses, policy, max_steps, scan_every, events=None):
    obs, reward, done, info = env.reset(poses)
    rows = []

    def snapshot(obs, reward, done, info, action):
        n = env.num_agents
        rows.append(
            np.concatenate(
                [
                    action.reshape(-1),
                    obs["poses_x"],
                    obs["poses_y"],
                    obs["poses_theta"],
                    obs["linear_vels_x"],
                    obs["linear_vels_y"],
                    obs["ang_vels_z"],
                    obs["collisions"],
                    obs["lap_times"],
                    obs["lap_counts"],
                    np.asarray(info["checkpoint_done"], dtype=np.float64),
                    [reward, float(done)],
                ]
            )
        )
        assert rows[-1].shape[0] == 2 * n + 10 * n + 2

    scans = []
    snapshot(obs, reward, done, info, np.zeros((env.num_agents, 2)))
    scans.append(np.array(obs["scans"]))
    step = 0
    while not done and step < max_steps:
        if events and step in events:
            events[step](env)
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        snapshot(obs, reward, done, info, action)
        if step % scan_every == 0 or done:
            scans.append(np.array(obs["scans"]))
    save(f"roll_{name}_rows", rows)
    save(f"roll_{name}_scans", scans)
    print(f"  {name}: {step} steps, done={done}")


def gen_rollouts():
    disable_scan_noise()
    from argparse import Namespace

    import yaml

    with open(os.path.join(EXAMPLES, "config_example_map.yaml")) as f:
        conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    conf.wpt_path = os.path.join(EXAMPLES, "example_waypoints.csv")
    work = {"tlad": 0.82461887897713965, "vgain": 1.375}
    wb = 0.17145 + 0.15875
    map_noext = os.path.join(EXAMPLES, "example_map")
    start = np.array([[conf.sx, conf.sy, conf.stheta]])

    # 1: the waypoint_follow example, one agent, two full laps
    planner = waypoint_follow.PurePursuitPlanner(conf, wb)
    env = F110Env(map=map_noext, map_ext=".png", num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    def pp1(obs):
        speed, steer = planner.plan(
            obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], work["tlad"], work["vgain"]
        )
        return np.array([[steer, speed]])

    record("pp_single", env, start, pp1, 12000, 100)

    # 2: two agents, a faster ego catches a slower opponent and rams it
    env = F110Env(map=map_noext, map_ext=".png", num_agents=2, timestep=0.01, integrator=Integrator.RK4)
    wpts = planner.waypoints
    opp_start = [wpts[200, conf.wpt_xind], wpts[200, conf.wpt_yind], wpts[200, conf.wpt_thind] + np.pi / 2]

    def pp2(obs):
        out = np.zeros((2, 2))
        for i, gain in enumerate([1.5, 0.5]):
            speed, steer = planner.plan(obs["poses_x"][i], obs["poses_y"][i], obs["poses_theta"][i], work["tlad"], gain)
            out[i] = [steer, speed]
        return out

    record("pp_two", env, np.array([start[0], opp_start]), pp2, 12000, 50)

    # 3: Euler integrator, LiDAR offset, 0.02 s steps, steering into a wall
    env = F110Env(
        map=map_noext, map_ext=".png", num_agents=1, timestep=0.02, integrator=Integrator.Euler, lidar_dist=0.25
    )
    record("wall_euler", env, start, lambda obs: np.array([[0.35, 6.0]]), 3000, 10)

    # 4: per-agent parameter update partway through, then a global one
    env = F110Env(map=map_noext, map_ext=".png", num_agents=2, timestep=0.01, integrator=Integrator.RK4, seed=7)
    slippery = dict(PARAMS, mu=0.6, m=4.5, length=0.8, width=0.5)
    heavy = dict(PARAMS, m=5.0, a_max=6.0)
    events = {100: lambda e: e.update_params(slippery, index=1), 250: lambda e: e.update_params(heavy)}
    record("params_update", env, np.array([start[0], opp_start]), pp2, 600, 50, events)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261103)
    print("dynamics")
    gen_dynamics(rng)
    print("collision")
    gen_collision(rng)
    print("maps")
    gen_maps(rng)
    print("lidar geometry")
    gen_lidar_geometry(rng)
    print("rollouts")
    gen_rollouts()
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
