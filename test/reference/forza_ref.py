# Regenerates data/forza_st_pacejka.txt from ForzaETH's f1tenth_gym fork
# (https://github.com/ForzaETH/f1tenth_gym, branch humble, MIT license):
#   python3 forza_ref.py <fork>/gym/f110_gym/envs/dynamic_models.py > data/forza_st_pacejka.txt
# Each row: x (7), u (2), mu, ForzaETH's derivative (7).
# Evaluate ForzaETH's vehicle_dynamics_st_pacejka (f1tenth_gym fork) on fixed
# states, without importing their module (numba stubbed out).
import re
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1]).read_text()
names = ["accl_constraints", "steering_constraint", "vehicle_dynamics_ks", "vehicle_dynamics_st_pacejka"]
ns = {"np": np, "njit": lambda **k: lambda f: f}
for n in names:
    m = re.search(r"@njit\(cache=True\)\ndef " + n + r"\(.*?(?=\n@njit|\ndef |\nclass |\Z)", src, re.DOTALL)
    # run only the four pure-math functions, with numba stubbed out
    exec(m.group(0), ns)  # noqa: S102
f = ns["vehicle_dynamics_st_pacejka"]
tires = (6.4196, 0.9826, 0.4044, 0.5073, 7.5624, 1.3006, 0.3237, 0.6478)
geo = (0.165, 0.155, 0.02, 3.5, 0.0627)
lim = (-0.4189, 0.4189, -3.2, 3.2, 7.319, 9.51, -5.0, 20.0)
rng = np.random.default_rng(20261106)
rows = []
for k in range(40):
    x = np.array(
        [
            rng.uniform(-5, 5),
            rng.uniform(-5, 5),
            rng.uniform(-0.4, 0.4),
            rng.uniform(0.6, 12),
            rng.uniform(0, 6.28),
            rng.uniform(-4, 4),
            rng.uniform(-0.3, 0.3),
        ]
    )
    u = np.array([rng.uniform(-3, 3), 0.0])
    mu = 1.0 if k % 2 == 0 else 0.7
    d = f(x, u, mu, *tires, *geo, *lim)
    rows.append(list(x) + list(u) + [mu] + list(d))
for r in rows:
    print(" ".join(repr(float(v)) for v in r))
