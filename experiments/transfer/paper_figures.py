"""Figures for the paper, from the CSV files of results/full and results/ablation.

    python3 experiments/transfer/paper_figures.py OUTDIR

Writes OUTDIR/grid_ftg_example.pdf, OUTDIR/transfer_rows.tex and, if results/ablation has
its CSV files, OUTDIR/ablation_rows.tex.
"""

import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Colors (validated with the dataviz validator: categorical slots 1-3 pass all
# hard checks on a light surface; aqua needs labels, which the charts have).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CRASH = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Sequential blue, steps 250 -> 700: the fastest settings are the darkest.
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"][::-1]
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],  # TrueType, metric-compatible with Times
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,  # embedded TrueType fonts, as IEEE asks
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": SURFACE,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def scores(rows, mapname, ctrl, cond):
    """Score of each grid setting: the mean race time if all seeds finish, else None."""
    runs = defaultdict(list)
    for r in rows:
        if r["map"] == mapname and r["controller"] == ctrl and r["condition"] == cond:
            runs[(float(r["p1"]), float(r["p2"]))].append(r)
    out = {}
    for k, rs in runs.items():
        out[k] = sum(float(r["race_time"]) for r in rs) / len(rs) if all(r["finished"] == "true" for r in rs) else None
    return out


def grid_figure(outdir):
    rows = read(os.path.join(HERE, "results", "full", "grid.csv"))
    conds = [("strict", "strict (Python car)"), ("strict-nuc5", "strict-nuc5 (NUC5, Python model)"),
             ("realistic", "realistic (NUC5)")]
    S = {c: scores(rows, "example", "ftg", c) for c, _ in conds}
    xs = sorted({k[0] for k in S["strict"]})
    ys = sorted({k[1] for k in S["strict"]})
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    vmin, vmax = 1.0, 2.0
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05), sharey=True, constrained_layout=True)
    for ax, (c, title) in zip(axes, conds):
        sc = S[c]
        best_k = min((k for k, v in sc.items() if v is not None), key=lambda k: sc[k])
        best = sc[best_k]
        for (x, y), v in sc.items():
            if v is None:
                ax.add_patch(plt.Rectangle((x - dx / 2, y - dy / 2), dx, dy, facecolor=CRASH, edgecolor=SURFACE,
                                           linewidth=0.6, hatch="/////"))
            else:
                t = min(max(v / best, vmin), vmax)
                ax.add_patch(plt.Rectangle((x - dx / 2, y - dy / 2), dx, dy,
                                           facecolor=SEQ((t - vmin) / (vmax - vmin)), edgecolor=SURFACE,
                                           linewidth=0.6))
        ax.plot(*best_k, marker="*", markersize=9, color="white", markeredgecolor=INK, markeredgewidth=0.7,
                linestyle="none")
        ax.set_xlim(xs[0] - dx / 2, xs[-1] + dx / 2)
        ax.set_ylim(ys[0] - dy / 2, ys[-1] + dy / 2)
        ax.set_title(title, color=INK)
        ax.set_xlabel("top speed vmax (m/s)")
        ax.tick_params(length=2)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    axes[0].set_ylabel("steering gain")
    sm = matplotlib.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.01, aspect=25)
    cb.set_label("race time / best race time")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([1.0, 1.25, 1.5, 1.75, 2.0], labels=["1.0", "1.25", "1.5", "1.75", "≥2.0"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CRASH, edgecolor=INK_2, hatch="/////", linewidth=0.4),
               plt.Line2D([], [], marker="*", markersize=8, color="white", markeredgecolor=INK, linestyle="none")]
    fig.legend(handles, ["crash or timeout (any seed)", "best setting"], loc="outside lower center", ncol=2,
               frameon=False)
    path = os.path.join(outdir, "grid_ftg_example.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


ABLATION = [("strict-nuc5", "strict-nuc5"), ("realistic", "realistic"),
            ("realistic-linear-tires", "linear tires"), ("realistic-ideal-steering", "ideal steering"),
            ("realistic-ideal-speed", "ideal speed"), ("realistic-ideal-lidar", "ideal LiDAR"),
            ("realistic-all-ideal", "all ideal")]
CTRLS = [("pp", "pure pursuit"), ("ftg", "follow the gap"), ("mpc", "MPC (kinematic)")]


def ablation_table(outdir):
    """LaTeX rows: for each condition of the ablation, the races finished by the best strict-nuc5
    setting (all maps and seeds), and the mean rank agreement with strict-nuc5 over the maps where
    it is defined (a dagger marks a mean over fewer than three maps)."""
    path_t = os.path.join(HERE, "results", "ablation", "transfer.csv")
    if not os.path.isfile(path_t):
        return None
    T = read(path_t)
    R = read(os.path.join(HERE, "results", "ablation", "rank_agreement.csv"))
    lines = []
    for cond, label in ABLATION[1:]:
        cells = [label]
        for c, _ in CTRLS:
            rs = [r for r in T if r["controller"] == c and r["tuned_on"] == "strict-nuc5" and r["raced_on"] == cond]
            cells.append(f"{sum(r['finished'] == 'true' for r in rs)}/{len(rs)}")
            rho = [float(r["spearman"]) for r in R if r["controller"] == c and r["a"] == "strict-nuc5" and r["b"] == cond]
            rho = [v for v in rho if not math.isnan(v)]
            cells.append("--" if not rho else f"${sum(rho) / len(rho):+.2f}$" + ("$^\\dagger$" if len(rho) < 3 else ""))
        lines.append(" & ".join(cells) + r" \\")
    path = os.path.join(outdir, "ablation_rows.tex")
    with open(path, "w") as f:
        f.write("% generated by experiments/transfer/paper_figures.py from results/ablation; do not edit\n")
        f.write("\n".join(lines) + "\n\\bottomrule\n")
    return path


MAPS = [("example", "Ex."), ("Spa", "Spa"), ("Monza", "Monza")]
CTRLS_ALL = [("pp", "Pure pursuit"), ("ftg", "Follow the gap"), ("mpc", "MPC (kinematic)"),
             ("mpc-lag", "MPC (actuator model)"), ("rl", "ARS policy")]


def transfer_table(outdir):
    """LaTeX rows: finished races of the best setting of one condition on another, and rank agreement."""
    T = read(os.path.join(HERE, "results", "full", "transfer.csv"))
    R = read(os.path.join(HERE, "results", "full", "rank_agreement.csv"))

    def fin(m, c, a, b):
        rs = [r for r in T if r["map"] == m and r["controller"] == c and r["tuned_on"] == a and r["raced_on"] == b]
        return "--" if not rs else f"{sum(r['finished'] == 'true' for r in rs)}/{len(rs)}"

    def rho(m, c, a, b):
        for r in R:
            if r["map"] == m and r["controller"] == c and r["a"] == a and r["b"] == b:
                v = float(r["spearman"])
                return "--" if math.isnan(v) else f"${v:+.2f}$"
        return "--"

    lines = []
    for c, label in CTRLS_ALL:
        cells = [label]
        for pair in (("strict", "realistic"), ("strict-nuc5", "realistic"), ("realistic", "strict")):
            cells += [fin(m, c, *pair) for m, _ in MAPS]
        cells += [rho(m, c, "strict", "realistic") for m, _ in MAPS]
        lines.append(" & ".join(cells) + r" \\")
    path = os.path.join(outdir, "transfer_rows.tex")
    with open(path, "w") as f:
        f.write("% generated by experiments/transfer/paper_figures.py from results/full; do not edit\n")
        f.write("\n".join(lines) + "\n\\bottomrule\n")
    return path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for p in (grid_figure(out), ablation_table(out), transfer_table(out)):
        if p:
            print("wrote", p)
