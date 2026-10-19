"""Plot intensity histograms of the volumes of all femurs, for a quick visual check.

Writes to --out:
  histograms_<MOD>.png       one panel per femur (shared x-axis) for each modality
  f_NNN_histograms.png       one panel per modality for each femur
  histogram_stats.csv        min / percentiles / max / mean / std per femur and modality

Histograms are computed inside the bone mask (coloured line) and, with --whole, also
for the whole volume (grey line), from every --step'th voxel along each axis.

Example:
    python scripts/plot_histograms.py --root $DATA_ROOT
    python scripts/plot_histograms.py --root $DATA_ROOT --mods LR HR HR_hm SY_hm --whole --log
"""
import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, io  # noqa: E402

MODS = {"LR": ("LR", ""), "HR": ("HR", ""), "SY": ("SY", ""), "HR_hm": ("HR", "hm"), "SY_hm": ("SY", "hm")}
COLORS = {"LR": "#eb6834", "HR": "#2a78d6", "SY": "#1baf7a", "HR_hm": "#6250d6", "SY_hm": "#1baf7a"}
GREY = "#8a8984"
N_SAMPLES = 2_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--femurs", nargs="*", default=config.FEMURS)
    p.add_argument("--mods", nargs="*", default=["LR", "HR"], choices=list(MODS))
    p.add_argument("--step", type=int, default=4, help="Use every n'th voxel along each axis")
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--whole", action="store_true", help="Also plot the histogram of the whole volume")
    p.add_argument("--log", action="store_true", help="Logarithmic y-axis")
    p.add_argument("--out", default="results/histograms")
    return p.parse_args()


def sample(values, rng):
    return values if values.size <= N_SAMPLES else rng.choice(values, N_SAMPLES, replace=False)


def style(ax, log):
    if log:
        ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    s = slice(None, None, args.step)

    # Read (a sub-sample of) every volume once
    data, rows = {}, []
    for femur in args.femurs:
        name = io.femur_name(femur)
        if not os.path.exists(io.mask_path(args.root, femur)):
            print(f"{name}: no mask found, skipped ({io.mask_path(args.root, femur)})")
            continue
        mask = np.asarray(io.open_volume(io.mask_path(args.root, femur))[s, s, s]) > 0
        for mod in args.mods:
            path = io.get_path(args.root, femur, *MODS[mod])
            if not os.path.exists(path):
                print(f"{name} {mod}: not found, skipped ({path})")
                continue
            vol = io.open_volume(path)
            v = np.asarray(vol[s, s, s], dtype=np.float32)
            inside = v[mask]
            data[name, mod] = (sample(inside, rng), sample(v.ravel(), rng) if args.whole else None)
            q = np.percentile(inside, [0, 1, 50, 99, 100])
            rows.append({"femur": name, "modality": mod, "dtype": str(vol.dtype), "shape": "x".join(map(str, vol.shape)),
                         "min": q[0], "p1": q[1], "median": q[2], "p99": q[3], "max": q[4],
                         "mean": float(inside.mean()), "std": float(inside.std())})
            print(f"{name} {mod:6s} inside mask: min {q[0]:.1f}, p1 {q[1]:.1f}, median {q[2]:.1f}, "
                  f"p99 {q[3]:.1f}, max {q[4]:.1f}")

    with open(os.path.join(args.out, "histogram_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})

    # Shared x-range per modality (0.1-99.9 percentiles over all femurs, inside and outside the mask)
    ranges, hists = {}, {}
    for mod in args.mods:
        vals = [v for (n, m), (ins, whole) in data.items() if m == mod for v in (ins, whole) if v is not None]
        if not vals:
            continue
        ranges[mod] = (min(np.percentile(v, 0.1) for v in vals), max(np.percentile(v, 99.9) for v in vals))
    for (name, mod), (ins, whole) in data.items():
        edges = np.linspace(*ranges[mod], args.bins + 1)
        hists[name, mod] = [np.histogram(v, edges)[0] / v.size if v is not None else None for v in (ins, whole)]
    centres = {mod: 0.5 * (np.linspace(*r, args.bins + 1)[1:] + np.linspace(*r, args.bins + 1)[:-1])
               for mod, r in ranges.items()}

    def draw(ax, name, mod, show_legend):
        h_in, h_whole = hists[name, mod]
        if h_whole is not None:
            ax.plot(centres[mod], h_whole, color=GREY, lw=1, label="whole volume")
        ax.plot(centres[mod], h_in, color=COLORS[mod], lw=1.8, label="inside mask")
        style(ax, args.log)
        if show_legend and h_whole is not None:
            ax.legend(frameon=False, fontsize=7)

    # One figure per modality, one panel per femur
    names = [io.femur_name(f) for f in args.femurs]
    for mod in ranges:
        present = [n for n in names if (n, mod) in hists]
        ncols = min(4, len(present))
        nrows = int(np.ceil(len(present) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), sharex=True, squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k >= len(present):
                ax.set_axis_off()
                continue
            draw(ax, present[k], mod, k == 0)
            ax.set_title(present[k], fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("intensity", fontsize=8)
        fig.suptitle(f"{mod}: intensity histograms (fraction of voxels)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"histograms_{mod}.png"), dpi=110)
        plt.close(fig)

    # One figure per femur, one panel per modality (different units, so separate axes)
    for name in names:
        mods = [m for m in ranges if (name, m) in hists]
        if not mods:
            continue
        fig, axes = plt.subplots(1, len(mods), figsize=(3.6 * len(mods), 2.8), squeeze=False)
        for ax, mod in zip(axes[0], mods):
            draw(ax, name, mod, True)
            ax.set_title(mod, fontsize=9)
            ax.set_xlabel("intensity", fontsize=8)
        axes[0, 0].set_ylabel("fraction of voxels", fontsize=8)
        fig.suptitle(name, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{name}_histograms.png"), dpi=110)
        plt.close(fig)
    print(f"Figures and histogram_stats.csv written to {args.out}")


if __name__ == "__main__":
    main()
