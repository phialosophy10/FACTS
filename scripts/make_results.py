"""Collect test metrics and example slices into data and figures for the project page.

Reads
    <runs_dir>/*/test_{real,synth}/metrics.json      (from test.py --run ...)
    <results_dir>/baseline/test_{real,synth}/metrics.json   (from test.py --baseline)
    <results_dir>/preprocessing/f_NNN_histmatch.npz  (from scripts/histogram_match.py)
and writes
    docs/_data/results.json             summary + per-femur tables (rendered by the page)
    docs/assets/img/results/*.png       visual comparisons
    docs/assets/img/histmatch_*.png     histogram matching figures

Example:
    python scripts/make_results.py --runs_dir runs --results_dir results
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

METRICS = ["psnr", "ssim", "lpips", "bvtv", "bvtv_err", "bvtv_hr", "tbth_um", "tbth_err_um", "tbth_hr_um"]
MODEL_ORDER = {"Input": 0, "SRGAN": 1, "ESRGAN": 2}
DATA_LABEL = {"real": "Real (clinical CT)", "synth": "Synthetic", "-": "-"}
# Reference categorical palette, slots 1-3 (validated for up to three series)
COLORS = {"HR": "#2a78d6", "LR": "#eb6834", "HR_hm": "#1baf7a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--docs_dir", default="docs")
    p.add_argument("--zoom", type=int, default=200, help="Size (voxels) of the zoomed-in region")
    return p.parse_args()


def load_metrics(args):
    files = sorted(glob.glob(os.path.join(args.runs_dir, "*", "test_*", "metrics.json")))
    files += sorted(glob.glob(os.path.join(args.results_dir, "baseline", "test_*", "metrics.json")))
    rows = []
    for fn in files:
        with open(fn) as f:
            for r in json.load(f):
                r["dir"] = os.path.dirname(fn)
                rows.append(r)
    return rows


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["train_data"], r["test_data"]), []).append(r)
    summary = []
    for (model, train, test), rs in groups.items():
        s = {"model": model, "train_data": DATA_LABEL[train], "test_data": DATA_LABEL[test],
             "test_key": test, "n_femurs": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if m in r]
            if vals:
                s[m] = round(float(np.mean(vals)), 4)
                s[m + "_std"] = round(float(np.std(vals)), 4)
        summary.append(s)
    summary.sort(key=lambda s: (s["test_key"], MODEL_ORDER.get(s["model"], 9), s["train_data"]))
    return summary


def bbox_crop(*ims, mask):
    ys, xs = np.nonzero(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return [im[sl] for im in ims], mask[sl]


def zoom_box(mask, size):
    """Square region of `size` pixels centred on the mask centroid."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - size // 2, 0, max(mask.shape[0] - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(mask.shape[1] - size, 0)))
    return y0, x0


def comparison_figures(rows, out_dir, zoom):
    """One figure per (test data, femur): input | SR of each model | micro-CT, with zoom."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    keys = sorted({(r["test_data"], r["femur"]) for r in rows})
    for test, femur in keys:
        rs = [r for r in rows if r["test_data"] == test and r["femur"] == femur and r["model"] != "Input"]
        rs.sort(key=lambda r: (MODEL_ORDER.get(r["model"], 9), r["train_data"]))
        exs = [(r, np.load(os.path.join(r["dir"], f"{femur}_example.npz"))) for r in rs
               if os.path.exists(os.path.join(r["dir"], f"{femur}_example.npz"))]
        if not exs:
            continue
        ex0 = exs[0][1]
        panels = [("Input (" + ("clinical CT" if test == "real" else "synthetic") + ")", ex0["lr"])]
        panels += [(f"{r['model']} (trained on {r['train_data']})", ex["sr"]) for r, ex in exs]
        panels.append(("Micro-CT (HR)", ex0["hr"]))
        ims, mask = bbox_crop(*[p[1] for p in panels], mask=ex0["mask"])
        y0, x0 = zoom_box(mask, zoom)
        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
        for j, ((title, _), im) in enumerate(zip(panels, ims)):
            axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[0, j].add_patch(plt.Rectangle((x0, y0), zoom, zoom, fill=False, ec="#eda100", lw=1.5))
            axes[0, j].set_title(title, fontsize=9)
            axes[1, j].imshow(im[y0:y0 + zoom, x0:x0 + zoom], cmap="gray", vmin=0, vmax=1)
            for ax in axes[:, j]:
                ax.set_axis_off()
        fig.suptitle(f"{femur}, slice {int(ex0['slice'])} - {DATA_LABEL[test]} input", fontsize=10)
        fig.tight_layout()
        fn = f"{test}_{femur}.png"
        fig.savefig(os.path.join(out_dir, fn), dpi=110)
        plt.close(fig)
        made.append({"test_key": test, "femur": femur, "file": fn})
    return made


def histmatch_figures(pre_dir, out_dir):
    made = []
    for fn in sorted(glob.glob(os.path.join(pre_dir, "f_*_histmatch.npz"))):
        d = np.load(fn)
        femur = os.path.basename(fn)[:5]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        c_src = 0.5 * (d["src_edges"][:-1] + d["src_edges"][1:])
        c_ref = 0.5 * (d["ref_edges"][:-1] + d["ref_edges"][1:])

        def smooth(h, n=256):
            h = h.reshape(-1, n).sum(1).astype(float)
            return h / h.sum()

        def centres(c, n=256):
            return c.reshape(-1, n).mean(1)

        axes[0].plot(centres(c_src), smooth(d["src_hist"]), color=COLORS["HR"], lw=2, label="Micro-CT (original)")
        axes[0].set_title("Before: micro-CT intensities", fontsize=10)
        axes[1].plot(centres(c_ref), smooth(d["ref_hist"]), color=COLORS["LR"], lw=2, label="Clinical CT")
        axes[1].plot(centres(c_ref), smooth(d["hm_hist"]), color=COLORS["HR_hm"], lw=2, ls="--",
                     label="Micro-CT (calibrated)")
        axes[1].set_title("After: on the clinical-CT scale", fontsize=10)
        for ax in axes:
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Fraction of voxels in mask")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(femur, fontsize=10)
        fig.tight_layout()
        out = f"histmatch_{femur}.png"
        fig.savefig(os.path.join(out_dir, out), dpi=110)
        plt.close(fig)
        made.append({"femur": femur, "file": out})
    return made


def main():
    args = parse_args()
    img_dir = os.path.join(args.docs_dir, "assets", "img")
    os.makedirs(os.path.join(args.docs_dir, "_data"), exist_ok=True)
    rows = load_metrics(args)
    summary = summarise(rows)
    figs = comparison_figures(rows, os.path.join(img_dir, "results"), args.zoom)
    hm = histmatch_figures(os.path.join(args.results_dir, "preprocessing"), img_dir)
    per_femur = sorted(({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "dir"}
                        for r in rows),
                       key=lambda r: (r["test_data"], r["femur"], MODEL_ORDER.get(r["model"], 9), r["train_data"]))
    for r in per_femur:
        r["train_data"] = DATA_LABEL[r["train_data"]]
        r["test_key"] = r["test_data"]
        r["test_data"] = DATA_LABEL[r["test_data"]]
    out = {"summary": summary, "per_femur": per_femur, "figures": figs, "histmatch": hm}
    with open(os.path.join(args.docs_dir, "_data", "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{len(rows)} result rows, {len(figs)} comparison figures, {len(hm)} histogram figures")


if __name__ == "__main__":
    main()
