"""Uncertainty (input-noise sensitivity) of trained models, from synthetic to real input.

For each alpha in --alphas, the input is a blend (1 - alpha) * synthetic + alpha * real of
the test femurs. Each model super-resolves N noisy versions of it (see facts/uncertainty.py
for how the noise is added). Per voxel inside the mask:

  uncertainty = standard deviation of the N SR outputs
  error       = |mean of the N SR outputs - micro-CT|

Written to --out:
  uncertainty.csv            per run, femur and alpha: mean uncertainty and error (normalised
                             and in intensity units), Spearman correlation between uncertainty
                             and error, AUSE (sparsification), noise levels used
  uncertainty_vs_alpha.png   uncertainty, error, correlation and AUSE against alpha
  examples_<run>.png         input | SR | uncertainty | error | micro-CT at selected alphas
  sparsification.npz         sparsification curves

Example:
    python uncertainty.py --root $DATA_ROOT
    python uncertainty.py --root $DATA_ROOT --runs runs/esrgan_real --n_samples 20 --slice_stride 25
"""
import argparse
import csv
import json
import os
import time

import matplotlib
import numpy as np
import torch
from scipy.stats import spearmanr

from facts import config, io
from facts.data import FemurVolume, Normalizer
from facts.inference import load_generator, sr_batch
from facts.uncertainty import make_inputs, noise_levels, robust_noise_sigma, sparsification

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-4 (fixed order)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DEFAULT_RUNS = ["runs/srgan_real", "runs/srgan_synth", "runs/esrgan_real", "runs/esrgan_synth"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    p.add_argument("--runs", nargs="*", default=None, help=f"Run directories (default: those of {DEFAULT_RUNS} that exist)")
    p.add_argument("--femurs", nargs="*", default=config.TEST_FEMURS)
    p.add_argument("--alphas", nargs="*", type=float, default=[round(a, 2) for a in np.linspace(0, 1, 11)],
                   help="Blending coefficients: 0 = synthetic input, 1 = real input")
    p.add_argument("--n_samples", type=int, default=10, help="Noisy repetitions per input")
    p.add_argument("--noise", choices=["topup", "fixed"], default="topup",
                   help="topup: equalise the noise level of all blends to that of the real CT, "
                        "then perturb; fixed: perturb only")
    p.add_argument("--perturb_scale", type=float, default=1.0,
                   help="Perturbation noise std as a multiple of the real CT's noise std")
    p.add_argument("--slice_stride", type=int, default=50, help="Use every n'th slice with bone")
    p.add_argument("--voxels_per_slice", type=int, default=20000,
                   help="Voxels sampled per slice for correlation and sparsification")
    p.add_argument("--example_alphas", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/uncertainty")
    return p.parse_args()


def load_cropped(path, bbox, axis):
    """A volume cropped to bbox, with slices along `axis` first (like FemurVolume)."""
    return np.ascontiguousarray(np.moveaxis(io.open_volume(path)[bbox], axis, 0))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = args.runs or [r for r in DEFAULT_RUNS if os.path.exists(os.path.join(r, "generator.pt"))]
    if not runs:
        raise SystemExit("No trained runs found (looked for generator.pt in the run directories)")

    models = {}
    for r in runs:
        G, norm, meta = load_generator(r, device)
        models[os.path.basename(os.path.normpath(r))] = (G, norm, meta)
    axes = {m[2]["axis"] for m in models.values()}
    assert len(axes) == 1, f"Runs use different slice axes: {axes}"
    axis = axes.pop()
    names = list(models)
    print(f"Runs: {names} | alphas: {args.alphas} | N = {args.n_samples} | noise: {args.noise}")

    rows, curves, examples = [], {}, {}
    for fi, femur in enumerate(args.femurs):
        t0 = time.time()
        f = FemurVolume(args.root, femur, "real", axis=axis)
        sy = load_cropped(io.input_path(args.root, femur, "synth"), f.bbox, axis)
        slices = [z for z in range(f.shape[0]) if f.coarse_mask[z].any()][::args.slice_stride]
        masks = {z: f.mask[z] > 0 for z in slices}
        real = {z: f.lr[z].astype(np.float32) + f.lr_offset for z in slices}

        # Normalisation for the metrics, as in test.py: fixed per femur from the micro-CT
        lo, hi = np.percentile(f.hr[::4, ::4, ::4][f.coarse_mask[::4]], (0.5, 99.5))
        mnorm = Normalizer(lo, hi)
        sigma_real = robust_noise_sigma([real[z] for z in slices], [masks[z] for z in slices])
        sigma_synth = robust_noise_sigma([sy[z] for z in slices], [masks[z] for z in slices])
        sigma_p = args.perturb_scale * sigma_real
        print(f"{f.name}: {len(slices)} slices | noise std real {sigma_real:.1f}, synthetic {sigma_synth:.1f} "
              f"(intensity units) | perturbation {sigma_p:.1f}")

        acc = {(n, a): {"unc": 0.0, "abs": 0.0, "sq": 0.0, "n": 0, "u": [], "e": []}
               for n in names for a in args.alphas}
        ex_z = slices[len(slices) // 2]
        rng_vox = np.random.default_rng([args.seed, fi])
        for z in slices:
            m = masks[z]
            hr_n = mnorm(f.hr[z])
            vox = np.flatnonzero(m)
            vox = rng_vox.choice(vox, min(args.voxels_per_slice, len(vox)), replace=False)
            for ai, alpha in enumerate(args.alphas):
                s_eq, s_p = noise_levels(alpha, sigma_real, sigma_synth, sigma_p, args.noise)
                # Same noise for every model (seeded by femur, slice and alpha)
                x = make_inputs(sy[z], real[z], alpha, s_eq, s_p, args.n_samples,
                                np.random.default_rng([args.seed, fi, z, ai, 0]),
                                np.random.default_rng([args.seed, fi, z, ai, 1]))
                for name in names:
                    G, norm, _ = models[name]
                    sr = norm.inverse(sr_batch(G, norm(x), device, args.tile, args.overlap))
                    sr_n = mnorm(sr)
                    mean, std = sr_n.mean(0), sr_n.std(0, ddof=1)
                    err = np.abs(mean - hr_n)
                    a = acc[name, alpha]
                    a["unc"] += float(std[m].sum())
                    a["abs"] += float(err[m].sum())
                    a["sq"] += float((err[m] ** 2).sum())
                    a["n"] += int(m.sum())
                    a["u"].append(std.ravel()[vox])
                    a["e"].append(err.ravel()[vox])
                    if fi == 0 and z == ex_z and alpha in args.example_alphas:
                        examples[name, alpha] = dict(x=mnorm(x.mean(0)), sr=mean, std=std, err=err, hr=hr_n, mask=m)
                    if z == slices[-1]:
                        u, e = np.concatenate(a["u"]), np.concatenate(a["e"])
                        fr, c_unc, c_orc, ause = sparsification(u, e)
                        curves[name, f.name, alpha] = (c_unc, c_orc)
                        scale = hi - lo
                        meta = models[name][2]
                        rows.append({
                            "run": name, "model": meta["model"], "train_data": meta["train_data"], "femur": f.name,
                            "alpha": alpha, "n_slices": len(slices), "n_samples": args.n_samples,
                            "uncertainty": a["unc"] / a["n"], "mae": a["abs"] / a["n"],
                            "rmse": float(np.sqrt(a["sq"] / a["n"])),
                            "uncertainty_intensity": a["unc"] / a["n"] * scale, "mae_intensity": a["abs"] / a["n"] * scale,
                            "spearman": float(spearmanr(u, e)[0]), "ause": ause,
                            "sigma_real": sigma_real, "sigma_synth": sigma_synth, "sigma_eq": s_eq, "sigma_perturb": s_p})
        del f, sy, real
        print(f"{femur}: done in {(time.time() - t0) / 60:.1f} min")

    # ---- outputs
    with open(os.path.join(args.out, "uncertainty.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(os.path.join(args.out, "sparsification.npz"), fractions=fr,
                        **{f"{n}__{fe}__{a}__{k}": c for (n, fe, a), cs in curves.items()
                           for k, c in zip(("unc", "oracle"), cs)})
    with open(os.path.join(args.out, "settings.json"), "w") as fh:
        json.dump(vars(args) | {"runs": runs}, fh, indent=2)

    plot_summary(rows, names, args.alphas, os.path.join(args.out, "uncertainty_vs_alpha.png"))
    for name in names:
        plot_examples(examples, name, [a for a in args.example_alphas if (name, a) in examples],
                      os.path.join(args.out, f"examples_{name}.png"), args.femurs[0])
    print(f"Results written to {args.out}")


def plot_summary(rows, names, alphas, path):
    """Mean over femurs of each quantity against alpha, one line per run."""
    panels = [("uncertainty", "Uncertainty (mean std of SR)"), ("mae", "Error (MAE of mean SR)"),
              ("spearman", "Spearman correlation\nuncertainty vs. error"), ("ause", "AUSE (lower = uncertainty\nranks errors better)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (key, title) in zip(axes, panels):
        for i, name in enumerate(names):
            y = [np.mean([r[key] for r in rows if r["run"] == name and r["alpha"] == a]) for a in alphas]
            ax.plot(alphas, y, "-o", color=COLORS[i % len(COLORS)], lw=2, ms=4, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("alpha (0 = synthetic input, 1 = real input)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("normalised intensity", fontsize=8)
    axes[1].set_ylabel("normalised intensity", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(examples, name, alphas, path, femur):
    if not alphas:
        return
    ex0 = examples[name, alphas[0]]
    ys, xs = np.nonzero(ex0["mask"])
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    # Own colour scale for uncertainty and for error (their magnitudes differ a lot)
    vmax = {k: max(np.percentile(examples[name, a][k][ex0["mask"]], 99) for a in alphas) for k in ("std", "err")}
    ims = {}
    cols = [("x", "Input (mean of noisy inputs)"), ("sr", "SR (mean)"), ("hr", "Micro-CT"),
            ("std", "Uncertainty (std)"), ("err", "|Error|")]
    fig, axes = plt.subplots(len(alphas), len(cols), figsize=(3.0 * len(cols) + 1, 3.0 * len(alphas)),
                             squeeze=False, layout="constrained")
    for i, a in enumerate(alphas):
        ex = examples[name, a]
        for j, (k, title) in enumerate(cols):
            ax = axes[i, j]
            if k in ("std", "err"):
                ims[k] = ax.imshow(ex[k][crop], cmap="magma", vmin=0, vmax=vmax[k])
            else:
                ax.imshow(np.clip(ex[k][crop], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            ax.set_title(f"{title}\nalpha = {a:g}" if j == 0 else title, fontsize=8)
    fig.colorbar(ims["std"], ax=axes[:, 3].tolist(), shrink=0.6, label="uncertainty (normalised intensity)")
    fig.colorbar(ims["err"], ax=axes[:, 4].tolist(), shrink=0.6, label="|error| (normalised intensity)")
    fig.suptitle(f"{name}, {io.femur_name(femur)}", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
