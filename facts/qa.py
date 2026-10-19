"""Quality-assurance snapshots for the preprocessing (PNG figures + text summaries).

All figures show one 2D slice (along `axis`), cropped to the bone mask's bounding box,
with a zoomed-in region below. Intensities of the clinical CT are shown with the
histogram-matching offset added, so LR, HR_hm and SY_hm share one scale.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config, io  # noqa: E402

# Reference categorical palette (slots 1-3), validated for up to three series
C_LR, C_HR, C_SY = "#eb6834", "#2a78d6", "#1baf7a"
C_BOX = "#eda100"


def middle_slice(mask_vol, axis=config.SLICE_AXIS):
    """Index of the slice (along axis) through the middle of the mask."""
    coarse = np.asarray(mask_vol[::4, ::4, ::4]) > 0
    other = tuple(a for a in range(3) if a != axis)
    idx = np.flatnonzero(coarse.any(axis=other)) * 4
    return int(idx[len(idx) // 2])


def mask_bbox_2d(mask, pad=10):
    ys, xs = np.nonzero(mask)
    return (slice(max(ys.min() - pad, 0), ys.max() + pad + 1),
            slice(max(xs.min() - pad, 0), xs.max() + pad + 1))


def zoom_origin(mask, size):
    """Top-left corner of a size x size box around the mask centroid."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - size // 2, 0, max(mask.shape[0] - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(mask.shape[1] - size, 0)))
    return y0, x0


def window(im, mask, p=(1, 99)):
    lo, hi = np.percentile(im[mask], p)
    return float(lo), float(hi if hi > lo else lo + 1)


def _show(ax, im, lo, hi, title):
    ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def stages_figure(path, panels, mask, zoom=200, common=None, suptitle=""):
    """Panels side by side: full slice (own window), zoom (own window) and,
    if `common` = (lo, hi) is given, zoom in that common window (as seen in training).

    panels: list of (title, image, comparable) - comparable=True: on the LR/HR_hm scale (also shown in
    the training window); False: own window only; (lo, hi): fixed window
    """
    crop = mask_bbox_2d(mask)
    m = mask[crop]
    zoom = min(zoom, *m.shape)
    y0, x0 = zoom_origin(m, zoom)
    zs = (slice(y0, y0 + zoom), slice(x0, x0 + zoom))
    rows = 3 if common else 2
    fig, axes = plt.subplots(rows, len(panels), figsize=(2.7 * len(panels), 2.8 * rows), squeeze=False)
    for j, (title, im, comparable) in enumerate(panels):
        im = np.asarray(im, dtype=np.float32)[crop]
        lo, hi = comparable if isinstance(comparable, tuple) else window(im, m)
        comparable = comparable is True
        _show(axes[0, j], im, lo, hi, f"{title}\nwindow [{lo:.0f}, {hi:.0f}]")
        axes[0, j].add_patch(plt.Rectangle((x0, y0), zoom, zoom, fill=False, ec=C_BOX, lw=1))
        _show(axes[1, j], im[zs], lo, hi, "zoom, own window")
        if common:
            if comparable:
                _show(axes[2, j], im[zs], *common, f"zoom, training window [{common[0]:.0f}, {common[1]:.0f}]")
            else:
                axes[2, j].set_axis_off()
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overlay_figure(path, lr, hr_hm, mask, suptitle=""):
    """Registration check: clinical CT in magenta, histogram-matched micro-CT in green
    (white where both are bright, coloured where they disagree)."""
    crop = mask_bbox_2d(mask)
    m = mask[crop]
    a = np.clip((lr[crop] - window(lr[crop], m)[0]) / np.ptp(window(lr[crop], m)), 0, 1)
    b = np.clip((hr_hm[crop] - window(hr_hm[crop], m)[0]) / np.ptp(window(hr_hm[crop], m)), 0, 1)
    rgb = np.stack([a, b, a], axis=-1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("Registration check: clinical CT = magenta, HR_hm = green, overlap = white", fontsize=9)
    ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mapping_gain(lut_x, lut_y, n=400):
    """Slope of the mapping on a coarse grid (the fine 2^16-bin mapping is step-like)."""
    gx = np.linspace(lut_x[0], lut_x[-1], n + 1)
    gy = np.interp(gx, lut_x, lut_y)
    return 0.5 * (gx[1:] + gx[:-1]), np.diff(gy) / np.diff(gx)


def mapping_figure(path, lut_x, lut_y, offset, hr_hist=None, hr_edges=None, after=None, suptitle=""):
    """The histogram-matching mapping (micro-CT -> clinical-CT scale) and how much it
    stretches each intensity range (gain = slope of the mapping).

    after: optional list of (label, values, colour) inside the mask, for histograms after matching
    """
    n = 3 + (after is not None)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.3 * n), squeeze=False)
    axes = axes[:, 0]
    if hr_hist is not None:
        c = 0.5 * (hr_edges[:-1] + hr_edges[1:])
        k = 256
        hh = hr_hist[: len(hr_hist) // k * k].reshape(-1, k).sum(1)
        axes[0].plot(c[: len(hh) * k].reshape(-1, k).mean(1), hh / hh.sum(), color=C_HR, lw=2)
    axes[0].set_ylabel("fraction\nof voxels")
    axes[0].set_title("Micro-CT histogram inside the mask (before matching)", fontsize=9)
    axes[1].plot(lut_x, lut_y + offset, color=C_HR, lw=2)
    axes[1].set_ylabel("HR_hm value")
    axes[1].set_title("Mapping: original micro-CT value -> HR_hm value (incl. offset)", fontsize=9)
    gx, gain = mapping_gain(lut_x, lut_y)
    axes[2].semilogy(gx, np.maximum(gain, 1e-2), color=C_HR, lw=1.5)
    if np.ptp(gain) < 0.01 * abs(gain.mean()):  # linear mapping: constant gain
        axes[2].set_ylim(gain.mean() / 3, gain.mean() * 3)
        axes[2].set_title(f"Gain of the mapping: constant ({gain.mean():.3g}), linear mapping", fontsize=9)
    axes[2].set_ylabel("gain\n(log scale)")
    if not axes[2].get_title():
        axes[2].set_title("Gain of the mapping: how much each intensity range (and its noise) is stretched",
                          fontsize=9)
    for ax in axes[:3]:
        ax.set_xlabel("original micro-CT intensity")
    if after is not None:
        lo = min(np.percentile(v, 0.1) for _, v, _ in after)
        hi = max(np.percentile(v, 99.9) for _, v, _ in after)
        for label, v, col in after:
            h, e = np.histogram(v, bins=256, range=(lo, hi))
            axes[3].plot(0.5 * (e[:-1] + e[1:]), h / h.sum(), color=col, lw=2, label=label)
        axes[3].legend(frameon=False, fontsize=8)
        axes[3].set_xlabel("intensity (clinical-CT scale incl. offset)")
        axes[3].set_ylabel("fraction\nof voxels")
        axes[3].set_title("Histograms inside the mask after matching", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def profile_figure(path, lines, hr_orig, mask, suptitle=""):
    """Intensity profiles along a horizontal line through the middle of the mask.

    lines: list of (label, image, colour) on the clinical-CT scale (incl. offset).
    hr_orig: original micro-CT slice (own units, plotted separately).
    """
    ys, xs = np.nonzero(mask)
    row = int(ys.mean())
    cols = slice(xs.min(), xs.max() + 1)
    x = np.arange(cols.start, cols.stop)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for label, im, col in lines:
        axes[0].plot(x, np.asarray(im, np.float32)[row, cols], color=col, lw=1.2, label=label)
    axes[0].legend(frameon=False, fontsize=8, ncol=len(lines))
    axes[0].set_ylabel("clinical-CT scale")
    axes[0].set_title(f"Profile along row {row}", fontsize=9)
    axes[1].plot(x, np.asarray(hr_orig, np.float32)[row, cols], color=C_HR, lw=1.2)
    axes[1].set_ylabel("original micro-CT")
    axes[1].set_xlabel("column (voxels)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_slice(path, idx, axis):
    return io.get_slice(io.open_volume(path), idx, axis)


def inspect_femur(root, femur, out_dir, qa_dir="results/preprocessing", idx=None, axis=config.SLICE_AXIS,
                  zoom=200):
    """Write snapshot figures and a text summary of the preprocessing of one femur.

    Uses the mapping saved by scripts/histogram_match.py (<qa_dir>/f_NNN_histmatch.npz) to
    recompute HR_hm in floating point, so problems in the uint16 conversion/writing show up
    as a difference to the saved file. Returns the summary as a dict.
    """
    from .histmatch import apply_lut

    name = io.femur_name(femur)
    os.makedirs(out_dir, exist_ok=True)
    mask_vol = io.open_volume(io.mask_path(root, femur))
    if idx is None:
        idx = middle_slice(mask_vol, axis)
    tag = f"{name}_axis{axis}_slice{idx}"
    title = f"{name}, slice {idx} along axis {axis}"

    mask = io.get_slice(mask_vol, idx, axis) > 0
    offset = io.hm_offset(root, femur)
    lr = io.get_slice(io.open_volume(io.get_path(root, femur, "LR")), idx, axis).astype(np.float32) + offset
    hr = io.get_slice(io.open_volume(io.get_path(root, femur, "HR")), idx, axis).astype(np.float32)
    hm_path, sy_path = io.target_path(root, femur), io.input_path(root, femur, "synth")
    hm_vol = io.open_volume(hm_path) if os.path.exists(hm_path) else None
    hm = io.get_slice(hm_vol, idx, axis).astype(np.float32) if hm_vol is not None else None
    sy = load_slice(sy_path, idx, axis).astype(np.float32) if os.path.exists(sy_path) else None

    npz = os.path.join(qa_dir, f"{name}_histmatch.npz")
    lut = np.load(npz) if os.path.exists(npz) else None
    hm_float = None
    if lut is not None and "lin_a" in lut:
        hm_float = (float(lut["lin_a"]) * hr + float(lut["lin_b"]) + offset).astype(np.float32)
    elif lut is not None:
        hm_float = apply_lut(hr, (lut["lut_x"], lut["lut_y"])).astype(np.float32) + offset

    # Window used for training: percentiles 0.5/99.5 of the (offset) clinical CT inside the mask
    lr_vol = io.open_volume(io.get_path(root, femur, "LR"))
    sub_m = np.asarray(mask_vol[::4, ::4, ::4]) > 0
    train_win = tuple(float(v) for v in np.percentile(np.asarray(lr_vol[::4, ::4, ::4])[sub_m] + offset, (0.5, 99.5)))

    summary = {"femur": name, "slice": idx, "axis": axis, "offset": offset, "training_window": train_win,
               "method": str(lut["method"]) if lut is not None and "method" in lut else "histogram"}
    if lut is not None and "lin_a" in lut:
        summary["linear"] = f"HR_hm = {float(lut['lin_a']):.4g} * HR + {float(lut['lin_b']):.4g}"
    for label, im in [("LR+offset", lr), ("HR", hr), ("HR_hm", hm), ("HR_hm_float", hm_float), ("SY_hm", sy)]:
        if im is not None:
            summary[label] = {k: float(v) for k, v in
                              zip(("min", "p1", "median", "p99", "max"), np.percentile(im[mask], [0, 1, 50, 99, 100]))}
    if hm_vol is not None:
        info = np.iinfo(hm_vol.dtype) if np.issubdtype(hm_vol.dtype, np.integer) else None
        summary["HR_hm_dtype"] = str(hm_vol.dtype)
        if info:
            summary["HR_hm_clipped_low_%"] = float(np.mean(hm[mask] <= info.min) * 100)
            summary["HR_hm_clipped_high_%"] = float(np.mean(hm[mask] >= info.max) * 100)
    if hm is not None and hm_float is not None:
        d = np.abs(hm - hm_float)[mask]
        summary["HR_hm_file_vs_recomputed_maxabs"] = float(d.max())
        summary["HR_hm_file_vs_recomputed_mean"] = float(d.mean())
    if lut is not None:
        gx, gain = mapping_gain(lut["lut_x"], lut["lut_y"])
        hr_m = hr[mask]
        summary["mapping_gain_at_HR_p1_p50_p99"] = [float(np.interp(np.percentile(hr_m, p), gx, gain))
                                                     for p in (1, 50, 99)]
    # Noise estimate in a flat region: std of the difference of neighbouring voxels inside the mask
    for label, im in [("LR+offset", lr), ("HR_hm", hm), ("SY_hm", sy)]:
        if im is not None:
            dm = mask[:, 1:] & mask[:, :-1]
            summary.setdefault("neighbour_diff_std", {})[label] = float(np.std(np.diff(im, axis=1)[dm]) / np.sqrt(2))

    panels = [("Clinical CT (LR + offset)", lr, True), ("Micro-CT original (HR)", hr, False)]
    if hm_float is not None:
        panels.append(("HR_hm recomputed (float)", hm_float, True))
    if hm is not None:
        panels.append((f"HR_hm file ({summary.get('HR_hm_dtype')})", hm, True))
    if hm is not None and hm_float is not None:
        panels.append(("|file - recomputed|\n(<= 0.5 unless clipped)", np.abs(hm - hm_float), (0.0, 2.0)))
    if sy is not None:
        panels.append(("Synthetic (SY_hm)", sy, True))
    stages_figure(os.path.join(out_dir, f"{tag}_stages.png"), panels, mask, zoom, train_win, title)
    if hm is not None:
        overlay_figure(os.path.join(out_dir, f"{tag}_overlay.png"), lr, hm, mask, title)
    if lut is not None:
        after = [("Clinical CT (LR + offset)", lr[mask], C_LR)]
        if hm is not None:
            after.append(("HR_hm", hm[mask], C_HR))
        if sy is not None:
            after.append(("SY_hm", sy[mask], C_SY))
        mapping_figure(os.path.join(out_dir, f"{name}_mapping.png"), lut["lut_x"], lut["lut_y"], offset,
                       lut["src_hist"], lut["src_edges"], after, f"{name} (histograms of slice {idx})")
    lines = [("LR + offset", lr, C_LR)]
    if hm is not None:
        lines.append(("HR_hm", hm, C_HR))
    if sy is not None:
        lines.append(("SY_hm", sy, C_SY))
    profile_figure(os.path.join(out_dir, f"{tag}_profile.png"), lines, hr, mask, title)

    with open(os.path.join(out_dir, f"{tag}_summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary
