"""Bring the micro-CT (HR) volumes to the intensity scale of the clinical CT (LR).

Two methods (--method):

  histogram  The histogram of the HR volume inside the bone mask is matched to the
             histogram of the LR volume inside the same mask (non-linear mapping).
             Note: the micro-CT has sharp, separated intensity peaks while the clinical
             CT is blurred, so the mapping stretches narrow intensity ranges and
             amplifies micro-CT noise there.
  linear     HR_hm = a * HR + b, with a, b fitted so that a * blur(HR) + b matches the
             clinical CT inside the mask (least squares; the blur width is chosen
             automatically). Keeps noise and contrast proportions of the micro-CT.

The mapping is applied to the whole HR volume, which is saved as

    <root>/f_NNN/HR/f_NNN_hm.tif   (uint16 by default)

Example:
    python scripts/histogram_match.py --root /path/to/FACTS --method linear --offset 1024
    python scripts/histogram_match.py --root /path/to/FACTS --femurs 001 002
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, io, qa  # noqa: E402
from facts.histmatch import apply_lut, compute_matching, fit_linear  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    p.add_argument("--femurs", nargs="*", default=config.FEMURS, help="Femur IDs (default: all)")
    p.add_argument("--method", choices=["histogram", "linear"], default="histogram",
                   help="Intensity mapping: histogram matching or linear calibration (see above)")
    p.add_argument("--dtype", default="uint16", help="Output data type (default: uint16)")
    p.add_argument("--offset", type=float, default=0.0,
                   help="Constant added after matching, e.g. 1024 if the LR scale (HU) has negative values. "
                        "It is stored in f_NNN_hm.json and added to the clinical CT when training/testing")
    p.add_argument("--zero_outside", action="store_true",
                   help="Set voxels outside the mask to 0 in the output")
    p.add_argument("--qa_dir", default="results/preprocessing",
                   help="Where to save histograms and mapping for quality assurance")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_snapshots", action="store_true",
                   help="Do not write snapshot figures to <qa_dir>/snapshots (see scripts/inspect_preprocessing.py)")
    return p.parse_args()


def hist_percentile(hist, edges, p):
    cdf = np.cumsum(hist) / hist.sum()
    return float(np.interp(p / 100, cdf, edges[1:]))


def report_clipping(name, stats, a, b, offset, dtype):
    """Print the fraction of voxels inside the mask that the linear mapping pushes out of dtype's range."""
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return
    info = np.iinfo(dtype)
    centres = 0.5 * (stats["src_edges"][:-1] + stats["src_edges"][1:])
    mapped = a * centres + b + offset
    n = stats["src_hist"].sum()
    low = stats["src_hist"][mapped < info.min].sum() / n * 100
    high = stats["src_hist"][mapped > info.max].sum() / n * 100
    print(f"{name}: clipped to the {dtype} range inside the mask: {low:.3f}% low, {high:.3f}% high")
    if low > 0.1 or high > 0.1:
        lo = a * hist_percentile(stats["src_hist"], stats["src_edges"], 0.01) + b
        print(f"{name}: WARNING: consider a larger offset to avoid clipping noise, "
              f"e.g. --offset {int(np.ceil(-lo / 256) * 256)} (use the same offset for all femurs), or --dtype float32")


def check_range(lo, hi, offset, dtype):
    """Exit if the mapped intensity range [lo, hi] (+ offset) does not fit in dtype."""
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return
    lo, hi = lo + offset, hi + offset
    info = np.iinfo(dtype)
    if lo < info.min or hi > info.max:
        sys.exit(f"Matched intensities [{lo:.1f}, {hi:.1f}] do not fit in {dtype}. "
                 f"Use --offset (e.g. --offset {int(np.ceil(-lo))}) or --dtype float32.")


def main():
    args = parse_args()
    os.makedirs(args.qa_dir, exist_ok=True)
    for femur in args.femurs:
        name = io.femur_name(femur)
        out_path = io.target_path(args.root, femur)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{name}: {out_path} exists, skipping (use --overwrite)")
            continue

        print(f"{name}: computing masked histograms (method: {args.method})")
        hr = io.open_volume(io.get_path(args.root, femur, "HR"))
        lr = io.open_volume(io.get_path(args.root, femur, "LR"))
        mask = io.open_volume(io.mask_path(args.root, femur))
        assert hr.shape == lr.shape == mask.shape, f"Shape mismatch: {hr.shape}, {lr.shape}, {mask.shape}"

        lut, stats = compute_matching(hr, lr, mask)
        info = {"method": args.method}
        if args.method == "histogram":
            check_range(lut[1].min(), lut[1].max(), args.offset, args.dtype)

            def transform(im):
                return apply_lut(im, lut)
        else:
            print(f"{name}: fitting linear calibration")
            a, b, sigma, fits = fit_linear(hr, lr, mask)
            print(f"{name}: HR_hm = {a:.4g} * HR + {b:.4g}  (blur sigma {sigma:g} voxels, "
                  f"R^2 = {max(f[3] for f in fits):.3f})")
            info.update(a=a, b=b, sigma=sigma, fits=[dict(zip(("sigma", "a", "b", "r2"), f)) for f in fits])
            # Micro-CT noise in air/marrow can map below the clinical CT's floor (-1024 HU);
            # report how much of the mask would be clipped instead of aborting
            report_clipping(name, stats, a, b, args.offset, args.dtype)
            lut = (lut[0], a * lut[0] + b)   # the linear mapping, for the QA figures

            def transform(im):
                return a * im.astype(np.float64) + b

        print(f"{name}: writing {out_path}")
        out = io.create_volume(out_path, hr.shape, args.dtype)
        out_hist = np.zeros_like(stats["ref_hist"])
        for i in range(hr.shape[0]):
            im = transform(hr[i]) + args.offset
            m = mask[i] > 0
            if args.zero_outside:
                im[~m] = 0
            out[i] = io.to_dtype(im, args.dtype)
            if m.any():
                out_hist += np.histogram(out[i][m].astype(np.float64) - args.offset, bins=len(out_hist),
                                         range=stats["ref_range"])[0]
        out.flush()
        del out

        # Sidecar with the offset, read by the training/test code (facts.io.hm_offset)
        with open(io.get_path(args.root, femur, *config.TARGET_KEY, ext="json"), "w") as f:
            json.dump({"offset": args.offset, "lr_range": stats["ref_range"], **info}, f, indent=2)
        lin = {"lin_a": info["a"], "lin_b": info["b"]} if args.method == "linear" else {}
        np.savez_compressed(os.path.join(args.qa_dir, f"{name}_histmatch.npz"),
                            lut_x=lut[0], lut_y=lut[1], hm_hist=out_hist, method=args.method, **lin, **stats)
        with open(os.path.join(args.qa_dir, f"{name}_histmatch.json"), "w") as f:
            json.dump({"femur": name, "hr_range": stats["src_range"], "lr_range": stats["ref_range"],
                       "offset": args.offset, "dtype": args.dtype, "zero_outside": args.zero_outside, **info},
                      f, indent=2)
        if not args.no_snapshots:
            snap_dir = os.path.join(args.qa_dir, "snapshots")
            qa.inspect_femur(args.root, femur, snap_dir, args.qa_dir)
            print(f"{name}: snapshots in {snap_dir}")
        print(f"{name}: done")


if __name__ == "__main__":
    main()
