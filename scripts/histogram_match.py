"""Bring the micro-CT (HR) volumes to the intensity scale of the clinical CT (LR).

The histogram of the HR volume inside the bone mask is matched to the
histogram of the LR volume inside the same mask. The resulting mapping is
applied to the whole HR volume, which is saved as

    <root>/f_NNN/HR/f_NNN_hm.tif   (uint16 by default)

Example:
    python scripts/histogram_match.py --root /path/to/FACTS
    python scripts/histogram_match.py --root /path/to/FACTS --femurs 001 002
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, io  # noqa: E402
from facts.histmatch import apply_lut, compute_matching  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    p.add_argument("--femurs", nargs="*", default=config.FEMURS, help="Femur IDs (default: all)")
    p.add_argument("--dtype", default="uint16", help="Output data type (default: uint16)")
    p.add_argument("--offset", type=float, default=0.0,
                   help="Constant added after matching, e.g. 1024 if the LR scale (HU) has negative values. "
                        "It is stored in f_NNN_hm.json and added to the clinical CT when training/testing")
    p.add_argument("--zero_outside", action="store_true",
                   help="Set voxels outside the mask to 0 in the output")
    p.add_argument("--qa_dir", default="results/preprocessing",
                   help="Where to save histograms and mapping for quality assurance")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def check_range(lut, offset, dtype):
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        return
    lo, hi = lut[1].min() + offset, lut[1].max() + offset
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

        print(f"{name}: computing masked histograms")
        hr = io.open_volume(io.get_path(args.root, femur, "HR"))
        lr = io.open_volume(io.get_path(args.root, femur, "LR"))
        mask = io.open_volume(io.mask_path(args.root, femur))
        assert hr.shape == lr.shape == mask.shape, f"Shape mismatch: {hr.shape}, {lr.shape}, {mask.shape}"

        lut, stats = compute_matching(hr, lr, mask)
        check_range(lut, args.offset, args.dtype)

        print(f"{name}: writing {out_path}")
        out = io.create_volume(out_path, hr.shape, args.dtype)
        out_hist = np.zeros_like(stats["ref_hist"])
        for i in range(hr.shape[0]):
            im = apply_lut(hr[i], lut) + args.offset
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
            json.dump({"offset": args.offset, "lr_range": stats["ref_range"]}, f, indent=2)
        np.savez_compressed(os.path.join(args.qa_dir, f"{name}_histmatch.npz"),
                            lut_x=lut[0], lut_y=lut[1], hm_hist=out_hist, **stats)
        with open(os.path.join(args.qa_dir, f"{name}_histmatch.json"), "w") as f:
            json.dump({"femur": name, "hr_range": stats["src_range"], "lr_range": stats["ref_range"],
                       "offset": args.offset, "dtype": args.dtype, "zero_outside": args.zero_outside}, f, indent=2)
        print(f"{name}: done")


if __name__ == "__main__":
    main()
