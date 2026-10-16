"""Snapshot figures for checking the preprocessing of already processed femurs.

For each femur, one slice (default: through the middle of the mask, along the training
slice axis) is shown at every stage, and written to --out:

  *_stages.png    clinical CT (+offset) | original micro-CT | HR_hm recomputed in float |
                  HR_hm as saved (uint16) | difference | SY_hm
                  rows: full slice, zoom (own window), zoom in the window used for training
  *_overlay.png   registration check (clinical CT magenta, HR_hm green)
  *_mapping.png   histogram-matching mapping and its gain (noise amplification)
  *_profile.png   intensity profiles along one row
  *_summary.txt   statistics, clipping, file-vs-recomputed difference, noise estimates

Example:
    python scripts/inspect_preprocessing.py --root $DATA_ROOT --femurs 001
    python scripts/inspect_preprocessing.py --root $DATA_ROOT --femurs 001 --slice 700 --zoom 150
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, qa  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--femurs", nargs="*", default=["001"])
    p.add_argument("--slice", type=int, default=None, help="Slice index (default: middle of the mask)")
    p.add_argument("--axis", type=int, default=config.SLICE_AXIS)
    p.add_argument("--zoom", type=int, default=200, help="Size of the zoomed-in region (voxels)")
    p.add_argument("--qa_dir", default="results/preprocessing", help="Where histogram_match.py saved the mapping")
    p.add_argument("--out", default="results/inspect")
    args = p.parse_args()
    for femur in args.femurs:
        s = qa.inspect_femur(args.root, femur, args.out, args.qa_dir, args.slice, args.axis, args.zoom)
        print(f"\n=== {s['femur']} (slice {s['slice']}, axis {s['axis']}) -> {args.out}")
        for k, v in s.items():
            if k not in ("femur", "slice", "axis"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
