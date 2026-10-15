"""Print shape, dtype and intensity statistics (inside the bone mask) of a femur's volumes.

Use it before preprocessing, e.g. to see whether the clinical CT has negative
values (then histogram_match.py needs --offset), and afterwards to check the
histogram-matched and synthetic volumes.

Example:
    python scripts/check_data.py --root /path/to/FACTS --femurs 001
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, io  # noqa: E402

VOLUMES = [("HR", ""), ("LR", ""), ("SY", ""), ("HR", "hm"), ("SY", "hm"), ("MS", "")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--femurs", nargs="*", default=config.FEMURS, help="Femur IDs (default: all)")
    p.add_argument("--step", type=int, default=4, help="Sub-sampling of the voxels used for the statistics")
    args = p.parse_args()

    for femur in args.femurs:
        print(f"\n=== {io.femur_name(femur)} ===")
        mask = io.open_volume(io.mask_path(args.root, femur))
        m = np.asarray(mask[::args.step, ::args.step, ::args.step]) > 0
        print(f"mask: {m.mean() * 100:.1f}% of the volume is bone")
        for mod, suffix in VOLUMES:
            path = io.get_path(args.root, femur, mod, suffix)
            label = f"{mod}{'_' + suffix if suffix else ''}"
            if not os.path.exists(path):
                print(f"{label:6s} (not found: {path})")
                continue
            vol = io.open_volume(path)
            v = np.asarray(vol[::args.step, ::args.step, ::args.step])
            mmap = "memory-mapped" if isinstance(vol, np.memmap) else "read into memory"
            inside = v[m] if v.shape == m.shape else v.ravel()
            q = np.percentile(inside, [0, 1, 50, 99, 100])
            print(f"{label:6s} {str(vol.shape):20s} {str(vol.dtype):8s} ({mmap}) | inside mask: "
                  f"min {q[0]:.1f}, p1 {q[1]:.1f}, median {q[2]:.1f}, p99 {q[3]:.1f}, max {q[4]:.1f}")
        hm_offset = io.hm_offset(args.root, femur)
        if hm_offset:
            print(f"offset from histogram matching: {hm_offset}")


if __name__ == "__main__":
    main()
