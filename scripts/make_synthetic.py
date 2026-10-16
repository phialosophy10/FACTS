"""Create synthetic low-resolution volumes from the histogram-matched micro-CT.

Each 2D slice (along --axis) of <root>/f_NNN/HR/f_NNN_hm.tif is
  1. blurred with a Gaussian kernel (sigma = 1.2),
  2. downsampled by a factor of 4,
  3. upsampled 4x with linear interpolation back to the original size,
and the result is saved as <root>/f_NNN/SY/f_NNN_hm.tif with the same dtype.

Example:
    python scripts/make_synthetic.py --root /path/to/FACTS
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts import config, io, qa  # noqa: E402
from facts.synthetic import degrade_slice  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    p.add_argument("--femurs", nargs="*", default=config.FEMURS, help="Femur IDs (default: all)")
    p.add_argument("--sigma", type=float, default=config.SYNTH_SIGMA)
    p.add_argument("--scale", type=int, default=config.SYNTH_SCALE)
    p.add_argument("--axis", type=int, default=config.SLICE_AXIS, help="Slicing axis for the 2D degradation")
    p.add_argument("--workers", type=int, default=4, help="Number of threads")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--qa_dir", default="results/preprocessing", help="Where histogram_match.py saved the mapping")
    p.add_argument("--no_snapshots", action="store_true",
                   help="Do not write snapshot figures to <qa_dir>/snapshots (see scripts/inspect_preprocessing.py)")
    return p.parse_args()


def main():
    args = parse_args()
    for femur in args.femurs:
        name = io.femur_name(femur)
        out_path = io.input_path(args.root, femur, "synth")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{name}: {out_path} exists, skipping (use --overwrite)")
            continue

        print(f"{name}: degrading slices along axis {args.axis}")
        # Read fully: slicing a memory map along axis != 0 is very slow
        hr = io.open_volume(io.target_path(args.root, femur), mmap=False)
        out = np.empty_like(hr)

        def work(i):
            im = degrade_slice(io.get_slice(hr, i, args.axis), sigma=args.sigma, scale=args.scale)
            return i, io.to_dtype(im, hr.dtype)

        with ThreadPoolExecutor(args.workers) as ex:
            for i, im in ex.map(work, range(hr.shape[args.axis])):
                io.set_slice(out, i, im, args.axis)
        io.save_volume(out_path, out)
        print(f"{name}: saved {out_path}")
        if not args.no_snapshots:
            snap_dir = os.path.join(args.qa_dir, "snapshots")
            qa.inspect_femur(args.root, femur, snap_dir, args.qa_dir, axis=args.axis)
            print(f"{name}: snapshots in {snap_dir}")


if __name__ == "__main__":
    main()
