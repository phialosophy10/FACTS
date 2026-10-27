"""Re-upload the example slices of finished test runs to their Weights & Biases runs.

test.py saves an example slice per test femur (<test dir>/f_NNN_example.npz) and the
wandb run id (<test dir>/wandb_id.txt). This script logs those examples again to the same
wandb runs, e.g. after a fix of how the images are logged. No inference is repeated.

Example:
    python scripts/relog_wandb_examples.py
    python scripts/relog_wandb_examples.py --dirs runs/esrgan_real/test_real
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from facts.tracking import add_wandb_args, example_image  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="*", default=None,
                   help="Test directories (default: runs/*/test_* and results/baseline/test_*)")
    add_wandb_args(p)
    args = p.parse_args()
    dirs = args.dirs or sorted(glob.glob("runs/*/test_*") + glob.glob("results/baseline/test_*"))

    import wandb
    for d in dirs:
        id_file = os.path.join(d, "wandb_id.txt")
        examples = sorted(glob.glob(os.path.join(d, "f_*_example.npz")))
        if not os.path.exists(id_file) or not examples:
            print(f"{d}: no wandb run or no examples, skipped")
            continue
        with open(id_file) as f:
            run_id = f.read().strip()
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id, resume="allow", dir=d)
        for path in examples:
            ex = np.load(path)
            name = os.path.basename(path)[:5]
            run.log({f"examples/{name}": example_image(ex["lr"], ex["sr"], ex["hr"], ex["mask"],
                                                       f"{name}, slice {int(ex['slice'])}: input | SR | micro-CT")})
        run.finish()
        print(f"{d}: re-logged {len(examples)} examples to wandb run {run_id}")


if __name__ == "__main__":
    main()
