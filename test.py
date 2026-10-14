"""Apply a trained generator to the test femurs and evaluate it against the micro-CT.

For every test femur, all 2D slices inside the bone mask are super-resolved
(tiled inference); the SR volume is saved (unless --no_save) and PSNR, SSIM,
LPIPS, BV/TV and Tb.Th are computed on every --eval_stride'th slice, inside
the mask.

Cross-testing: --input selects the input data independently of what the model
was trained on (e.g. trained on synthetic, tested on real).

--baseline evaluates the input itself (no network), i.e. the LR reference.

Examples:
    python test.py --root /path/to/FACTS --run runs/esrgan_real --input real
    python test.py --root /path/to/FACTS --run runs/esrgan_real --input synth
    python test.py --root /path/to/FACTS --baseline --input real
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from facts import config, io
from facts.data import FemurVolume, Normalizer
from facts.metrics import LPIPSMetric, bone_metrics, psnr_masked, ssim_masked
from facts.models import build_generator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Dataset root directory")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run directory containing generator.pt")
    g.add_argument("--baseline", action="store_true", help="Evaluate the input data itself")
    p.add_argument("--input", choices=["real", "synth"], required=True, help="Test input data")
    p.add_argument("--weights", default="generator.pt", help="Generator file inside the run dir")
    p.add_argument("--femurs", nargs="*", default=config.TEST_FEMURS)
    p.add_argument("--out", default=None, help="Output dir (default: <run>/test_<input> or results/baseline/test_<input>)")
    p.add_argument("--eval_stride", type=int, default=10, help="Evaluate every n'th slice")
    p.add_argument("--tile", type=int, default=512, help="Tile size for inference")
    p.add_argument("--overlap", type=int, default=64, help="Tile overlap for inference")
    p.add_argument("--no_save", action="store_true", help="Do not save the SR volumes")
    p.add_argument("--no_lpips", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def sr_slice(G, im, device, tile=512, overlap=64):
    """Super-resolve a normalised 2D image with overlapping tiles."""
    H, W = im.shape
    out = np.zeros((H, W), np.float32)
    step = tile - 2 * overlap
    x = torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32))[None, None].to(device)
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(y0 - overlap, 0), max(x0 - overlap, 0)
            yb, xb = min(y0 + step + overlap, H), min(x0 + step + overlap, W)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = G(x[..., ya:yb, xa:xb]).float()[0, 0].cpu().numpy()
            yc, xc = min(y0 + step, H), min(x0 + step, W)
            out[y0:yc, x0:xc] = pred[y0 - ya:yc - ya, x0 - xa:xc - xa]
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.baseline:
        G, axis = None, config.SLICE_AXIS
        meta = {"model": "Input", "train_data": "-"}
        out_dir = args.out or os.path.join("results", "baseline", f"test_{args.input}")
    else:
        meta = torch.load(os.path.join(args.run, args.weights), map_location=device, weights_only=False)
        G = build_generator(meta["model"], meta["n_blocks"]).to(device).eval()
        G.load_state_dict(meta.pop("G"))
        axis = meta["axis"]
        norm = Normalizer(**meta["normalizer"])
        out_dir = args.out or os.path.join(args.run, f"test_{args.input}")
    os.makedirs(out_dir, exist_ok=True)
    lpips_fn = None if args.no_lpips else LPIPSMetric(device)

    results = []
    for femur in args.femurs:
        t0 = time.time()
        f = FemurVolume(args.root, femur, args.input, axis=axis)
        # Normalisation used for the metrics: fixed per femur from the micro-CT, identical for all methods
        lo, hi = np.percentile(f.hr[::4, ::4, ::4][f.coarse_mask[::4]], (0.5, 99.5))
        mnorm = Normalizer(lo, hi)

        slices = [z for z in range(f.shape[0]) if f.coarse_mask[z].any()]
        eval_idx = set(slices[::args.eval_stride])
        sr = np.zeros(f.shape, f.hr.dtype) if (G is not None and not args.no_save) else None
        per_slice = {"psnr": [], "ssim": [], "lpips": []}
        ev = {"hr": [], "sr": [], "mask": [], "z": []}
        for z in slices:
            lr_z = f.lr[z].astype(np.float32) + f.lr_offset
            if G is None:
                sr_z = lr_z
            elif z in eval_idx or sr is not None:
                sr_z = norm.inverse(sr_slice(G, norm(lr_z), device, args.tile, args.overlap))
            else:
                continue
            if sr is not None:
                sr[z] = io.to_dtype(sr_z, sr.dtype)
            if z in eval_idx:
                m = f.mask[z] > 0
                if not m.any():
                    continue
                hr_n = np.clip(mnorm(f.hr[z]), 0, 1)
                sr_n = np.clip(mnorm(sr_z), 0, 1)
                per_slice["psnr"].append(psnr_masked(hr_n, sr_n, m))
                per_slice["ssim"].append(ssim_masked(hr_n, sr_n, m))
                if lpips_fn is not None:
                    per_slice["lpips"].append(lpips_fn(hr_n, sr_n, m))
                ev["hr"].append(hr_n)
                ev["sr"].append(sr_n)
                ev["mask"].append(m)
                ev["z"].append(z)

        bone_hr = bone_metrics(ev["hr"], ev["mask"])
        bone_sr = bone_metrics(ev["sr"], ev["mask"])
        res = {"femur": f.name, "model": meta["model"], "train_data": meta["train_data"], "test_data": args.input,
               "n_slices": len(ev["hr"])}
        for k, v in per_slice.items():
            if v:
                res[k], res[k + "_std"] = float(np.mean(v)), float(np.std(v))
        res.update({"bvtv": bone_sr["bvtv"], "bvtv_hr": bone_hr["bvtv"], "bvtv_err": bone_sr["bvtv"] - bone_hr["bvtv"],
                    "tbth_um": bone_sr["tbth_um"], "tbth_hr_um": bone_hr["tbth_um"],
                    "tbth_err_um": bone_sr["tbth_um"] - bone_hr["tbth_um"]})
        results.append(res)
        print(json.dumps(res))

        # Example slice (middle of the evaluated slices) for figures
        k = len(ev["hr"]) // 2
        z = ev["z"][k]
        np.savez_compressed(os.path.join(out_dir, f"{f.name}_example.npz"), hr=ev["hr"][k], sr=ev["sr"][k],
                            lr=np.clip(mnorm(f.lr[z].astype(np.float32) + f.lr_offset), 0, 1), mask=ev["mask"][k], slice=z)

        if sr is not None:
            # SR volume on the micro-CT grid and scale; zero outside the bone bounding box
            full = np.zeros(io.open_volume(io.target_path(args.root, femur)).shape, sr.dtype)
            full[f.bbox] = np.moveaxis(sr, 0, axis)
            io.save_volume(os.path.join(out_dir, f"{f.name}.tif"), full)
            del full
        del sr, f
        print(f"{femur}: done in {(time.time() - t0) / 60:.1f} min")

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved metrics to {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
