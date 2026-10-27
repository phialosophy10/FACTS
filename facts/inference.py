"""Tiled inference with a trained generator, and loading of trained generators."""
import os

import numpy as np
import torch

from .data import Normalizer
from .models import build_generator


def load_generator(run_dir, device, weights="generator.pt"):
    """Generator of a training run (eval mode), its normaliser and its metadata."""
    meta = torch.load(os.path.join(run_dir, weights), map_location=device, weights_only=False)
    G = build_generator(meta["model"], meta["n_blocks"]).to(device).eval()
    G.load_state_dict(meta.pop("G"))
    return G, Normalizer(**meta["normalizer"]), meta


@torch.inference_mode()
def sr_batch(G, ims, device, tile=512, overlap=64):
    """Super-resolve a stack of normalised 2D images (B, H, W) with overlapping tiles."""
    ims = np.asarray(ims, dtype=np.float32)
    B, H, W = ims.shape
    out = np.zeros((B, H, W), np.float32)
    step = tile - 2 * overlap
    x = torch.from_numpy(np.ascontiguousarray(ims))[:, None].to(device)
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(y0 - overlap, 0), max(x0 - overlap, 0)
            yb, xb = min(y0 + step + overlap, H), min(x0 + step + overlap, W)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = G(x[..., ya:yb, xa:xb]).float()[:, 0].cpu().numpy()
            yc, xc = min(y0 + step, H), min(x0 + step, W)
            out[:, y0:yc, x0:xc] = pred[:, y0 - ya:yc - ya, x0 - xa:xc - xa]
    return out


def sr_slice(G, im, device, tile=512, overlap=64):
    """Super-resolve a single normalised 2D image with overlapping tiles."""
    return sr_batch(G, np.asarray(im)[None], device, tile, overlap)[0]
