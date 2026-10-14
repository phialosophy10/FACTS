"""Datasets and intensity normalisation for 2D training on FACTS."""
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config, io

# Coarse grid (voxels) used to find positions inside the bone mask
COARSE = 4


class Normalizer:
    """Linear map of intensities [lo, hi] -> [0, 1] (shared by input and target,
    which are on the same scale after histogram matching)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float32) - self.lo) / (self.hi - self.lo)

    def inverse(self, x):
        return np.asarray(x, dtype=np.float32) * (self.hi - self.lo) + self.lo

    def state_dict(self):
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def from_femurs(cls, femurs, p=(0.5, 99.5)):
        """Percentiles of the (coarsely sub-sampled) input voxels inside the masks."""
        vals = [np.asarray(f.lr[::COARSE, ::COARSE, ::COARSE])[f.coarse_mask[::COARSE]] + f.lr_offset
                for f in femurs]
        lo, hi = np.percentile(np.concatenate(vals), p)
        return cls(lo, hi)


def mask_bbox(mask):
    """Bounding box (tuple of slices) of a mask, computed on a coarse grid."""
    coarse = np.asarray(mask[::COARSE, ::COARSE, ::COARSE]) > 0
    if not coarse.any():
        raise ValueError("Empty mask")
    box = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(coarse.any(axis=other))
        lo = max(idx[0] * COARSE - COARSE, 0)
        hi = min((idx[-1] + 2) * COARSE, mask.shape[ax])
        box.append(slice(lo, hi))
    return tuple(box)


class FemurVolume:
    """Input, target and mask of one femur, cropped to the mask bounding box and
    re-ordered so that 2D slices (along `axis`) are contiguous in memory."""

    def __init__(self, root, femur, data_type, axis=config.SLICE_AXIS, in_memory=True, keep_mask=True):
        self.name = io.femur_name(femur)
        mask = io.open_volume(io.mask_path(root, femur))
        self.bbox = mask_bbox(mask)
        self.axis = axis

        def prep(vol):
            vol = np.moveaxis(vol[self.bbox], axis, 0)
            return np.ascontiguousarray(vol) if in_memory else vol

        self.lr = prep(io.open_volume(io.input_path(root, femur, data_type)))
        # Offset that brings the clinical CT to the scale of the histogram-matched target
        self.lr_offset = io.hm_offset(root, femur) if data_type == "real" else 0.0
        self.hr = prep(io.open_volume(io.target_path(root, femur)))
        mask = prep(mask) if keep_mask else np.moveaxis(mask[self.bbox], axis, 0)
        # Coarse positions inside the mask, used for sampling patch centres
        self.coarse_mask = np.asarray(mask[:, ::COARSE, ::COARSE]) > 0
        self.candidates = np.argwhere(self.coarse_mask)
        self.mask = mask if keep_mask else None
        self.shape = self.hr.shape


class PatchDataset(Dataset):
    """Random 2D patches whose centre lies inside the bone mask.

    The dataset is 'infinite': `length` sets the number of samples per epoch.
    """

    def __init__(self, femurs, normalizer, patch_size=128, length=100_000, augment=True):
        self.femurs = femurs
        self.norm = normalizer
        self.ps = patch_size
        self.length = length
        self.augment = augment
        # Sample femurs in proportion to their mask size
        sizes = np.array([len(f.candidates) for f in femurs], dtype=np.float64)
        self.p = sizes / sizes.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample(np.random.default_rng())

    def sample(self, rng):
        f = self.femurs[rng.choice(len(self.femurs), p=self.p)]
        z, y, x = f.candidates[rng.integers(len(f.candidates))]
        y = y * COARSE + rng.integers(COARSE)
        x = x * COARSE + rng.integers(COARSE)
        ps, (_, H, W) = self.ps, f.shape
        y0 = int(np.clip(y - ps // 2, 0, max(H - ps, 0)))
        x0 = int(np.clip(x - ps // 2, 0, max(W - ps, 0)))
        lr = pad_to(self.norm(f.lr[z, y0:y0 + ps, x0:x0 + ps] + f.lr_offset), ps)
        hr = pad_to(self.norm(f.hr[z, y0:y0 + ps, x0:x0 + ps]), ps)
        if self.augment:
            k = rng.integers(4)
            lr, hr = np.rot90(lr, k), np.rot90(hr, k)
            if rng.random() < 0.5:
                lr, hr = lr[:, ::-1], hr[:, ::-1]
        return {"lr": torch.from_numpy(lr.copy())[None], "hr": torch.from_numpy(hr.copy())[None]}


def pad_to(im, size):
    """Zero-pad a 2D image at the bottom/right to (at least) size x size."""
    ph, pw = max(size - im.shape[0], 0), max(size - im.shape[1], 0)
    if ph or pw:
        im = np.pad(im, ((0, ph), (0, pw)))
    return im


def fixed_patches(femurs, normalizer, n, patch_size=128, seed=0):
    """A fixed (reproducible) batch of patches, e.g. for monitoring during training."""
    ds = PatchDataset(femurs, normalizer, patch_size, augment=False)
    rng = np.random.default_rng(seed)
    batch = [ds.sample(rng) for _ in range(n)]
    return {k: torch.stack([b[k] for b in batch]) for k in ("lr", "hr")}
