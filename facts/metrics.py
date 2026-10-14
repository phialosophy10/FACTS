"""Image similarity and bone metrics, computed slice-wise inside the bone mask.

All images are expected to be normalised to [0, 1].
"""
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

from . import config


def psnr_masked(hr, sr, mask):
    mse = np.mean((hr[mask] - sr[mask]) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim_masked(hr, sr, mask):
    _, smap = structural_similarity(hr, sr, data_range=1.0, full=True)
    return float(smap[mask].mean())


class LPIPSMetric:
    """LPIPS (AlexNet) averaged over the mask (uses the spatial LPIPS map)."""

    def __init__(self, device):
        import lpips
        self.device = device
        self.net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()

    @torch.inference_mode()
    def __call__(self, hr, sr, mask):
        def t(im):
            return torch.from_numpy(2 * im - 1).float()[None, None].repeat(1, 3, 1, 1).to(self.device)

        dmap = self.net(t(hr), t(sr))[0, 0].cpu().numpy()
        return float(dmap[mask].mean())


def otsu_in_mask(slices, masks):
    return float(threshold_otsu(np.concatenate([s[m] for s, m in zip(slices, masks)])))


def local_thickness_mean(bone):
    """Mean local thickness (diameter, in voxels) of the foreground of a 2D binary image."""
    import localthickness as lt
    if not bone.any():
        return np.nan
    radius = lt.local_thickness(bone)  # returns the radius of the largest inscribed disc
    return float(2 * radius[bone].mean())


def bone_metrics(slices, masks, threshold=None):
    """BV/TV and Tb.Th (um) over a list of 2D slices (binarised with a common threshold)."""
    if threshold is None:
        threshold = otsu_in_mask(slices, masks)
    bv = tv = 0
    thick = []
    for s, m in zip(slices, masks):
        bone = (s > threshold) & m
        bv += bone.sum()
        tv += m.sum()
        thick.append(local_thickness_mean(bone))
    return {"bvtv": float(bv / tv), "tbth_um": float(np.nanmean(thick) * config.VOXEL_SIZE_UM),
            "threshold": threshold}
