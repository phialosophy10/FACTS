"""Synthetic low-resolution degradation (2D, slice-wise)."""
import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize

from . import config


def degrade_slice(im, sigma=config.SYNTH_SIGMA, scale=config.SYNTH_SCALE):
    """Gaussian blur -> downsample by `scale` -> upsample back with linear interpolation.

    Returns a float32 image with the same shape as `im`.
    """
    shape = im.shape
    im = gaussian(im.astype(np.float32), sigma=sigma, preserve_range=True)
    small_shape = tuple(int(np.ceil(s / scale)) for s in shape)
    im = resize(im, small_shape, order=1, anti_aliasing=False, preserve_range=True)
    im = resize(im, shape, order=1, anti_aliasing=False, preserve_range=True)
    return im.astype(np.float32)
