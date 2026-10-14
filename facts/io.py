"""Paths and volume I/O for the FACTS dataset.

Folder layout:  <root>/f_NNN/<MOD>/f_NNN[_suffix].tif
with MOD in {HR, LR, SY, MS}.
"""
import json
import os
import warnings

import numpy as np
import tifffile

from . import config


def femur_name(femur):
    """'1', '001' or 'f_001' -> 'f_001'."""
    femur = str(femur)
    if femur.startswith("f_"):
        return femur
    return "f_" + femur.zfill(3)


def get_path(root, femur, mod, suffix="", ext="tif"):
    """Full path of a volume, e.g. get_path(root, '001', 'HR', 'hm')."""
    name = femur_name(femur)
    suffix = "_" + suffix if suffix else ""
    return os.path.join(root, name, mod, f"{name}{suffix}.{ext}")


def input_path(root, femur, data_type):
    mod, suffix = config.INPUT_KEYS[data_type]
    return get_path(root, femur, mod, suffix)


def target_path(root, femur):
    return get_path(root, femur, *config.TARGET_KEY)


def mask_path(root, femur):
    return get_path(root, femur, *config.MASK_KEY)


def hm_offset(root, femur):
    """Offset added to the intensities by scripts/histogram_match.py (0 if none).

    The same offset must be added to the clinical CT so that input and target
    share one intensity scale.
    """
    path = get_path(root, femur, *config.TARGET_KEY, ext="json")
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        return float(json.load(f).get("offset", 0.0))


def open_volume(path, mmap=True):
    """Open a TIFF volume, memory-mapped if possible.

    Memory-mapping only works for uncompressed, contiguous TIFFs. If that
    fails the whole volume is read into memory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if mmap:
        try:
            return tifffile.memmap(path, mode="r")
        except (ValueError, OSError):
            warnings.warn(f"Could not memory-map {path}; reading into memory.")
    return tifffile.imread(path)


def create_volume(path, shape, dtype):
    """Create an (uncompressed, memory-mappable) TIFF volume for writing slice by slice."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return tifffile.memmap(path, shape=tuple(shape), dtype=dtype, bigtiff=True)


def save_volume(path, vol):
    """Save a volume as an uncompressed (memory-mappable) TIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, vol, bigtiff=True)


def get_slice(vol, idx, axis=config.SLICE_AXIS):
    """2D slice idx of vol along axis (as a contiguous array)."""
    return np.ascontiguousarray(np.take(vol, idx, axis=axis))


def set_slice(vol, idx, im, axis=config.SLICE_AXIS):
    sl = [slice(None)] * vol.ndim
    sl[axis] = idx
    vol[tuple(sl)] = im


def to_dtype(arr, dtype):
    """Round (for integer types) and clip arr to the range of dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(np.rint(arr), info.min, info.max)
    return arr.astype(dtype)
