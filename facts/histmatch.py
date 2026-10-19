"""Masked histogram matching of (large) volumes, processed slice by slice."""
import numpy as np

N_BINS = 2 ** 16


def masked_range(vol, mask):
    """Min and max of vol inside mask."""
    vmin, vmax = np.inf, -np.inf
    for i in range(vol.shape[0]):
        m = mask[i] > 0
        if m.any():
            v = vol[i][m]
            vmin, vmax = min(vmin, v.min()), max(vmax, v.max())
    if not np.isfinite(vmin):
        raise ValueError("Mask is empty.")
    return float(vmin), float(vmax)


def masked_histogram(vol, mask, vrange, n_bins=N_BINS):
    """Histogram of vol inside mask with n_bins bins over vrange."""
    hist = np.zeros(n_bins, dtype=np.int64)
    for i in range(vol.shape[0]):
        m = mask[i] > 0
        if m.any():
            hist += np.histogram(vol[i][m], bins=n_bins, range=vrange)[0]
    edges = np.linspace(vrange[0], vrange[1], n_bins + 1)
    return hist, edges


def matching_lut(src_hist, src_edges, ref_hist, ref_edges):
    """Monotone mapping (src value -> ref value) that matches the CDFs.

    Returns (x, y) such that np.interp(v, x, y) maps a source intensity v
    to the reference scale.
    """
    src_cdf = np.cumsum(src_hist) / src_hist.sum()
    ref_cdf = np.cumsum(ref_hist) / ref_hist.sum()
    # Source bin centres and their quantiles
    x = 0.5 * (src_edges[:-1] + src_edges[1:])
    q = src_cdf - 0.5 * src_hist / src_hist.sum()
    # Inverse reference CDF (evaluated at right bin edges)
    keep = np.r_[True, np.diff(ref_cdf) > 0]
    y = np.interp(q, ref_cdf[keep], ref_edges[1:][keep])
    return x, y


def compute_matching(src, ref, mask, n_bins=N_BINS):
    """Compute the LUT that brings src (e.g. HR) to the scale of ref (e.g. LR), inside mask."""
    src_range = masked_range(src, mask)
    ref_range = masked_range(ref, mask)
    src_hist, src_edges = masked_histogram(src, mask, src_range, n_bins)
    ref_hist, ref_edges = masked_histogram(ref, mask, ref_range, n_bins)
    x, y = matching_lut(src_hist, src_edges, ref_hist, ref_edges)
    stats = dict(src_range=src_range, ref_range=ref_range,
                 src_hist=src_hist, src_edges=src_edges,
                 ref_hist=ref_hist, ref_edges=ref_edges)
    return (x, y), stats


def apply_lut(im, lut):
    x, y = lut
    return np.interp(im.astype(np.float64), x, y)


# ------------------------------------------------------------------ linear calibration ---
def _block_starts(mask, block, margin, n_blocks, rng, step=8):
    """Random block corners whose central voxel is inside the mask (searched on a coarse grid)."""
    coarse = np.asarray(mask[::step, ::step, ::step]) > 0
    cand = np.argwhere(coarse) * step
    lo = margin
    hi = np.array(mask.shape) - block - margin
    if len(cand) == 0 or np.any(hi < lo):
        raise ValueError("Empty mask or volume too small for the requested block size")
    cand = np.unique(np.clip(cand - block // 2, lo, hi), axis=0)
    return cand[rng.choice(len(cand), size=min(n_blocks, len(cand)), replace=False)]


def fit_linear(src, ref, mask, sigmas=(0, 1, 2, 3, 4, 6, 8, 12), n_blocks=12, block=96, seed=0):
    """Fit ref ~ a * G_sigma(src) + b inside the mask (least squares).

    The clinical CT (ref) is a blurred version of the micro-CT (src) on (roughly) a linear
    intensity scale, so the micro-CT is first blurred with a Gaussian of width sigma
    (voxels; the best sigma is chosen by R^2) before fitting. The fit is done on random
    blocks inside the mask, so the whole volume never has to be blurred.

    Returns a, b, best sigma and a list of (sigma, a, b, r2) for all sigmas tried.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    # Keep only blur widths whose margin fits in the volume, and shrink blocks for small volumes
    sigmas = [s for s in sigmas if 2 * int(np.ceil(3 * s)) + 16 <= min(mask.shape)] or [0]
    margin = int(np.ceil(3 * max(sigmas)))
    block = min(block, min(mask.shape) - 2 * margin)
    starts = _block_starts(mask, block, margin, n_blocks, rng)
    blocks = []
    for z, y, x in starts:
        sl = tuple(slice(s - margin, s + block + margin) for s in (z, y, x))
        inner = tuple(slice(margin, margin + block) for _ in range(3))
        blocks.append((np.asarray(src[sl], dtype=np.float32), np.asarray(ref[sl], dtype=np.float32)[inner],
                       np.asarray(mask[sl])[inner] > 0, inner))

    results = []
    for sigma in sigmas:
        xs, ys = [], []
        for s_blk, r_blk, m_blk, inner in blocks:
            g = gaussian_filter(s_blk, sigma) if sigma > 0 else s_blk
            xs.append(g[inner][m_blk])
            ys.append(r_blk[m_blk])
        x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
        a, b = np.polyfit(x, y, 1)
        r2 = 1 - np.var(y - (a * x + b)) / np.var(y)
        results.append((float(sigma), float(a), float(b), float(r2)))
    best = max(results, key=lambda r: r[3])
    if best[1] <= 0:
        raise ValueError(f"Linear fit gave a non-positive slope (a = {best[1]:.3g}); check registration/masks")
    return best[1], best[2], best[0], results
