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
