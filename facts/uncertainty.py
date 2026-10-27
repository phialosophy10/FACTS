"""Input-noise sensitivity as an uncertainty measure, along an interpolation between
synthetic and real low-resolution input.

For a blending coefficient alpha the input is

    x_alpha = (1 - alpha) * synthetic + alpha * real          (same intensity scale)

Noise is added in two parts:
  * equalisation noise: one fixed draw per alpha, bringing the noise level of x_alpha up to
    that of the real clinical CT (the blend is less noisy than the real scan for alpha < 1);
    so every alpha has the same total noise level and only the image structure changes;
  * perturbation noise: a fresh draw for each of N repetitions, with the same strength for
    every alpha. The per-voxel spread of the N outputs is the uncertainty.

All noise is white Gaussian (a simplification: CT noise is spatially correlated).
"""
import numpy as np


def robust_noise_sigma(slices, masks):
    """Noise standard deviation from neighbouring-voxel differences inside the mask,
    estimated robustly (median absolute deviation), so edges contribute little."""
    diffs = []
    for im, m in zip(slices, masks):
        im = np.asarray(im, dtype=np.float32)
        both = m[:, 1:] & m[:, :-1]
        diffs.append(np.diff(im, axis=1)[both])
    d = np.concatenate(diffs)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2))


def noise_levels(alpha, sigma_real, sigma_synth, perturb_sigma, mode="topup"):
    """(equalisation sigma, perturbation sigma) for one alpha. mode 'fixed': no equalisation."""
    if mode == "fixed":
        return 0.0, perturb_sigma
    blend_var = ((1 - alpha) * sigma_synth) ** 2 + (alpha * sigma_real) ** 2
    return float(np.sqrt(max(sigma_real ** 2 - blend_var, 0.0))), perturb_sigma


def make_inputs(synth, real, alpha, sigma_eq, sigma_p, n, rng_eq, rng_p):
    """N noisy versions (n, H, W) of the blended input, in raw intensity units."""
    x = (1 - alpha) * synth.astype(np.float32) + alpha * real.astype(np.float32)
    if sigma_eq > 0:
        x = x + rng_eq.normal(0, sigma_eq, x.shape).astype(np.float32)
    return x[None] + rng_p.normal(0, sigma_p, (n, *x.shape)).astype(np.float32)


def sparsification(unc, err, fractions=np.linspace(0, 0.9, 19)):
    """Mean remaining error after removing the most uncertain voxels (by unc) and, as the
    oracle, the voxels with the largest error. Both curves are divided by the mean error.

    Returns fractions, curve by uncertainty, oracle curve and AUSE (area between them;
    0 = uncertainty ranks voxels exactly like the error does).
    """
    unc, err = np.asarray(unc), np.asarray(err)
    n = len(err)
    by_unc = err[np.argsort(-unc)]
    by_err = np.sort(err)[::-1]
    mean = err.mean() if err.mean() > 0 else 1.0
    c_unc = np.array([by_unc[int(round(f * n)):].mean() for f in fractions]) / mean
    c_orc = np.array([by_err[int(round(f * n)):].mean() for f in fractions]) / mean
    ause = float(np.trapezoid(c_unc - c_orc, fractions) if hasattr(np, "trapezoid")
                 else np.trapz(c_unc - c_orc, fractions))
    return fractions, c_unc, c_orc, ause
