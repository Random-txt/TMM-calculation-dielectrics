"""
utils.py
--------
Beam-divergence modelling utilities.

The laser beam incident on the TIRF/dielectric multilayer coverslip has
a finite angular spread (divergence).  We model this spread with a
*super-Gaussian* kernel:

    w(Δθ) ∝ exp( −|Δθ/σ|^p ) · exp( −skew · Δθ )

where
  * Δθ  = deviation from the nominal incidence angle (degrees)
  * σ   = standard-deviation-like width, derived from the FWHM
  * p   = super-Gaussian exponent (p = 2 → ordinary Gaussian;
           p > 2 → flatter top and steeper edges)
  * skew = optional asymmetry parameter (0 → symmetric kernel)

Helper
------
fwhm_to_sigma(fwhm_deg) → sigma_deg
    Convert a FWHM in degrees to the σ parameter used by the
    super-Gaussian kernel.
"""

import numpy as np


def super_gaussian_weight(
    delta_angles: np.ndarray,
    sigma: float,
    exponent: int = 4,
    skew: float = 0.0,
) -> np.ndarray:
    """Compute normalised super-Gaussian weights for an array of angle offsets.

    Parameters
    ----------
    delta_angles : array_like
        Angular offsets from the nominal incidence angle, in degrees.
    sigma : float
        Width parameter (σ).  Use :func:`fwhm_to_sigma` to convert FWHM.
    exponent : int, optional
        Super-Gaussian order.  ``exponent=2`` reproduces an ordinary
        Gaussian; higher values give a flatter top and steeper shoulders.
    skew : float, optional
        Asymmetric tilt applied via ``exp(−skew · Δθ)``.
        ``0.0`` (default) → symmetric kernel.

    Returns
    -------
    numpy.ndarray
        Normalised weight array (sums to 1).
    """
    delta_angles = np.asarray(delta_angles, dtype=float)

    # Core super-Gaussian envelope
    weights = np.exp(-np.abs(delta_angles / sigma) ** exponent)

    # Optional asymmetry term
    if skew != 0.0:
        weights *= np.exp(-skew * delta_angles)

    # Normalise so weights sum to 1 (incoherent intensity average)
    total = np.sum(weights)
    if total == 0.0:
        raise ValueError("All super-Gaussian weights are zero — check sigma/exponent values.")
    return weights / total


def fwhm_to_sigma(fwhm_deg: float) -> float:
    """Convert a full-width-at-half-maximum (FWHM) to the σ parameter.

    For a standard Gaussian the relation is σ = FWHM / (2√(2 ln 2)).
    The same conversion is used here regardless of the super-Gaussian
    exponent; the exponent then reshapes the kernel while σ still
    controls the overall angular width.

    Parameters
    ----------
    fwhm_deg : float
        Full-width at half-maximum of the beam divergence, in degrees.

    Returns
    -------
    float
        Corresponding σ in degrees.
    """
    return fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))