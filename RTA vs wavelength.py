"""
rta_vs_wavelength.py
--------------------
Reflectance (R), transmission (T) and absorbance (A) of a registered
coverslip over wavelength x angle.

Differences from CoverslipRegistry.plot_transmission_vs_wavelength:
  * n(lambda) and k(lambda) are recomputed at each wavelength
    (dispersion and layer loss are included).
  * R, T and A come from the same TMM call, so R + T + A = 1.
  * The TIR critical angle is marked on each heatmap.

A = 1 - R - T. With a lossless sample medium, A is the power absorbed
inside the stack layers. Above the critical angle T = 0, so A = 1 - R.

Run this file directly in the VS Code debugger (F5).
Edit the settings in the __main__ block at the bottom.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coverslip_registry import CoverslipRegistry
from materials import nk_measured, excitation_coeff


# ---------------------------------------------------------------------------
# Compute engine
# ---------------------------------------------------------------------------

def _n_complex_at(registry: CoverslipRegistry, materials: list[str], wl: float) -> list:
    """Complex index profile (n + ik) of the full stack at one wavelength."""
    return (
        [complex(registry.n_super, 0.0)]
        + [complex(nk_measured(wl, m), excitation_coeff(wl, m)) for m in materials]
        + [complex(registry.n_subs, 0.0)]
    )


def compute_rta_grid(
    registry: CoverslipRegistry,
    name: str,
    wavelengths: np.ndarray,
    angles: np.ndarray,
    s_weight: float = 1.0,
    p_weight: float = 0.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R, T, A grids with shape (len(wavelengths), len(angles))."""
    if name not in registry.registry:
        raise ValueError(f"No coverslip named '{name}'.")
    if s_weight == 0.0 and p_weight == 0.0:
        raise ValueError("At least one of s_weight or p_weight must be > 0.")

    e_prof = registry.registry[name][0]
    materials = registry.registry[name][4]

    wavelengths = np.asarray(wavelengths, dtype=float)
    angles = np.asarray(angles, dtype=float)

    R = np.zeros((len(wavelengths), len(angles)))
    T = np.zeros_like(R)
    A = np.zeros_like(R)

    for i, wl in enumerate(wavelengths):
        n_complex = _n_complex_at(registry, materials, wl)
        R[i], T[i], A[i] = registry._calc_rta(
            n_complex, e_prof, wl, angles, s_weight, p_weight
        )
        if verbose and (i % 50 == 0 or i == len(wavelengths) - 1):
            print(f"  [{name}] lambda = {wl:.1f} nm  ({i + 1}/{len(wavelengths)})")

    return R, T, A


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_LABELS = {"R": "Reflectance R", "T": "Transmission T", "A": "Absorbance A"}


def plot_rta_vs_wavelength(
    registry: CoverslipRegistry,
    name: str,
    wavelength_range: tuple[float, float] = (400, 800),
    num: int = 300,
    heatmap: bool = True,
    angle_range: tuple[float, float] = (0, 85),
    angle_num: int = 171,
    theta: list[float] | float = 0,
    quantities: tuple[str, ...] = ("R", "T", "A"),
    cmap: str = "jet",
    wavelength_min: Optional[float] = None,
    wavelength_max: Optional[float] = None,
    export_excel: bool = False,
    savefig: Optional[str] = None,
    s_weight: float = 1.0,
    p_weight: float = 0.0,
) -> dict:
    """Plot R, T and/or A vs wavelength.

    Heatmap mode (heatmap=True)
        One panel per quantity. X = angle, Y = wavelength, colour = value.

    Line mode (heatmap=False)
        One panel per quantity. One curve per angle in *theta*.

    Parameters
    ----------
    registry : CoverslipRegistry
    name : str
        Registry key.
    wavelength_range : (min_nm, max_nm)
    num : int
        Number of wavelength samples.
    heatmap : bool
        Heatmap (True) or line plot (False).
    angle_range : (min_deg, max_deg)
        Angle axis for heatmap mode.
    angle_num : int
        Number of angle samples for heatmap mode.
    theta : float or list of float
        Angle(s) in degrees for line mode.
    quantities : tuple of {'R', 'T', 'A'}
        Which quantities to plot.
    cmap : str
        Colormap for heatmap mode.
    wavelength_min, wavelength_max : float, optional
        Sub-range for the mean-value printout.
    export_excel : bool
        Export one sheet per quantity.
    savefig : str, optional
        Output figure path. Format comes from the extension.
    s_weight, p_weight : float
        Polarisation weights. Normalised internally.

    Returns
    -------
    dict with keys 'wavelengths', 'angles', 'R', 'T', 'A'.
    """
    for q in quantities:
        if q not in _LABELS:
            raise ValueError(f"Unknown quantity {q!r}. Use 'R', 'T' or 'A'.")

    pol_lbl = registry._pol_label(s_weight, p_weight)
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num)

    if heatmap:
        angles = np.linspace(angle_range[0], angle_range[1], angle_num)
    else:
        angles = np.atleast_1d(np.asarray(theta, dtype=float))

    R, T, A = compute_rta_grid(registry, name, wavelengths, angles, s_weight, p_weight)
    grids = {"R": R, "T": T, "A": A}

    theta_c = registry._critical_angle_deg(registry.n_super, registry.n_subs)

    # ---- Plot ----
    n_panels = len(quantities)

    if heatmap:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, q in zip(axes, quantities):
            grid = grids[q]
            if q == "A":
                vmin, vmax = 0.0, max(float(np.nanmax(grid)), 1e-6)
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.pcolormesh(angles, wavelengths, grid, shading="auto",
                               cmap=cmap, vmin=vmin, vmax=vmax)
            if theta_c is not None and angles[0] <= theta_c <= angles[-1]:
                ax.axvline(theta_c, color="white", lw=1.2, ls="--")
                ax.text(theta_c, wavelengths[-1], f" θc={theta_c:.1f}°",
                        color="white", va="top", ha="left", fontsize=10)
            ax.set_xlabel("Incidence Angle (°)")
            ax.set_title(_LABELS[q])
            plt.colorbar(im, ax=ax, label=q)
        axes[0].set_ylabel("Wavelength (nm)")
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, q in zip(axes, quantities):
            for j, ang in enumerate(angles):
                ax.plot(wavelengths, grids[q][:, j], lw=2, label=f"{ang:g}°")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel(q)
            ax.set_title(_LABELS[q])
            if q != "A":
                ax.set_ylim(-0.02, 1.02)
            ax.grid(True)
            ax.legend(fontsize=9)

    fig.suptitle(
        f"{name}: {', '.join(quantities)} vs wavelength  ({pol_lbl}-pol, "
        f"n_super={registry.n_super}, n_subs={registry.n_subs})"
    )
    plt.tight_layout()

    if savefig:
        fmt = savefig.rsplit(".", 1)[-1] if "." in savefig else "png"
        plt.savefig(savefig, format=fmt, dpi=180, bbox_inches="tight")
        print(f"Figure saved to '{savefig}'.")

    plt.show()

    # ---- Mean values in a wavelength window ----
    idx_a = 0 if wavelength_min is None else int(np.abs(wavelengths - wavelength_min).argmin())
    idx_b = (len(wavelengths) - 1 if wavelength_max is None
             else int(np.abs(wavelengths - wavelength_max).argmin()))
    print(f"\nMean values, {wavelengths[idx_a]:.0f}-{wavelengths[idx_b]:.0f} nm, "
          f"all plotted angles:")
    for q in quantities:
        print(f"  {q}: {np.nanmean(grids[q][idx_a:idx_b + 1, :]):.4f}")

    # ---- Excel export ----
    if export_excel:
        mode = "heatmap" if heatmap else "lines"
        outname = (
            f"{name}_RTA_{mode}_{pol_lbl}_"
            f"{wavelength_range[0]:.0f}to{wavelength_range[1]:.0f}nm_"
            f"{angles[0]:.0f}to{angles[-1]:.0f}deg.xlsx"
        )
        with pd.ExcelWriter(outname) as xw:
            for q in quantities:
                df = pd.DataFrame(grids[q],
                                  index=np.round(wavelengths, 4),
                                  columns=np.round(angles, 4))
                df.index.name = "Wavelength_nm"
                df.columns.name = "Angle_deg"
                df.to_excel(xw, sheet_name=q)
        print(f"Data exported to '{outname}'.")

    return {"wavelengths": wavelengths, "angles": angles, "R": R, "T": T, "A": A}


# ---------------------------------------------------------------------------
# Run in the VS Code debugger (F5)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from run import build_registry

    reg = build_registry(wavelength=561, divergence=2, n_super=1.523, n_subs=1.35)

    # Heatmaps of R, T, A over the full angle range (past the critical angle)
    out = plot_rta_vs_wavelength(
        reg, "MAAPS488nm",
        wavelength_range=(400, 700), num=400,
        angle_range=(0, 85), angle_num=171,
        s_weight=0.5, p_weight=0.5,
        export_excel=False,
    )

        # Heatmaps of R, T, A over the full angle range (past the critical angle)
    out = plot_rta_vs_wavelength(
        reg, "MAAPS488nm_inverted",
        wavelength_range=(400, 700), num=400,
        angle_range=(0, 85), angle_num=171,
        s_weight=0.5, p_weight=0.5,
        export_excel=False,
    )

        # Heatmaps of R, T, A over the full angle range (past the critical angle)
    out = plot_rta_vs_wavelength(
        reg, "Fabien_optimized_P_1",
        wavelength_range=(400, 700), num=400,
        angle_range=(0, 85), angle_num=171,
        s_weight=0.5, p_weight=0.5,
        export_excel=False,
    )

        # Heatmaps of R, T, A over the full angle range (past the critical angle)
    out = plot_rta_vs_wavelength(
        reg, "Fabien_optimized_P_1_inv",
        wavelength_range=(400, 700), num=400,
        angle_range=(0, 85), angle_num=171,
        s_weight=0.5, p_weight=0.5,
        export_excel=False,
    )


    plt.close("all")