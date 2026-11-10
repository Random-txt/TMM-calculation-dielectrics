"""
coverslip_registry.py
---------------------
Central class for defining, registering, and analysing dielectric
multilayer coverslips used in TIRF / MAAPS experiments.

The class wraps the `tmm` transfer-matrix library and exposes:
  * Angular intensity curves (plane-wave and divergence-averaged)
  * Optical intensity depth profiles
  * Reflectance / Transmission / Absorbance vs angle
  * Transmission vs total optical thickness (thickness sweep)
  * Transmission heatmaps (wavelength × angle)
  * Effective evanescent penetration depth estimation
  * Excitation-wavelength × angle intensity heatmaps

Typical usage
-------------
    registry = CoverslipRegistry(wavelength=488, pol='s')
    registry.add_coverslip('MAAPS488', thicknesses=[...], materials=[...])
    registry.plot_coverslips(['MAAPS488'])
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import inf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tmm import coh_tmm, position_resolved, find_in_structure_with_inf
import os

from materials import nk_measured, excitation_coeff
from utils import super_gaussian_weight, fwhm_to_sigma

# ---------------------------------------------------------------------------
# Global plot aesthetics
# ---------------------------------------------------------------------------
sns.set_theme(style="ticks", context="talk")


class CoverslipRegistry:
    """Registry and analysis engine for dielectric multilayer coverslips.

    Parameters
    ----------
    wavelength : float
        Excitation wavelength in nm used for all Fresnel calculations.
    n_super : float
        Refractive index of the superstrate (objective-side immersion
        medium, typically glass: 1.518).
    n_subs : float
        Refractive index of the substrate (sample medium, typically
        aqueous: ~1.33–1.35).
    div_at_zero : float
        Beam divergence FWHM (°) used at 0° incidence.  For a
        single fixed divergence set ``div_at_zero == max_div``.
    max_div : float
        Beam divergence FWHM (°) at 85° incidence (linearly
        interpolated between ``div_at_zero`` and ``max_div``).
    exponent : int
        Super-Gaussian exponent for the divergence kernel.
    skew : float
        Asymmetry parameter for the divergence kernel (0 → symmetric).
    num_samples : int
        Number of sub-angle samples used in divergence averaging.
    pol : {'s', 'p'}
        Default polarisation for all computations.
    """

    def __init__(
        self,
        wavelength: float = 488,
        n_super: float = 1.518,
        n_subs: float = 1.3512,
        div_at_zero: float = 1.0,
        max_div: float = 1.0,
        exponent: int = 4,
        skew: float = 0.0,
        num_samples: int = 100,
        pol: str = "s",
    ):
        self.wavelength  = wavelength
        self.n_super     = n_super
        self.n_subs      = n_subs
        self.div_at_zero = div_at_zero
        self.max_div     = max_div
        self.exponent    = exponent
        self.skew        = skew
        self.num_samples = num_samples
        self.polarization = pol

        # Stores coverslip definitions keyed by name:
        # name -> (e_profile, n_profile, k_profile, thicknesses, materials)
        self.registry: dict[str, tuple] = {}

    # -----------------------------------------------------------------------
    # Coverslip registration
    # -----------------------------------------------------------------------

    def add_coverslip(
        self,
        name: str,
        thicknesses: list[float],
        materials: list[str],
    ) -> None:
        """Register a multilayer coverslip definition.

        The superstrate and substrate semi-infinite layers are added
        automatically; only the finite intermediate layers need to be
        specified.

        Parameters
        ----------
        name : str
            Unique identifier for this coverslip.
        thicknesses : list of float
            Physical thickness of each layer in nm (ordered from
            superstrate to substrate).
        materials : list of str
            Material key for each layer (must match an entry in
            :mod:`materials`).  Must have the same length as
            ``thicknesses``.
        """
        if len(thicknesses) != len(materials):
            raise ValueError(
                f"'thicknesses' ({len(thicknesses)}) and 'materials' "
                f"({len(materials)}) must have the same length."
            )

        # Build TMM input arrays: semi-infinite layers represented by inf
        e_profile = [inf] + list(thicknesses) + [inf]
        n_profile = (
            [self.n_super]
            + [nk_measured(self.wavelength, m) for m in materials]
            + [self.n_subs]
        )
        k_profile = (
            [0.0]
            + [excitation_coeff(self.wavelength, m) for m in materials]
            + [0.0]
        )

        self.registry[name] = (e_profile, n_profile, k_profile, thicknesses, materials)

    # -----------------------------------------------------------------------
    # Core intensity computations
    # -----------------------------------------------------------------------

    def refractive_index_list(self, materials: list[str]) -> list[float]:
        """Helper to convert material keys to refractive indices."""
        return [nk_measured(self.wavelength, m) for m in materials]
    
    def imaginary_index_list(self, materials: list[str]) -> list[float]:
        """Helper to convert material keys to extinction coefficients."""
        return [excitation_coeff(self.wavelength, m) for m in materials]

    def compute_nominal_intensity(
        self,
        e_profile: list,
        n_profile: list,
        angles: np.ndarray,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute plane-wave (zero-divergence) electric intensity vs angle.

        The intensity is evaluated at the bottom of the last finite layer
        (i.e. just at the stack / sample interface) using the TMM
        `position_resolved` helper.

        Only polarisations with weight > 0 are computed.  The final
        intensity is the normalised weighted sum of contributions from each
        active polarisation.

        Parameters
        ----------
        e_profile, n_profile : list
            TMM thickness and refractive-index profiles (including inf caps).
        angles : array_like
            Incidence angles in degrees.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude it.

        Returns
        -------
        angles : numpy.ndarray
        intensities : numpy.ndarray
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")

        norm = s_weight + p_weight
        angles = np.asarray(angles)
        I_vals = np.zeros(len(angles))

        total_thick = sum(e_profile[1:-1])
        layer_idx, dpos = find_in_structure_with_inf(e_profile, total_thick)

        for pol, weight in (("s", s_weight), ("p", p_weight)):
            if weight == 0.0:
                continue
            for j, theta_deg in enumerate(angles):
                theta_rad = np.deg2rad(theta_deg)
                coh  = coh_tmm(pol, n_profile, e_profile, theta_rad, self.wavelength)
                data = position_resolved(layer_idx, dpos, coh)
                if pol == "s":
                    I_vals[j] += weight * float(np.abs(data["Ey"]) ** 2)
                else:
                    I_vals[j] += weight * float(np.abs(data["Ex"]) ** 2 + np.abs(data["Ez"]) ** 2)

        return angles, I_vals / norm

    def compute_electric_field(
        self,
        e_profile: list,
        n_profile: list,
        angles: np.ndarray,
        pol: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute electric field amplitude squared vs angle.

        Parameters
        ----------
        pol : {'s', 'p'}, optional
            Polarisation to evaluate.  Defaults to ``self.polarization``.
            s-pol returns |E_y|²; p-pol returns |E_x|² + |E_z|².

        Returns
        -------
        angles : numpy.ndarray
        E_squared : numpy.ndarray
        """
        pol = pol if pol is not None else self.polarization
        E_vals = []
        total_thick = sum(e_profile[1:-1])
        layer_idx, dpos = find_in_structure_with_inf(e_profile, total_thick)

        for theta_deg in angles:
            theta_rad = np.deg2rad(theta_deg)
            coh  = coh_tmm(pol, n_profile, e_profile, theta_rad, self.wavelength)
            data = position_resolved(layer_idx, dpos, coh)
            if pol == "s":
                E_vals.append(data["Ey"])
            else:
                E_vals.append(np.sqrt(data["Ex"] ** 2 + data["Ez"] ** 2))

        return np.asarray(angles), np.asarray(E_vals) ** 2

    def compute_supergaussian_intensity(
        self,
        e_profile: list,
        n_profile: list,
        angles: np.ndarray,
        fwhm: float = 3.0,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convolve the plane-wave intensity curve with a super-Gaussian kernel.

        This approximates the effect of a beam with a finite angular
        spread (divergence) via a 1-D incoherent convolution.

        Parameters
        ----------
        fwhm : float
            Full-width at half-maximum of the divergence kernel in degrees.
        s_weight : float
            Weight for s-polarisation.  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation.  Set to 0 to exclude it.

        Returns
        -------
        angles : numpy.ndarray
        I_convolved : numpy.ndarray
        """
        th, I_nom = self.compute_nominal_intensity(e_profile, n_profile, angles,
                                                   s_weight=s_weight, p_weight=p_weight)

        # Need at least two angles to define a step size
        if len(th) < 2:
            return th, I_nom

        dtheta = th[1] - th[0]
        sigma  = fwhm_to_sigma(fwhm)

        # Build a symmetric delta-angle axis for the kernel
        N     = len(th)
        half  = N // 2
        delta = (np.arange(N) - half) * dtheta

        # Super-Gaussian kernel (normalised)
        kernel = np.exp(-np.abs(delta / sigma) ** self.exponent)
        kernel /= np.sum(kernel)

        I_conv = np.convolve(I_nom, kernel, mode="same")
        return th, I_conv

    # -----------------------------------------------------------------------
    # Divergence sub-angle grid
    # -----------------------------------------------------------------------

    def _divergence_fwhm_at_angle(self, theta_deg: float) -> float:
        """Linearly interpolate divergence FWHM between 0° and 85°.

        Maps ``div_at_zero`` (at 0°) to ``max_div`` (at 85°).
        If both are equal the divergence is angle-independent.
        """
        if self.max_div == self.div_at_zero:
            return self.div_at_zero
        t = np.clip(theta_deg, 0.0, 85.0) / 85.0
        return (1.0 - t) * self.div_at_zero + t * self.max_div

    def _angular_grid_and_weights(
        self,
        center_deg: float,
        fwhm_deg: float,
        samples: int,
        exponent: int = 4,
        skew: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a sub-angle grid and normalised super-Gaussian weights.

        The grid spans ±4σ around *center_deg* to capture >99.99% of
        the kernel mass.

        Returns
        -------
        sub_angles : numpy.ndarray  (degrees)
        weights    : numpy.ndarray  (sum to 1)
        """
        sigma = fwhm_to_sigma(fwhm_deg)
        span  = 4.0 * sigma

        sub_angles = np.linspace(center_deg - span, center_deg + span, samples)
        delta      = sub_angles - center_deg

        weights = super_gaussian_weight(delta, sigma, exponent, skew)
        return sub_angles, weights

    # -----------------------------------------------------------------------
    # Plotting methods
    # -----------------------------------------------------------------------

    def plot_coverslips(
        self,
        names: Optional[list[str]] = None,
        angles: Optional[np.ndarray] = None,
        wavelength: Optional[float] = None,
        pol: Optional[str] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0
    ) -> None:
        """Plot intensity vs incidence angle for one or more coverslips.

                Creates two figures:
                    1) Plane-wave (nominal) intensity vs angle
                    2) Divergence-averaged intensity vs angle

                Peaks of the divergence curve are annotated and both data sets are
                exported to Excel.

        Parameters
        ----------
        names : list of str, optional
            Subset of registry keys to plot.  Defaults to all registered
            coverslips.
        angles : array_like, optional
            Angular sweep in degrees.  Defaults to ``np.linspace(55, 82, 200)``.
        wavelength : float, optional
            Wavelength label used in the output filename (informational
            only; the simulation already used ``self.wavelength``).
        pol : str, optional
            Deprecated.  Use ``s_weight`` / ``p_weight`` instead.  If
            provided it is used only for the output filename.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude it.
        """

        if names is None:
            names = list(self.registry.keys())
        if angles is None:
            angles = np.linspace(55, 82, 200)

        pol_lbl = self._pol_label(s_weight, p_weight)
        fig_nom, ax_nom = plt.subplots(figsize=(12, 8))
        fig_div, ax_div = plt.subplots(figsize=(12, 8))

        # --- Plane-wave curves (nominal figure) ---
        for name in names:
            e_prof, n_prof, k_prof = self.registry[name][:3]
            n_complex = [complex(n, k) for n, k in zip(n_prof, k_prof)]
            th, I_nom = self.compute_nominal_intensity(e_prof, n_complex, angles,
                                                       s_weight=s_weight, p_weight=p_weight)
            ax_nom.plot(th, I_nom, linewidth=3, label=name)

            # Export plane-wave data
            pd.DataFrame({"Angle (deg)": th, "Intensity (a.u.)": I_nom}).to_excel(
                f"coverslip_{name}_wavelength_{wavelength}_intensity_nominal_{pol_lbl}_{self.n_super:2f}_{self.n_subs:2f}.xlsx",
                index=False,
            )
            peaks, _ = find_peaks(I_nom)
            print(f"[{name}] Plane-wave peak angles: {[round(float(th[p]), 2) for p in peaks]}")

        # --- Divergence-averaged curves (divergence figure) ---
        for name in names:
            e_prof, n_prof, k_prof = self.registry[name][:3]
            th, I_div = self.compute_supergaussian_intensity(
                e_prof, n_prof, angles, fwhm=self.div_at_zero,
                s_weight=s_weight, p_weight=p_weight,
            )
            ax_div.plot(
                th, I_div,
                label=f"{name} (div FWHM={self.max_div}°)",
                linewidth=3,
            )

            # Mark peaks
            peaks, _ = find_peaks(I_div)
            ax_div.scatter(th[peaks], I_div[peaks], marker="X", s=60, color="red")
            for p in peaks:
                ax_div.annotate(
                    f"({th[p]:.2f}°, {I_div[p]:.2f})",
                    (th[p], I_div[p]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=16,
                )

            # Export divergence-averaged data
            pd.DataFrame({"Angle (deg)": th, "Intensity (a.u.)": I_div}).to_excel(
                f"coverslip_{name}_wavelength_{wavelength}"
                f"_intensity_divergence_{self.div_at_zero}deg_{pol_lbl}_{self.n_super:2f}_{self.n_subs:2f}.xlsx",
                index=False,
            )

        ax_nom.set_title(
            f"Nominal Intensity vs Incidence Angle — {', '.join(names)}\n"
            f"(pol = {pol_lbl}, λ = {self.wavelength} nm)"
        )
        ax_nom.set_xlabel("Incidence Angle (°)")
        ax_nom.set_ylabel("Intensity (a.u.)")
        ax_nom.legend(loc="best", frameon=True)

        ax_div.set_title(
            f"Divergence-Averaged Intensity vs Incidence Angle — {', '.join(names)}\n"
            f"(div FWHM = {self.max_div}°, pol = {pol_lbl}, λ = {self.wavelength} nm)"
        )
        ax_div.set_xlabel("Incidence Angle (°)")
        ax_div.set_ylabel("Intensity (a.u.)")
        ax_div.legend(loc="best", frameon=True)
        plt.close(fig_nom)  # Prevent display in interactive environments
        # Display both figures in a single blocking call.
        # plt.show()

    def plot_optical_intensity_distribution(
        self,
        name: str,
        theta_inc: list[float] | float = 60,
        num_depth: int = 1000,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> None:
        """Plot |E|² vs depth through the multilayer stack.

        The depth sweep runs from −10 nm (into the superstrate) to
        ``total_thickness + 500 nm`` (into the sample).  The intensity
        is a weighted blend of s- and p-polarisations; only polarisations
        with weight > 0 are computed.

        Parameters
        ----------
        name : str
            Registry key of the coverslip to analyse.
        theta_inc : float or list of float
            Incidence angle(s) in degrees.
        num_depth : int
            Number of depth samples.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude it.
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}' in the registry.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm     = s_weight + p_weight

        e_prof, n_prof, k_prof = self.registry[name][:3]
        total_thickness = sum(e_prof[1:-1])
        print(f"Total stack thickness of '{name}': {total_thickness:.1f} nm")

        ds = np.linspace(-10, total_thickness + 500, num_depth)

        if not hasattr(theta_inc, "__iter__"):
            theta_inc = [theta_inc]

        fig, ax = plt.subplots(figsize=(12, 8))

        for angle in theta_inc:
            theta_rad = np.deg2rad(angle)

            # Run TMM only for polarisations with non-zero weight
            coh_s = coh_tmm("s", n_prof, e_prof, theta_rad, self.wavelength) if s_weight > 0 else None
            coh_p = coh_tmm("p", n_prof, e_prof, theta_rad, self.wavelength) if p_weight > 0 else None

            I_s_arr, I_p_arr, I_comb_arr = [], [], []
            for d in ds:
                layer_idx, dpos = find_in_structure_with_inf(e_prof, d)
                val = 0.0
                if coh_s is not None:
                    f = position_resolved(layer_idx, dpos, coh_s)
                    i_s = float(np.abs(f["Ey"]) ** 2)
                    I_s_arr.append(i_s)
                    val += (s_weight / norm) * i_s
                if coh_p is not None:
                    f = position_resolved(layer_idx, dpos, coh_p)
                    i_p = float(np.abs(f["Ex"]) ** 2 + np.abs(f["Ez"]) ** 2)
                    I_p_arr.append(i_p)
                    val += (p_weight / norm) * i_p
                I_comb_arr.append(val)

            I_comb = np.array(I_comb_arr)

            ax.plot(ds, I_comb, label=f"{pol_lbl}-pol @ {angle}°", linewidth=2)

            # Mark layer interfaces
            interfaces = np.insert(np.cumsum(e_prof[1:-1]), 0, 0)
            for pos in interfaces:
                if ds[0] <= pos <= ds[-1]:
                    ax.axvline(x=pos, color="grey", linestyle="--", alpha=0.5)

            # Build export DataFrame — include per-pol columns when both are active
            df_dict: dict = {"Depth (nm)": ds, f"I_combined ({pol_lbl})": I_comb}
            if coh_s is not None:
                df_dict["I_s (TE)"] = np.array(I_s_arr)
            if coh_p is not None:
                df_dict["I_p (TM)"] = np.array(I_p_arr)
            df = pd.DataFrame(df_dict)
            df.to_excel(f"{name}@{angle}_{pol_lbl}_optical_intensity_distribution.xlsx", index=False)

            # Sample-side profile (after last interface), depth re-zeroed and normalised
            df_after = df[ds >= total_thickness].copy()
            df_after.loc[:, "Depth (nm)"] -= total_thickness
            comb_max = df_after[f"I_combined ({pol_lbl})"].max() or 1.0
            df_after.loc[:, f"I_combined ({pol_lbl})"] /= comb_max
            df_after.to_excel(f"{name}@{angle}_{pol_lbl}_intensity_after_total_thickness.xlsx", index=False)
            print(f"  [{name}@{angle}°] Sample-side data exported.")

        ax.set_xlabel("Depth (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Optical Intensity Distribution — {name}  ({pol_lbl}-pol)")
        ax.legend(loc="best", frameon=True)
        plt.show()

    def plot_reflectance_transmission_absorbance_vs_angle(
        self,
        name: str,
        angle_range: tuple[float, float] = (0, 85),
        num: int = 500,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> None:
        """Plot R, T, and A vs incidence angle for a registered coverslip.

        Complex refractive indices (n + ik) are used to account for
        absorbing layers.  R, T, A are the weighted combination of
        s- and p-polarisation contributions as set by the weights.

        Parameters
        ----------
        name : str
            Registry key.
        angle_range : (min_deg, max_deg)
        num : int
            Number of angle samples.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude it.
        """
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        e_prof, n_prof, k_prof = self.registry[name][:3]
        n_complex = [complex(n, k) for n, k in zip(n_prof, k_prof)]
        angles = np.linspace(angle_range[0], angle_range[1], num)

        R_vals, T_vals, A_vals = self._calc_rta(
            n_complex, e_prof, self.wavelength, angles, s_weight, p_weight
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(angles, R_vals, lw=2, label="Reflectance R")
        ax.plot(angles, T_vals, lw=2, label="Transmission T")
        ax.plot(angles, A_vals, lw=2, label="Absorbance A")
        ax.set_xlabel("Incidence Angle (°)", fontsize=14)
        ax.set_ylabel("R, T, A", fontsize=14)
        ax.set_title(
            f"{name}: R, T, A vs Incidence Angle\n({pol_lbl}-pol, λ = {self.wavelength} nm)",
            fontsize=16,
        )
        ax.grid(True)
        ax.legend()
        plt.show()

    def plot_transmission_vs_stack_optical_thickness(
        self,
        name: str,
        scale_range: tuple[float, float] = (0.5, 2.0),
        num: int = 200,
        theta: float = 0,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> None:
        """Plot transmission T vs total optical thickness Σ(n_i d_i).

        All layer physical thicknesses are uniformly scaled by a factor
        swept over *scale_range*, while refractive indices remain fixed.
        Useful for sensitivity and manufacturing-tolerance analyses.

        Parameters
        ----------
        name : str
            Registry key.
        scale_range : (min_scale, max_scale)
            Multiplicative scale factors applied to all layer thicknesses.
        num : int
            Number of scale-factor samples.
        theta : float
            Incidence angle in degrees (default 0° = normal incidence).
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude.
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm    = s_weight + p_weight

        reg_entry = self.registry[name]
        e_orig, n_orig = reg_entry[0], reg_entry[1]

        ds_orig  = np.array(e_orig[1:-1], dtype=float)
        n_layers = np.array(n_orig[1:-1], dtype=float)
        base_opt_thick = float(np.sum(n_layers * ds_orig))

        scales     = np.linspace(scale_range[0], scale_range[1], num)
        opt_thicks = base_opt_thick * scales
        T_vals     = []

        for s in scales:
            ds     = ds_orig * s
            e_prof = [inf] + list(ds) + [inf]
            T_blend = 0.0
            for pol, weight in (("s", s_weight), ("p", p_weight)):
                if weight == 0.0:
                    continue
                coh = coh_tmm(pol, n_orig, e_prof, np.deg2rad(theta), self.wavelength)
                T_blend += (weight / norm) * float(coh["T"])
            T_vals.append(T_blend)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(opt_thicks, T_vals, lw=2)
        ax.set_xlabel(r"Total Optical Thickness $\sum n_i d_i$ (nm)", fontsize=14)
        ax.set_ylabel(r"Transmission $T$", fontsize=14)
        ax.set_title(
            f"{name}: $T$ vs Stack Optical Thickness\n"
            f"(@ {theta}°, {pol_lbl}-pol)",
            fontsize=16,
        )
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_transmission_vs_wavelength(
        self,
        name: str,
        wavelength_range: tuple[float, float] = (400, 800),
        num: int = 500,
        theta: list[float] | float = 0,
        heatmap: bool = True,
        angle_range: tuple[float, float] = (0, 60),
        angle_num: int = 100,
        cmap: str = "jet",
        export_excel: bool = False,
        wavelength_min: Optional[int] = None,
        wavelength_max: Optional[int] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ):
        """Sweep wavelength and plot transmission (line or heatmap).

        Line mode
        ---------
        Plots T and R vs wavelength for each angle in *theta*.

        Heatmap mode (``heatmap=True``)
        --------------------------------
        Produces a 2-D colour map: X = incidence angle, Y = wavelength,
        colour = T.  Also prints mean transmission over user-defined
        wavelength windows.

        Parameters
        ----------
        name : str
            Registry key.
        wavelength_range : (min_nm, max_nm)
        num : int
            Number of wavelength samples.
        theta : float or list of float
            Incidence angle(s) in degrees (line mode only).
        heatmap : bool
            Switch between line (False) and 2-D heatmap (True) output.
        angle_range : (min_deg, max_deg)
            Angular axis for the heatmap.
        angle_num : int
            Number of angle samples for the heatmap.
        cmap : str
            Matplotlib colormap for the heatmap.
        export_excel : bool
            If True, export the heatmap grid to an Excel file.
        wavelength_min, wavelength_max : int, optional
            Sub-range for the mean-T printout in heatmap mode.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude.

        Returns
        -------
        (wavelengths, angles, T_grid) in heatmap mode, else None.
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm    = s_weight + p_weight

        e_prof, n_prof, k_prof = self.registry[name][:3]
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num)

        # ---- 2-D heatmap mode ----
        if heatmap:
            angles = np.linspace(angle_range[0], angle_range[1], angle_num)
            T_grid = np.zeros((len(wavelengths), len(angles)), dtype=float)

            materials = self.registry[name][4]
            for i, wl in enumerate(wavelengths):
                n_wl = ([self.n_super]
                        + [complex(nk_measured(wl, m), excitation_coeff(wl, m)) for m in materials]
                        + [self.n_subs])
                for j, ang in enumerate(angles):
                    T_blend = 0.0
                    for pol, weight in (("s", s_weight), ("p", p_weight)):
                        if weight == 0.0:
                            continue
                        coh = coh_tmm(pol, n_wl, e_prof, np.deg2rad(ang), wl)
                        T_blend += (weight / norm) * float(coh.get("T", 0.0))
                    T_grid[i, j] = T_blend

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.pcolormesh(angles, wavelengths, T_grid, shading="auto", cmap=cmap)
            ax.set_xlabel("Incidence Angle (°)")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title(f"{name}: Transmission heatmap ({pol_lbl}-pol), {self.wavelength} nm")
            plt.colorbar(im, ax=ax, label="Transmission T")
            plt.tight_layout()
            plt.show()

            idx_a = (
                0 if wavelength_min is None
                else int(np.abs(wavelengths - wavelength_min).argmin())
            )
            idx_b = (
                len(wavelengths) - 1 if wavelength_max is None
                else int(np.abs(wavelengths - wavelength_max).argmin())
            )
            print(
                f"Mean T (all λ, all θ): {np.mean(T_grid):.4f}\n"
                f"Mean T ({wavelengths[idx_a]:.0f}–{wavelengths[idx_b]:.0f} nm, all θ): "
                f"{np.mean(T_grid[idx_a:idx_b+1, :]):.4f}"
            )

            # print the transmission at an interval of 10nm for all the angles from wavelength_min to wavelength_max
            print(f"\nTransmission values at 10nm intervals from {wavelengths[idx_a]:.0f} nm to {wavelengths[idx_b]:.0f} nm:")
            for wl in range(int(wavelengths[idx_a]), int(wavelengths[idx_b]) + 1, 10):
                idx_wl = int(np.abs(wavelengths - wl).argmin())
                mean_T_at_wl = np.mean(T_grid[idx_wl, :])
                print(f"  {wavelengths[idx_wl]:.0f} nm: Mean T across angles = {mean_T_at_wl:.4f}")

            if export_excel:
                df = pd.DataFrame(
                    T_grid,
                    index=np.round(wavelengths, 6),
                    columns=np.round(angles, 6),
                )
                df.index.name = "Wavelength_nm"
                outname = (
                    f"{name}_T_heatmap_{angle_range[0]}to{angle_range[1]}deg_"
                    f"{wavelength_range[0]}to{wavelength_range[1]}nm.xlsx"
                )
                df.to_excel(outname)
                print(f"Heatmap data exported to '{outname}'.")

            return wavelengths, angles, T_grid

        # ---- Line mode: one curve per angle ----
        if not hasattr(theta, "__iter__") or isinstance(theta, (str, bytes)):
            theta = [theta]

        fig_line, ax_line = plt.subplots(figsize=(8, 5))
        all_T, all_R = [], []

        for theta_ in theta:
            T_vals, R_vals = [], []
            for wl in wavelengths:
                T_blend = R_blend = 0.0
                for pol, weight in (("s", s_weight), ("p", p_weight)):
                    if weight == 0.0:
                        continue
                    coh = coh_tmm(pol, n_prof, e_prof, np.deg2rad(theta_), wl)
                    T_blend += (weight / norm) * float(coh.get("T", 0.0))
                    R_blend += (weight / norm) * float(coh.get("R", 0.0))
                T_vals.append(T_blend)
                R_vals.append(R_blend)
            all_T.append(T_vals)
            all_R.append(R_vals)
            ax_line.plot(wavelengths, T_vals, lw=2, label=f"T @ {theta_}°")
            ax_line.plot(wavelengths, R_vals, lw=1.5, linestyle="--", label=f"R @ {theta_}°")

        ax_line.set_xlabel("Wavelength (nm)", fontsize=14)
        ax_line.set_ylabel("Transmission / Reflectance", fontsize=14)
        ax_line.set_ylim(0, 1)
        ax_line.grid(True)
        ax_line.legend()
        plt.tight_layout()
        plt.show()

        # Also display the mean T and R across all specified angles
        if all_T:
            fig_mean, ax_mean = plt.subplots(figsize=(8, 5))
            mean_T = np.mean(np.array(all_T), axis=0)
            mean_R = np.mean(np.array(all_R), axis=0)
            ax_mean.plot(wavelengths, mean_T, lw=2, label="Mean T")
            ax_mean.plot(wavelengths, mean_R, lw=2, linestyle="--", label="Mean R")
            ax_mean.set_xlabel("Wavelength (nm)")
            ax_mean.set_ylabel("Mean Transmission / Reflectance")
            ax_mean.set_ylim(0, 1)
            ax_mean.grid(True)
            ax_mean.legend()
            plt.tight_layout()
            plt.show()

    def plot_rta_vs_wavelength(
        self,
        name: str,
        wavelength_range: tuple[float, float] = (400, 800),
        num: int = 500,
        theta: list[float] | float = 0,
        heatmap: bool = True,
        angle_range: tuple[float, float] = (0, 60),
        angle_num: int = 100,
        cmap: str = "jet",
        export_excel: bool = False,
        wavelength_min: Optional[int] = None,
        wavelength_max: Optional[int] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ):
        """Sweep wavelength and plot reflectance, transmission and absorbance.

        Same interface as :meth:`plot_transmission_vs_wavelength`, but R, T
        and A are all computed.  The dispersive complex index (n + ik) of
        each layer is re-evaluated at every wavelength in both modes, so
        absorbance is meaningful across the whole sweep.

        Line mode
        ---------
        One figure per quantity (R, T, A) with a curve per angle in *theta*,
        followed by a figure of the mean R, T, A across those angles.

        Heatmap mode (``heatmap=True``)
        --------------------------------
        Three side-by-side 2-D colour maps (X = incidence angle,
        Y = wavelength, colour = R / T / A).  Also prints mean values over
        the user-defined wavelength window.

        Parameters
        ----------
        name : str
            Registry key.
        wavelength_range : (min_nm, max_nm)
        num : int
            Number of wavelength samples.
        theta : float or list of float
            Incidence angle(s) in degrees (line mode only).
        heatmap : bool
            Switch between line (False) and 2-D heatmap (True) output.
        angle_range : (min_deg, max_deg)
            Angular axis for the heatmap.
        angle_num : int
            Number of angle samples for the heatmap.
        cmap : str
            Matplotlib colormap for the heatmap.
        export_excel : bool
            If True, export the heatmap grids to an Excel file (one sheet
            each for R, T, A).
        wavelength_min, wavelength_max : int, optional
            Sub-range for the mean printout in heatmap mode.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude.

        Returns
        -------
        (wavelengths, angles, R_grid, T_grid, A_grid) in heatmap mode,
        else (wavelengths, thetas, R, T, A) with arrays of shape
        (len(theta), num).
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl   = self._pol_label(s_weight, p_weight)
        e_prof    = self.registry[name][0]
        materials = self.registry[name][4]
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num)

        def n_at(wl: float) -> list:
            return ([self.n_super]
                    + [complex(nk_measured(wl, m), excitation_coeff(wl, m)) for m in materials]
                    + [self.n_subs])

        if heatmap:
            angles = np.linspace(angle_range[0], angle_range[1], angle_num)
        else:
            if not hasattr(theta, "__iter__") or isinstance(theta, (str, bytes)):
                theta = [theta]
            angles = np.asarray(theta, dtype=float)

        # grids have shape (n_wavelengths, n_angles)
        R_grid = np.zeros((len(wavelengths), len(angles)))
        T_grid = np.zeros_like(R_grid)
        A_grid = np.zeros_like(R_grid)
        for i, wl in enumerate(wavelengths):
            R_grid[i], T_grid[i], A_grid[i] = self._calc_rta(
                n_at(wl), e_prof, wl, angles, s_weight, p_weight
            )

        quantities = (
            ("R", "Reflectance", R_grid),
            ("T", "Transmission", T_grid),
            ("A", "Absorbance", A_grid),
        )

        # ---- 2-D heatmap mode ----
        if heatmap:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            for ax, (sym, label, grid) in zip(axes, quantities):
                im = ax.pcolormesh(angles, wavelengths, grid, shading="auto", cmap=cmap)
                ax.set_xlabel("Incidence Angle (°)")
                ax.set_title(f"{label} {sym}")
                plt.colorbar(im, ax=ax, label=f"{label} {sym}")
            axes[0].set_ylabel("Wavelength (nm)")
            fig.suptitle(f"{name}: R, T, A heatmaps ({pol_lbl}-pol)")
            plt.tight_layout()
            plt.show()

            idx_a = (
                0 if wavelength_min is None
                else int(np.abs(wavelengths - wavelength_min).argmin())
            )
            idx_b = (
                len(wavelengths) - 1 if wavelength_max is None
                else int(np.abs(wavelengths - wavelength_max).argmin())
            )
            for sym, _, grid in quantities:
                print(
                    f"Mean {sym} (all λ, all θ): {np.nanmean(grid):.4f} | "
                    f"({wavelengths[idx_a]:.0f}–{wavelengths[idx_b]:.0f} nm, all θ): "
                    f"{np.nanmean(grid[idx_a:idx_b+1, :]):.4f}"
                )

            print(f"\nMean R / T / A across angles at 10nm intervals "
                  f"from {wavelengths[idx_a]:.0f} nm to {wavelengths[idx_b]:.0f} nm:")
            for wl in range(int(wavelengths[idx_a]), int(wavelengths[idx_b]) + 1, 10):
                idx_wl = int(np.abs(wavelengths - wl).argmin())
                print(
                    f"  {wavelengths[idx_wl]:.0f} nm: "
                    f"R = {np.nanmean(R_grid[idx_wl, :]):.4f}, "
                    f"T = {np.nanmean(T_grid[idx_wl, :]):.4f}, "
                    f"A = {np.nanmean(A_grid[idx_wl, :]):.4f}"
                )

            if export_excel:
                outname = (
                    f"{name}_RTA_heatmap_{angle_range[0]}to{angle_range[1]}deg_"
                    f"{wavelength_range[0]}to{wavelength_range[1]}nm.xlsx"
                )
                with pd.ExcelWriter(outname) as writer:
                    for sym, _, grid in quantities:
                        df = pd.DataFrame(
                            grid,
                            index=np.round(wavelengths, 6),
                            columns=np.round(angles, 6),
                        )
                        df.index.name = "Wavelength_nm"
                        df.to_excel(writer, sheet_name=sym)
                print(f"Heatmap data exported to '{outname}'.")

            return wavelengths, angles, R_grid, T_grid, A_grid

        # ---- Line mode: one figure per quantity, one curve per angle ----
        for sym, label, grid in quantities:
            fig_line, ax_line = plt.subplots(figsize=(8, 5))
            for j, theta_ in enumerate(angles):
                ax_line.plot(wavelengths, grid[:, j], lw=2, label=f"{sym} @ {theta_:g}°")
            ax_line.set_xlabel("Wavelength (nm)", fontsize=14)
            ax_line.set_ylabel(f"{label} {sym}", fontsize=14)
            ax_line.set_title(f"{name}: {label} vs Wavelength ({pol_lbl}-pol)")
            ax_line.set_ylim(0, 1)
            ax_line.grid(True)
            ax_line.legend()
            plt.tight_layout()
            plt.show()

        # Mean R, T, A across all specified angles
        fig_mean, ax_mean = plt.subplots(figsize=(8, 5))
        for (sym, _, grid), ls in zip(quantities, ("--", "-", ":")):
            ax_mean.plot(wavelengths, np.nanmean(grid, axis=1), lw=2, linestyle=ls,
                         label=f"Mean {sym}")
        ax_mean.set_xlabel("Wavelength (nm)")
        ax_mean.set_ylabel("Mean R / T / A")
        ax_mean.set_title(f"{name}: mean over θ = {', '.join(f'{a:g}' for a in angles)}°")
        ax_mean.set_ylim(0, 1)
        ax_mean.grid(True)
        ax_mean.legend()
        plt.tight_layout()
        plt.show()

        return wavelengths, angles, R_grid.T, T_grid.T, A_grid.T

    # -----------------------------------------------------------------------
    # Penetration-depth estimation (divergent beam)
    # -----------------------------------------------------------------------

    @staticmethod
    def _exp_intensity(z: np.ndarray, A: float, delta: float, B: float) -> np.ndarray:
        """Evanescent intensity decay model: I(z) = A·exp(−2z/δ) + B.

        Parameters
        ----------
        z : numpy.ndarray
            Depth into the sample (nm, from the last interface).
        A : float
            Amplitude (intensity at z = 0).
        delta : float
            Penetration depth δ (nm).
        B : float
            Background / offset term.
        """
        return A * np.exp(-2.0 * z / np.maximum(delta, 1e-9)) + B

    def penetration_depth_from_divergent_beam(
        self,
        name: str,
        theta_nom_deg: float,
        divergence_fwhm_deg: float = 2.0,
        samples: int = 101,
        exponent: Optional[int] = None,
        skew: Optional[float] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
        fit_range_nm: tuple[float, float] = (0.0, 300.0),
        depth_nm_after_stack: float = 500.0,
        num_depth: int = 1200,
        plot: bool = True,
        export_excel: bool = True,
    ) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate the effective evanescent penetration depth δ.

        The beam is modelled as an incoherent superposition of plane waves
        with weights given by the super-Gaussian divergence kernel.  The
        resulting angle-averaged depth profile I(z) in the sample half-space
        is fitted to A·exp(−2z/δ) + B to extract δ.

        Parameters
        ----------
        name : str
            Registry key.
        theta_nom_deg : float
            Nominal (central) incidence angle in degrees.
        divergence_fwhm_deg : float
            FWHM of the beam divergence kernel in degrees.
        samples : int
            Number of sub-angles to integrate over.
        exponent : int, optional
            Super-Gaussian exponent (defaults to ``self.exponent``).
        skew : float, optional
            Asymmetry (defaults to ``self.skew``).
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude it.
        fit_range_nm : (z0, z1)
            Depth range (nm) within which the exponential is fitted.
        depth_nm_after_stack : float
            How far (nm) into the sample the depth grid extends.
        num_depth : int
            Number of depth samples.
        plot : bool
            Whether to produce a matplotlib figure.
        export_excel : bool
            Whether to write fit results to an Excel file.

        Returns
        -------
        delta_nm : float
            Fitted penetration depth in nm.
        params : dict
            Dictionary with keys 'A', 'B', 'R2'.
        z_after : numpy.ndarray
            Depth axis (nm, from last interface).
        I_after : numpy.ndarray
            Angle-averaged intensity profile.
        I_fit : numpy.ndarray
            Model intensity evaluated over ``z_after``.
        """
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")

        reg_entry = self.registry[name]
        e_prof, n_prof = reg_entry[0], reg_entry[1]

        if exponent is None: exponent = self.exponent
        if skew    is None: skew     = self.skew

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm    = s_weight + p_weight

        total_thickness = float(np.sum(e_prof[1:-1]))
        ds = np.linspace(-10.0, total_thickness + depth_nm_after_stack, num_depth)

        # Build divergence-weighted sub-angle grid
        sub_angles, weights = self._angular_grid_and_weights(
            center_deg=theta_nom_deg,
            fwhm_deg=float(divergence_fwhm_deg),
            samples=int(samples),
            exponent=int(exponent),
            skew=float(skew),
        )

        # Incoherent (intensity) average across sub-angles
        I_accum = np.zeros_like(ds, dtype=float)
        for th_sub, w in zip(sub_angles, weights):
            theta_rad = np.deg2rad(th_sub)

            coh_s = coh_tmm("s", n_prof, e_prof, theta_rad, self.wavelength) \
                    if s_weight > 0 else None
            coh_p = coh_tmm("p", n_prof, e_prof, theta_rad, self.wavelength) \
                    if p_weight > 0 else None

            I_sub = np.zeros_like(ds, dtype=float)
            for i, d in enumerate(ds):
                layer_idx, dpos = find_in_structure_with_inf(e_prof, d)
                if s_weight > 0:
                    f = position_resolved(layer_idx, dpos, coh_s)
                    I_sub[i] += (s_weight / norm) * np.abs(f["Ey"]) ** 2
                if p_weight > 0:
                    f = position_resolved(layer_idx, dpos, coh_p)
                    I_sub[i] += (p_weight / norm) * (np.abs(f["Ex"]) ** 2 + np.abs(f["Ez"]) ** 2)

            I_accum += w * I_sub

        # Extract and re-zero the sample-side profile
        mask_after = ds >= total_thickness
        z_after    = ds[mask_after] - total_thickness
        I_after    = I_accum[mask_after]

        # Fit window
        z0, z1 = float(fit_range_nm[0]), float(fit_range_nm[1])
        wmask  = (z_after >= z0) & (z_after <= z1)
        if not np.any(wmask):
            raise ValueError("fit_range_nm does not overlap the available depth range.")
        z_fit    = z_after[wmask]
        I_fitwin = I_after[wmask]

        # Robust baseline estimate from the far tail of the profile
        tail_mask = z_after >= min(z1 + 50.0, z_after[-1])
        B0 = float(np.median(I_after[tail_mask])) if np.any(tail_mask) \
             else float(np.percentile(I_fitwin, 5))

        # Amplitude initial guess
        A0 = max(I_fitwin[0] - B0, np.max(I_fitwin) - B0, 1e-6)

        # Analytic plane-wave penetration depth as starting guess
        n1   = complex(self.n_super)
        n2   = complex(self.n_subs)
        sin2 = n1.real ** 2 * np.sin(np.deg2rad(theta_nom_deg)) ** 2
        kappa = np.sqrt(max((sin2 - n2.real ** 2), 1e-12))
        delta0 = self.wavelength / (4.0 * np.pi * kappa)

        p0    = [A0, max(delta0, 50.0), B0]
        lower = [0.0, 10.0, -np.inf]
        upper = [np.inf, 2000.0, np.inf]

        try:
            popt, _ = curve_fit(
                self._exp_intensity, z_fit, I_fitwin,
                p0=p0, bounds=(lower, upper), maxfev=20_000
            )
        except RuntimeError:
            # Fall back to unconstrained fit if bounded fit fails
            popt, _ = curve_fit(self._exp_intensity, z_fit, I_fitwin,
                                p0=p0, maxfev=20_000)

        A_hat, delta_hat, B_hat = popt
        I_model = self._exp_intensity(z_fit, *popt)

        # Goodness-of-fit (R²)
        ss_res = float(np.sum((I_fitwin - I_model) ** 2))
        ss_tot = float(np.sum((I_fitwin - np.mean(I_fitwin)) ** 2)) + 1e-15
        r2 = 1.0 - ss_res / ss_tot

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                z_after, I_after, lw=2,
                label=f"Avg. intensity @ {theta_nom_deg:.2f}° (FWHM {divergence_fwhm_deg:.2f}°)"
            )
            ax.plot(
                z_fit, I_model, lw=2, linestyle="--",
                label=rf"Fit: $A\,e^{{-2z/\delta}}+B$;  "
                      rf"$\delta$={delta_hat:.1f} nm,  $R^2$={r2:.3f}"
            )
            ax.axvspan(z0, z1, color="0.9", zorder=0, label="Fit window")
            ax.set_xlabel("Depth into sample (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(
                f"{name} — Effective penetration depth "
                f"(divergent beam, pol = {pol_lbl})"
            )
            ax.legend(loc="best", frameon=True)
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        if export_excel:
            outname = (
                f"{name}@{theta_nom_deg:.3f}-deg_div_FWHM_"
                f"{divergence_fwhm_deg:.2f}-{pol_lbl}_penetration_depth.xlsx"
            )
            with pd.ExcelWriter(outname, engine="xlsxwriter") as xw:
                pd.DataFrame({"z_nm_after_stack": z_after, "I_avg": I_after}).to_excel(
                    xw, index=False, sheet_name="Profile"
                )
                pd.DataFrame(
                    {"z_nm_fit": z_fit, "I_fit_data": I_fitwin, "I_fit_model": I_model}
                ).to_excel(xw, index=False, sheet_name="FitWindow")
                pd.DataFrame(
                    [{"theta_nom_deg": theta_nom_deg,
                      "divergence_fwhm_deg": divergence_fwhm_deg,
                      "samples": samples, "pol": pol_lbl,
                      "s_weight": s_weight, "p_weight": p_weight,
                      "fit_z0_nm": z0, "fit_z1_nm": z1,
                      "A_hat": A_hat, "delta_hat_nm": delta_hat,
                      "B_hat": B_hat, "R2": r2}]
                ).to_excel(xw, index=False, sheet_name="FitParams")
            print(f"Penetration-depth results exported to '{outname}'.")

        I_fit_full = self._exp_intensity(z_after, A_hat, delta_hat, B_hat)
        return (
            float(delta_hat),
            {"A": float(A_hat), "B": float(B_hat), "R2": float(r2)},
            z_after,
            I_after,
            I_fit_full,
        )

    # -----------------------------------------------------------------------
    # Excitation wavelength × angle heatmap
    # -----------------------------------------------------------------------

    def plot_excitation_wavelength_vs_angle_with_colormap(
        self,
        name: str,
        angle_range: tuple[float, float] = (60, 89),
        wavelength_range: tuple[float, float] = (300, 800),
        wavelength_control: Optional[int] = 602,
        num_angles: int = 241,
        num_wavelengths: int = 501,
        cmap: str = "jet",
        export_excel: bool = False,
        savefig: Optional[str] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> np.ndarray:
        """Plot a 2-D heatmap of evanescent excitation intensity over wavelength × angle.

        At each (wavelength, angle) point the method recomputes layer
        refractive indices n(λ), calls ``coh_tmm`` at that wavelength,
        and evaluates the weighted field intensity at the stack / sample
        interface.

        Parameters
        ----------
        name : str
            Registry key of the coverslip to analyse.
        angle_range : (min_deg, max_deg)
        wavelength_range : (min_nm, max_nm)
        num_angles, num_wavelengths : int
            Grid resolution.
        cmap : str
            Matplotlib colormap.
        export_excel : bool
            Export intensity grid to Excel.
        savefig : str or None
            Output figure path (format inferred from extension).
        s_weight : float
            Weight for s-polarisation.  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation.  Set to 0 to exclude it.

        Returns
        -------
        I_grid : numpy.ndarray, shape (num_wavelengths, num_angles)
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm     = s_weight + p_weight

        reg_entry  = self.registry[name]
        e_prof     = reg_entry[0]
        materials  = reg_entry[4]

        angles      = np.linspace(angle_range[0],      angle_range[1],      num_angles)
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
        I_grid      = np.zeros((len(wavelengths), len(angles)), dtype=float)

        total_thickness = float(sum(e_prof[1:-1]))
        layer_idx, dpos = find_in_structure_with_inf(e_prof, total_thickness)

        for i, wl in enumerate(wavelengths):
            print(f"  [{name}] λ = {wl:.1f} nm  ({i + 1}/{len(wavelengths)})")

            n_profile_wl = (
                [self.n_super]
                + [nk_measured(wl, m) for m in materials]
                + [self.n_subs]
            )

            for j, ang in enumerate(angles):
                theta_rad = np.deg2rad(ang)
                val = 0.0
                for pol, weight in (("s", s_weight), ("p", p_weight)):
                    if weight == 0.0:
                        continue
                    coh  = coh_tmm(pol, n_profile_wl, e_prof, theta_rad, wl)
                    data = position_resolved(layer_idx, dpos, coh)
                    if pol == "s":
                        val += (weight / norm) * float(np.abs(data["Ey"]) ** 2)
                    else:
                        val += (weight / norm) * float(
                            np.abs(data["Ex"]) ** 2 + np.abs(data["Ez"]) ** 2
                        )
                I_grid[i, j] = val

            row_max = I_grid[i].max()
            if row_max > 0:
                I_grid[i] /= row_max

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(I_grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=(angles[0], angles[-1], wavelengths[0], wavelengths[-1]))
        ax.set_xlabel("Incidence Angle (°)")
        ax.set_ylabel("Excitation Wavelength (nm)")
        ax.set_title(
            f"{name}: λ_excitation swept | λ_control = λ_excitation  ({pol_lbl}-pol)"
        )
        plt.colorbar(im, ax=ax, label="Normalised Intensity (a.u.)")
        plt.tight_layout()
        if savefig:
            path = r"code\Modular"
            savefig = os.path.join(path, savefig)
            plt.savefig(savefig, format="svg")
            print(f"Heatmap figure saved to '{savefig}'.")
        plt.show()

        if export_excel:
            df = pd.DataFrame(
                I_grid,
                index=np.round(wavelengths, 3),
                columns=np.round(angles, 3),
            )
            df.index.name   = "Wavelength_nm"
            df.columns.name = "Angle_deg"
            outname = (
                f"{name}_excitation_heatmap_"
                f"{wavelength_range[0]:.0f}to{wavelength_range[1]:.0f}nm_"
                f"{angle_range[0]:.0f}to{angle_range[1]:.0f}deg.xlsx"
            )
            df.to_excel(outname)
            print(f"Heatmap exported to '{outname}'.")

        return I_grid

    # -----------------------------------------------------------------------
    # Control-wavelength × angle heatmap  (fixed excitation λ)
    # -----------------------------------------------------------------------

    def plot_control_wavelength_vs_angle_with_colormap(
        self,
        name: str,
        lambda_excitation: float,
        angle_range: tuple[float, float] = (60, 90),
        control_wavelength_range: tuple[float, float] = (300, 800),
        num_angles: int = 241,
        num_control_wavelengths: int = 500,
        cmap: str = "jet",
        export_excel: bool = False,
        savefig: Optional[str] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> np.ndarray:
        """Heatmap of excitation intensity vs control wavelength × angle.

        The excitation wavelength is fixed and passed to ``coh_tmm``;
        the control wavelength is swept only for the n(λ) look-up.

        Parameters
        ----------
        name : str
            Registry key of the coverslip to analyse.
        lambda_excitation : float
            Fixed excitation wavelength in nm (passed to ``coh_tmm``).
        angle_range : (min_deg, max_deg)
        control_wavelength_range : (min_nm, max_nm)
        num_angles, num_control_wavelengths : int
            Grid resolution.
        cmap : str
            Matplotlib colormap.
        export_excel : bool
            Export intensity grid to Excel.
        savefig : str or None
            Output figure path.
        s_weight : float
            Weight for s-polarisation.  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation.  Set to 0 to exclude it.

        Returns
        -------
        I_grid : numpy.ndarray, shape (num_control_wavelengths, num_angles)
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm     = s_weight + p_weight

        reg_entry  = self.registry[name]
        e_prof     = reg_entry[0]
        materials  = reg_entry[4]

        angles              = np.linspace(angle_range[0],               angle_range[1],               num_angles)
        control_wavelengths = np.linspace(control_wavelength_range[0],  control_wavelength_range[1],  num_control_wavelengths)
        I_grid              = np.zeros((len(control_wavelengths), len(angles)), dtype=float)

        total_thickness = float(sum(e_prof[1:-1]))
        layer_idx, dpos = find_in_structure_with_inf(e_prof, total_thickness)

        for i, wl_ctrl in enumerate(control_wavelengths):
            print(
                f"  [{name}] λ_control = {wl_ctrl:.1f} nm  "
                f"(λ_excitation fixed @ {lambda_excitation} nm)  "
                f"({i + 1}/{len(control_wavelengths)})"
            )

            n_profile_ctrl = (
                [self.n_super]
                + [nk_measured(wl_ctrl, m) for m in materials]
                + [self.n_subs]
            )

            for j, ang in enumerate(angles):
                theta_rad = np.deg2rad(ang)
                val = 0.0
                for pol, weight in (("s", s_weight), ("p", p_weight)):
                    if weight == 0.0:
                        continue
                    coh  = coh_tmm(pol, n_profile_ctrl, e_prof, theta_rad, lambda_excitation)
                    data = position_resolved(layer_idx, dpos, coh)
                    if pol == "s":
                        val += (weight / norm) * float(np.abs(data["Ey"]) ** 2)
                    else:
                        val += (weight / norm) * float(
                            np.abs(data["Ex"]) ** 2 + np.abs(data["Ez"]) ** 2
                        )
                I_grid[i, j] = val

            row_max = I_grid[i].max()
            if row_max > 0:
                I_grid[i] /= row_max

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            I_grid,
            aspect="auto",
            origin="lower",
            extent=(angles[0], angles[-1],
                    control_wavelength_range[0], control_wavelength_range[1]),
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_xlabel("Incidence Angle (°)")
        ax.set_ylabel("Control Wavelength (nm)")
        ax.set_title(
            f"{name}: Intensity heatmap\n"
            f"λ_excitation = {lambda_excitation} nm (fixed),  "
            f"λ_control swept  ({pol_lbl}-pol)"
        )
        plt.colorbar(im, ax=ax, label="Normalised Intensity (a.u.)")
        plt.tight_layout()

        if savefig:
            fmt = savefig.rsplit(".", 1)[-1] if "." in savefig else "svg"
            path = r"code\Modular"
            savefig = os.path.join(path, savefig)
            plt.savefig(savefig, format=fmt)
            print(f"Figure saved to '{savefig}'.")

        plt.show()

        if export_excel:
            df = pd.DataFrame(
                I_grid,
                index=np.round(control_wavelengths, 3),
                columns=np.round(angles, 3),
            )
            df.index.name   = "Control_Wavelength_nm"
            df.columns.name = "Angle_deg"
            outname = (
                f"{name}_ctrl_heatmap_"
                f"exc{lambda_excitation:.0f}nm_"
                f"ctrl{control_wavelength_range[0]:.0f}to{control_wavelength_range[1]:.0f}nm_"
                f"{angle_range[0]:.0f}to{angle_range[1]:.0f}deg.xlsx"
            )
            df.to_excel(outname)
            print(f"Heatmap exported to '{outname}'.")

        return I_grid
    


    def plot_excitation_wavelength_vs_angle_with_colormap_2(
        self,
        name: str,
        lambda_control: float,
        angle_range: tuple[float, float] = (60, 90),
        excitation_wavelength_range: tuple[float, float] = (300, 800),
        num_angles: int = 241,
        num_excitation_wavelengths: int = 500,
        cmap: str = "jet",
        export_excel: bool = False,
        savefig: Optional[str] = None,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> np.ndarray:
        """Heatmap of excitation intensity vs angle with fixed control n/k.

        Layer refractive indices are held fixed at ``lambda_control``
        while the excitation wavelength is swept in ``coh_tmm``.

        Parameters
        ----------
        name : str
            Registry key of the coverslip to analyse.
        lambda_control : float
            Fixed wavelength (nm) used for n/k look-up of all layers.
        angle_range : (min_deg, max_deg)
        excitation_wavelength_range : (min_nm, max_nm)
        num_angles, num_excitation_wavelengths : int
            Grid resolution.
        cmap : str
            Matplotlib colormap.
        export_excel : bool
            Export intensity grid to Excel.
        savefig : str or None
            Output figure path.
        s_weight : float
            Weight for s-polarisation.  Set to 0 to exclude it.
        p_weight : float
            Weight for p-polarisation.  Set to 0 to exclude it.

        Returns
        -------
        I_grid : numpy.ndarray, shape (num_excitation_wavelengths, num_angles)
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}'.")

        pol_lbl = self._pol_label(s_weight, p_weight)
        norm     = s_weight + p_weight

        reg_entry  = self.registry[name]
        e_prof     = reg_entry[0]
        materials  = reg_entry[4]

        angles                 = np.linspace(angle_range[0],                  angle_range[1],                  num_angles)
        excitation_wavelengths = np.linspace(excitation_wavelength_range[0],  excitation_wavelength_range[1],  num_excitation_wavelengths)
        I_grid                 = np.zeros((len(excitation_wavelengths), len(angles)), dtype=float)

        total_thickness = float(sum(e_prof[1:-1]))
        layer_idx, dpos = find_in_structure_with_inf(e_prof, total_thickness)

        # n/k fixed at lambda_control throughout the sweep
        n_profile_ctrl = (
            [self.n_super]
            + [nk_measured(lambda_control, m) for m in materials]
            + [self.n_subs]
        )

        for i, wl_exc in enumerate(excitation_wavelengths):
            print(
                f"  [{name}] λ_excitation = {wl_exc:.1f} nm  "
                f"(λ_control fixed @ {lambda_control} nm)  "
                f"({i + 1}/{len(excitation_wavelengths)})"
            )

            for j, ang in enumerate(angles):
                theta_rad = np.deg2rad(ang)
                val = 0.0
                for pol, weight in (("s", s_weight), ("p", p_weight)):
                    if weight == 0.0:
                        continue
                    coh  = coh_tmm(pol, n_profile_ctrl, e_prof, theta_rad, wl_exc)
                    data = position_resolved(layer_idx, dpos, coh)
                    if pol == "s":
                        val += (weight / norm) * float(np.abs(data["Ey"]) ** 2)
                    else:
                        val += (weight / norm) * float(
                            np.abs(data["Ex"]) ** 2 + np.abs(data["Ez"]) ** 2
                        )
                I_grid[i, j] = val

            row_max = I_grid[i].max()
            if row_max > 0:
                I_grid[i] /= row_max

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            I_grid,
            aspect="auto",
            origin="lower",
            extent=(angles[0], angles[-1],
                    excitation_wavelength_range[0], excitation_wavelength_range[1]),
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_xlabel("Incidence Angle (°)")
        ax.set_ylabel("Excitation Wavelength (nm)")
        ax.set_title(
            f"{name}: Intensity heatmap\n"
            f"λ_control = {lambda_control} nm (fixed),  "
            f"λ_excitation swept  ({pol_lbl}-pol)"
        )
        plt.colorbar(im, ax=ax, label="Normalised Intensity (a.u.)")
        plt.tight_layout()

        if savefig:
            fmt = savefig.rsplit(".", 1)[-1] if "." in savefig else "svg"
            path = r"code\Modular"
            savefig = os.path.join(path, savefig)
            plt.savefig(savefig, format=fmt)
            print(f"Figure saved to '{savefig}'.")

        plt.show()

        if export_excel:
            df = pd.DataFrame(
                I_grid,
                index=np.round(excitation_wavelengths, 3),
                columns=np.round(angles, 3),
            )
            df.index.name   = "Excitation_Wavelength_nm"
            df.columns.name = "Angle_deg"
            outname = (
                f"{name}_excitation_heatmap_fixedctrl_"
                f"ctrl{lambda_control:.0f}nm_"
                f"exc{excitation_wavelength_range[0]:.0f}to{excitation_wavelength_range[1]:.0f}nm_"
                f"{angle_range[0]:.0f}to{angle_range[1]:.0f}deg.xlsx"
            )
            df.to_excel(outname)
            print(f"Heatmap exported to '{outname}'.")

        return I_grid

    # -----------------------------------------------------------------------
    # Polarisation helpers — unified compute engines
    # -----------------------------------------------------------------------

    @staticmethod
    def _pol_label(s_weight: float, p_weight: float) -> str:
        """Return a compact polarisation label for titles and filenames.

        Examples
        --------
        ``s_weight=1, p_weight=0``  →  ``"s"``
        ``s_weight=0, p_weight=1``  →  ``"p"``
        ``s_weight=0.7, p_weight=0.3``  →  ``"s:0.70+p:0.30"``
        """
        if p_weight == 0.0:
            return "s"
        if s_weight == 0.0:
            return "p"
        norm = s_weight + p_weight
        return f"s{s_weight / norm:.2f}p{p_weight / norm:.2f}"

    def _calc_rta(
        self,
        n_complex: list,
        e_prof: list,
        lam_nm: float,
        theta_arr: np.ndarray,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute weighted R, T, A vs angle for the requested polarisation mix.

        Only TMM calls for polarisations with ``weight > 0`` are executed.
        Weights are normalised internally so that R + T + A = 1 is preserved.

        Parameters
        ----------
        n_complex : list of complex
            Complex refractive indices for the full stack (including inf caps).
        e_prof : list
            Thickness profile in nm (with ``inf`` caps).
        lam_nm : float
            Wavelength in nm passed to ``coh_tmm``.
        theta_arr : array_like
            Incidence angles in degrees.
        s_weight : float
            Weight for s-polarisation.  Pass 0 to skip TMM call entirely.
        p_weight : float
            Weight for p-polarisation.  Pass 0 to skip TMM call entirely.

        Returns
        -------
        R, T, A : three numpy.ndarray
            Reflectance, transmittance, and absorptance at each angle.
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")

        norm  = s_weight + p_weight
        n     = len(theta_arr)
        R_acc = np.zeros(n)
        T_acc = np.zeros(n)
        A_acc = np.zeros(n)

        for pol, weight in (("s", s_weight), ("p", p_weight)):
            if weight == 0.0:
                continue
            for i, th in enumerate(theta_arr):
                try:
                    res = coh_tmm(pol, n_complex, e_prof, np.radians(th), lam_nm)
                    R   = float(res["R"])
                    T   = float(res["T"])
                    R_acc[i] += weight * R
                    T_acc[i] += weight * T
                    A_acc[i] += weight * (1.0 - R - T)
                except Exception:
                    R_acc[i] = T_acc[i] = A_acc[i] = np.nan

        return R_acc / norm, T_acc / norm, A_acc / norm

    def _calc_efield(
        self,
        n_complex: list,
        e_prof: list,
        lam_nm: float,
        theta_deg: float,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
        n_pts_per_layer: int = 300,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute weighted |E|² depth profile for the requested polarisation mix.

        Only TMM calls for polarisations with ``weight > 0`` are executed.

        Parameters
        ----------
        n_complex : list of complex
            Complex refractive indices for the full stack.
        e_prof : list
            Thickness profile in nm (with ``inf`` caps).
        lam_nm : float
            Wavelength in nm.
        theta_deg : float
            Incidence angle in degrees.
        s_weight : float
            Weight for s-polarisation.  Pass 0 to skip it.
        p_weight : float
            Weight for p-polarisation.  Pass 0 to skip it.
        n_pts_per_layer : int
            Sampling density per finite layer.

        Returns
        -------
        z_nm : numpy.ndarray
            Depth from first interface in nm.
        I_combined : numpy.ndarray
            Weighted |E|² / |E_inc|² at each depth point.
        """
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("At least one of s_weight or p_weight must be > 0.")

        norm     = s_weight + p_weight
        th_rad   = np.radians(theta_deg)
        finite_d = list(e_prof[1:-1])

        # Build the (real_layer_index, z_local, z_global) table once —
        # it is identical for s and p so we only compute it once.
        layer_points: list[tuple[int, float, float]] = []
        z_offset = 0.0
        for layer_i, thick in enumerate(finite_d):
            real_layer = layer_i + 1
            for z in np.linspace(0.0, thick, n_pts_per_layer, endpoint=False):
                layer_points.append((real_layer, float(z), z_offset + float(z)))
            z_offset += thick

        z_arr = np.array([pt[2] for pt in layer_points])
        I_acc = np.zeros(len(layer_points))

        for pol, weight in (("s", s_weight), ("p", p_weight)):
            if weight == 0.0:
                continue
            res = coh_tmm(pol, n_complex, e_prof, th_rad, lam_nm)
            for idx, (real_layer, z_local, _) in enumerate(layer_points):
                pr = position_resolved(real_layer, z_local, res)
                if pol == "s":
                    I_acc[idx] += weight * float(np.abs(pr["Ey"]) ** 2)
                else:
                    I_acc[idx] += weight * float(
                        np.abs(pr["Ex"]) ** 2 + np.abs(pr["Ez"]) ** 2
                    )

        return z_arr, I_acc / norm

    @staticmethod
    def _critical_angle_deg(n_in: float, n_out: float) -> Optional[float]:
        """TIR critical angle in degrees, or None if no TIR is possible."""
        ratio = float(np.real(n_out)) / float(np.real(n_in))
        if ratio >= 1.0:
            return None
        return float(np.degrees(np.arcsin(ratio)))

    def _print_optical_thickness_table(
        self,
        materials: list[str],
        thicknesses: list[float],
        lambda_exc: float,
        lambda_ctrl: float,
    ) -> None:
        """Print a formatted optical-thickness table to stdout.

        Columns: layer index, physical thickness (nm), n at lambda_exc,
        optical thickness at lambda_exc, QWOT at lambda_exc, and the
        same three columns repeated for lambda_ctrl.

        Parameters
        ----------
        materials : list of str
            Material keys for each finite layer.
        thicknesses : list of float
            Physical layer thicknesses in nm.
        lambda_exc, lambda_ctrl : float
            Wavelengths in nm for the two columns.
        """
        hdr = (
            f"  {'Layer':<16} {'d (nm)':>8}  {'n_exc':>6}  "
            f"{'OT_exc':>10}  {'QWOT_exc':>9}  "
            f"{'n_ctrl':>6}  {'OT_ctrl':>10}  {'QWOT_ctrl':>9}"
        )
        sep = "─" * len(hdr)
        print(f"\n{sep}\n{hdr}\n{sep}")
        for i, (mat, d) in enumerate(zip(materials, thicknesses)):
            ne   = float(np.real(nk_measured(lambda_exc,  mat)))
            nc   = float(np.real(nk_measured(lambda_ctrl, mat)))
            ot_e = ne * d
            ot_c = nc * d
            qw_e = ot_e / (lambda_exc  / 4.0)
            qw_c = ot_c / (lambda_ctrl / 4.0)
            label = f"L{i + 1} ({mat})"
            print(
                f"  {label:<16} {d:>8.1f}  {ne:>6.3f}  "
                f"{ot_e:>10.2f}  {qw_e:>9.4f}  "
                f"{nc:>6.3f}  {ot_c:>10.2f}  {qw_c:>9.4f}"
            )
        print(sep)

    def plot_rta_efield_dual_wavelength(
        self,
        name: str,
        lambda_exc: float,
        lambda_ctrl: float,
        theta_efield: float,
        angle_range: tuple[float, float] = (0.0, 80.0),
        num_angles: int = 500,
        n_pts_per_layer: int = 300,
        s_weight: float = 1.0,
        p_weight: float = 0.0,
        print_optical_table: bool = True,
        color_exc: str = "#1f77b4",
        color_ctrl: str = "#000000",
        savefig: Optional[str] = None,
    ) -> None:
        """Plot R, T, A vs angle and |E|² depth profile at two wavelengths.

        The polarisation blend is controlled by ``s_weight`` and ``p_weight``.
        Setting a weight to 0 skips that polarisation entirely — ``coh_tmm``
        is never called for it.  Equal non-zero values give a 50/50 blend.
        All computation is routed through :meth:`_calc_rta` (for R, T, A)
        and :meth:`_calc_efield` (for the depth profile).

        Figure layout (3 rows × 2 columns):

        * Row 0 — Reflectance R vs angle  |  Transmittance T vs angle
        * Row 1 — Absorptance A vs angle  |  Energy conservation (R+T+A)
        * Row 2 — |E|² depth profile spanning both columns

        Parameters
        ----------
        name : str
            Registry key of the coverslip to analyse.
        lambda_exc : float
            Excitation wavelength in nm.
        lambda_ctrl : float
            Control wavelength in nm.
        theta_efield : float
            Incidence angle in degrees used for the E-field depth profile.
        angle_range : (min_deg, max_deg)
            Angular sweep limits.
        num_angles : int
            Number of angle samples.
        n_pts_per_layer : int
            E-field sampling density per layer.
        s_weight : float
            Weight for s-polarisation (TE).  Set to 0 to exclude.
        p_weight : float
            Weight for p-polarisation (TM).  Set to 0 to exclude.
        print_optical_table : bool
            If True, print the optical-thickness table to stdout.
        color_exc : str
            Matplotlib colour string for excitation-wavelength curves.
        color_ctrl : str
            Matplotlib colour string for control-wavelength curves.
        savefig : str or None
            If given, save the figure (format inferred from extension).
        """
        if s_weight + p_weight == 0.0:
            raise ValueError("s_weight and p_weight cannot both be zero.")
        if name not in self.registry:
            raise ValueError(f"No coverslip named '{name}' in the registry.")

        e_prof, _, _, thicknesses, materials = self.registry[name]
        plabel = self._pol_label(s_weight, p_weight)

        # ── Complex refractive-index profiles at each wavelength ─────────────
        def _n_complex(lam: float) -> list:
            return (
                [complex(self.n_super, 0.0)]
                + [complex(nk_measured(lam, m), excitation_coeff(lam, m)) for m in materials]
                + [complex(self.n_subs, 0.0)]
            )

        n_exc  = _n_complex(lambda_exc)
        n_ctrl = _n_complex(lambda_ctrl)

        # ── Optical-thickness table ──────────────────────────────────────────
        if print_optical_table:
            self._print_optical_thickness_table(materials, thicknesses, lambda_exc, lambda_ctrl)

        # ── Critical angles ──────────────────────────────────────────────────
        theta_c_exc  = self._critical_angle_deg(np.real(n_exc[0]),  np.real(n_exc[-1]))
        theta_c_ctrl = self._critical_angle_deg(np.real(n_ctrl[0]), np.real(n_ctrl[-1]))
        print(
            f"\nTIR critical angle — exc  ({lambda_exc:.0f} nm):  "
            f"{'N/A' if theta_c_exc  is None else f'{theta_c_exc:.2f}°'}"
        )
        print(
            f"TIR critical angle — ctrl ({lambda_ctrl:.0f} nm):  "
            f"{'N/A' if theta_c_ctrl is None else f'{theta_c_ctrl:.2f}°'}"
        )

        # ── Angular scan — R, T, A ───────────────────────────────────────────
        theta_arr = np.linspace(angle_range[0], angle_range[1], num_angles)
        print(
            f"\nAngular scan: {num_angles} angles "
            f"{angle_range[0]:.1f}° → {angle_range[1]:.1f}°  [{plabel}] …"
        )
        R_e, T_e, A_e = self._calc_rta(n_exc,  e_prof, lambda_exc,  theta_arr, s_weight, p_weight)
        R_c, T_c, A_c = self._calc_rta(n_ctrl, e_prof, lambda_ctrl, theta_arr, s_weight, p_weight)

        # ── E-field depth profiles ───────────────────────────────────────────
        print(f"E-field depth profiles at θ = {theta_efield}°  [{plabel}] …")
        z_e, I_e = self._calc_efield(n_exc,  e_prof, lambda_exc,  theta_efield, s_weight, p_weight, n_pts_per_layer)
        z_c, I_c = self._calc_efield(n_ctrl, e_prof, lambda_ctrl, theta_efield, s_weight, p_weight, n_pts_per_layer)

        # ── Figure ───────────────────────────────────────────────────────────
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"{name} — TIRF multilayer optical scan  [{plabel}]\n"
            f"λ_exc = {lambda_exc:.0f} nm   λ_ctrl = {lambda_ctrl:.0f} nm",
            fontsize=13, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        def _vlines(ax):
            if theta_c_exc is not None:
                ax.axvline(theta_c_exc,  color=color_exc,  lw=0.8, ls=":", alpha=0.7)
            if theta_c_ctrl is not None:
                ax.axvline(theta_c_ctrl, color=color_ctrl, lw=0.8, ls=":", alpha=0.7)
            ax.axvline(theta_efield, color="k", lw=0.8, ls=":",
                       label=f"θ_TIRF = {theta_efield}°")

        # — Row 0: R and T —
        ax_R = fig.add_subplot(gs[0, 0])
        ax_R.plot(theta_arr, R_e, color=color_exc,  lw=1.5, label=f"λ = {lambda_exc:.0f} nm")
        ax_R.plot(theta_arr, R_c, color=color_ctrl, lw=1.5, label=f"λ = {lambda_ctrl:.0f} nm")
        _vlines(ax_R)
        ax_R.set(xlabel="θ (°)", ylabel="Reflectance R",
                 title=f"Reflectance vs angle  [{plabel}]",
                 xlim=angle_range, ylim=(0.0, 1.05))
        ax_R.legend(fontsize=7, ncol=2)

        ax_T = fig.add_subplot(gs[0, 1])
        ax_T.plot(theta_arr, T_e, color=color_exc,  lw=1.5, label=f"λ = {lambda_exc:.0f} nm")
        ax_T.plot(theta_arr, T_c, color=color_ctrl, lw=1.5, label=f"λ = {lambda_ctrl:.0f} nm")
        _vlines(ax_T)
        ax_T.set(xlabel="θ (°)", ylabel="Transmittance T",
                 title=f"Transmittance vs angle  [{plabel}]",
                 xlim=angle_range, ylim=(-0.02, 1.05))
        ax_T.legend(fontsize=7, ncol=2)

        # — Row 1: A and energy conservation —
        ax_A = fig.add_subplot(gs[1, 0])
        ax_A.plot(theta_arr, A_e, color=color_exc,  lw=1.5, label=f"λ = {lambda_exc:.0f} nm")
        ax_A.plot(theta_arr, A_c, color=color_ctrl, lw=1.5, label=f"λ = {lambda_ctrl:.0f} nm")
        _vlines(ax_A)
        ax_A.set(xlabel="θ (°)", ylabel="Absorptance A",
                 title=f"Absorptance vs angle  [{plabel}]", xlim=angle_range)
        ax_A.legend(fontsize=7, ncol=2)

        ax_chk = fig.add_subplot(gs[1, 1])
        ax_chk.plot(theta_arr, R_e + T_e + A_e, color=color_exc,  lw=1.5, label=f"λ = {lambda_exc:.0f} nm")
        ax_chk.plot(theta_arr, R_c + T_c + A_c, color=color_ctrl, lw=1.5, label=f"λ = {lambda_ctrl:.0f} nm")
        ax_chk.axhline(1.0, color="k", lw=0.8, ls="--")
        ax_chk.set(xlabel="θ (°)", ylabel="R + T + A",
                   title="Energy conservation check",
                   xlim=angle_range, ylim=(0.98, 1.02))
        ax_chk.legend(fontsize=7)

        # — Row 2: |E|² depth profile —
        ax_Ef = fig.add_subplot(gs[2, :])
        total_d = float(sum(e_prof[1:-1]))
        ymax    = max(I_e.max(), I_c.max()) * 1.15

        boundaries   = [0.0] + list(np.cumsum(thicknesses))
        pastel_cycle = plt.cm.Pastel1.colors
        for i, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ax_Ef.axvspan(x0, x1, alpha=0.25,
                          color=pastel_cycle[i % len(pastel_cycle)], zorder=0)
            ax_Ef.text((x0 + x1) / 2.0, ymax * 0.92,
                       f"L{i + 1}\n({materials[i]})",
                       ha="center", va="top", fontsize=6)
        for b in boundaries:
            ax_Ef.axvline(b, color="gray", lw=0.5, ls="--", zorder=1)

        ax_Ef.plot(z_e, I_e, color=color_exc,  lw=1.8, label=f"λ = {lambda_exc:.0f} nm")
        ax_Ef.plot(z_c, I_c, color=color_ctrl, lw=1.8, label=f"λ = {lambda_ctrl:.0f} nm")
        ax_Ef.set(
            xlabel="Physical depth from first interface (nm)",
            ylabel=r"$|E|^2 / |E_{inc}|^2$",
            title=f"Electric-field intensity profile  (θ = {theta_efield}°,  {plabel})",
            xlim=(0.0, total_d),
            ylim=(0.0, ymax),
        )
        ax_Ef.legend(fontsize=8, ncol=4, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if savefig:
            fmt = savefig.rsplit(".", 1)[-1] if "." in savefig else "png"
            plt.savefig(savefig, format=fmt, dpi=180, bbox_inches="tight")
            print(f"\nFigure saved → '{savefig}'.")

        plt.show()
# ── paste from here ──────────────────────────────────────────────────────────
 
    def virus_on_bsw(
        self,
        name: str,
        angle_range: tuple[float, float] = (60.0, 82.0),
        num_angles: int = 400,
        # particle geometry
        R_core_nm: float = 0.0,
        n_core: float = 1.50,
        R_shell_nm: float = 70.0,
        n_shell: float = 1.46,
        # medium
        n_medium: float = 1.35,
        # MAAPS operating angles — used to compute ratio shift
        theta_maaps: tuple[float, float] = (63.5, 75.3),
        # polarisation
        s_weight: float = 1.0,
        p_weight: float = 0.0,
        # output
        savefig: Optional[str] = None,
        export_excel: bool = False,
        print_summary: bool = True,
    ) -> dict:
        """Model the effect of a spherical nano-particle (virus or bead) on the BSW.
 
        The particle sitting on the multilayer surface is approximated as a
        thin homogeneous effective-medium slab inserted between the last finite
        layer and the semi-infinite aqueous substrate.  This is the standard
        biosensor perturbation model valid when the particle diameter is much
        smaller than the BSW lateral coherence length.
 
        Two sub-models are supported depending on *R_core_nm*:
 
        **Homogeneous sphere** (``R_core_nm == 0`` or ``R_core_nm == R_shell_nm``)
            The particle is treated as a single-material sphere (polystyrene
            bead, for example).  Maxwell-Garnett mixing gives::
 
                f     = (4/3 π R³) / (π R² · 2R)   = 2/3
                n_eff = Maxwell-Garnett(n_shell, n_medium, f=2/3)
 
            and d_eff = 2R (sphere diameter as slab thickness).
 
        **Core–shell sphere** (``0 < R_core_nm < R_shell_nm``)
            Applies the Bruggeman / nested Maxwell-Garnett model:
 
            1. Mix core material into shell material::
 
                   f_c   = (R_core / R_shell)³
                   n_cs  = Maxwell-Garnett(n_core, n_shell, f=f_c)
 
            2. Mix the resulting core–shell particle into the medium::
 
                   f_p   = 2/3  (sphere-in-slab, same as above)
                   n_eff = Maxwell-Garnett(n_cs, n_medium, f=f_p)
 
            and d_eff = 2 · R_shell.
 
        The effective slab is appended to the registered stack (without
        mutating the stored registry entry) and a full TMM intensity sweep is
        recomputed.  The bare-stack sweep is computed from the unmodified
        entry for direct comparison.
 
        Parameters
        ----------
        name : str
            Registry key of the coverslip (e.g. ``"MAAPS488nm"``).
        angle_range : (min_deg, max_deg)
            Angular range for the sweep.
        num_angles : int
            Number of angles to evaluate.
        R_core_nm : float
            Core radius in nm.  Set to ``0`` for a homogeneous sphere.
        n_core : float
            Refractive index of the core material.
        R_shell_nm : float
            Outer (shell) radius of the particle in nm.
        n_shell : float
            Refractive index of the shell material (or the whole sphere
            when ``R_core_nm == 0``).
        n_medium : float
            Refractive index of the surrounding aqueous medium.
        theta_maaps : (theta1, theta2)
            The two MAAPS operating angles (degrees).  Used only to evaluate
            the MAAPS ratio perturbation; does not affect the sweep.
        s_weight : float
            Weight for s-polarisation (TE).
        p_weight : float
            Weight for p-polarisation (TM).
        savefig : str or None
            If given, save the output figure (format from extension).
        export_excel : bool
            If True, export angular curves to an Excel file.
        print_summary : bool
            If True, print a summary table to stdout.
 
        Returns
        -------
        dict
            Keys described in the module docstring above.
 
        Raises
        ------
        ValueError
            If the coverslip name is not in the registry, or weights are
            both zero, or R_core_nm > R_shell_nm.
        """
        # ── Input validation ────────────────────────────────────────────────
        if name not in self.registry:
            raise ValueError(f"Coverslip '{name}' not found in registry.")
        if s_weight == 0.0 and p_weight == 0.0:
            raise ValueError("s_weight and p_weight cannot both be zero.")
        if R_core_nm > R_shell_nm:
            raise ValueError(
                f"R_core_nm ({R_core_nm}) must be ≤ R_shell_nm ({R_shell_nm})."
            )
        if R_shell_nm <= 0.0:
            raise ValueError("R_shell_nm must be > 0.")
 
        # ── Retrieve bare stack ─────────────────────────────────────────────
        e_prof_bare, n_prof_bare, k_prof_bare, thicknesses, materials = (
            self.registry[name]
        )
        # Build complex n list for bare stack (re-evaluate at self.wavelength)
        n_complex_bare = [
            complex(nk_measured(self.wavelength, m), excitation_coeff(self.wavelength, m))
            for m in materials
        ]
        n_complex_bare = (
            [complex(self.n_super, 0.0)]
            + n_complex_bare
            + [complex(n_medium, 0.0)]
        )
 
        # ── Effective medium calculation ────────────────────────────────────
        # Maxwell-Garnett mixing: inclusions (eps_i) in host (eps_h)
        # Returns the effective real refractive index.
        def _maxwell_garnett(n_inclusion: float, n_host: float, f: float) -> float:
            """Real-valued Maxwell-Garnett effective index."""
            eps_i = n_inclusion ** 2
            eps_h = n_host ** 2
            eps_eff = eps_h * (
                (eps_i + 2 * eps_h + 2 * f * (eps_i - eps_h))
                / (eps_i + 2 * eps_h - f * (eps_i - eps_h))
            )
            return float(np.sqrt(np.abs(eps_eff)))
 
        # Volume-filling fraction of a sphere inscribed in a slab of thickness 2R
        # (the slab replaces the sphere's bounding box):
        #   f = V_sphere / V_slab = (4/3 π R³) / (π R² · 2R) = 2/3
        F_SPHERE_IN_SLAB = 2.0 / 3.0
 
        homogeneous = (R_core_nm == 0.0) or (R_core_nm == R_shell_nm)
 
        if homogeneous:
            # Single-material sphere (bead)
            n_eff = _maxwell_garnett(n_shell, n_medium, F_SPHERE_IN_SLAB)
            d_eff = 2.0 * R_shell_nm
            geometry_label = (
                f"Homogeneous sphere  R={R_shell_nm:.0f} nm  n={n_shell:.4f}"
            )
        else:
            # Core–shell virus
            # Step 1: effective index of the composite core–shell particle
            f_core = (R_core_nm / R_shell_nm) ** 3   # volume fraction of core in shell
            n_core_shell = _maxwell_garnett(n_core, n_shell, f_core)
            # Step 2: mix that composite particle into the medium
            n_eff = _maxwell_garnett(n_core_shell, n_medium, F_SPHERE_IN_SLAB)
            d_eff = 2.0 * R_shell_nm
            geometry_label = (
                f"Core–shell  R_core={R_core_nm:.0f} nm (n={n_core:.4f})  "
                f"R_shell={R_shell_nm:.0f} nm (n={n_shell:.4f})"
            )
 
        # ── Build perturbed stack (insert slab before the semi-inf substrate) ─
        # e_prof layout: [inf, d1, d2, ..., dn, inf]
        # We insert the effective slab just before the last inf:
        e_prof_particle = list(e_prof_bare[:-1]) + [d_eff, np.inf]
        n_complex_particle = list(n_complex_bare[:-1]) + [
            complex(n_eff, 0.0),
            complex(n_medium, 0.0),
        ]
 
        # ── Angular sweeps ──────────────────────────────────────────────────
        angles = np.linspace(angle_range[0], angle_range[1], num_angles)
 
        _, I_bare = self.compute_nominal_intensity(
            e_prof_bare, n_complex_bare, angles,
            s_weight=s_weight, p_weight=p_weight,
        )
        _, I_particle = self.compute_nominal_intensity(
            e_prof_particle, n_complex_particle, angles,
            s_weight=s_weight, p_weight=p_weight,
        )
 
        # Relative intensity change
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_I_rel = np.where(
                I_bare > 0, (I_particle - I_bare) / I_bare, np.nan
            )
 
        # ── BSW resonance angles ────────────────────────────────────────────
        def _peak_angle(I_arr: np.ndarray) -> Optional[float]:
            peaks, props = find_peaks(I_arr, prominence=0.05)
            if len(peaks) == 0:
                return None
            # Return the most prominent peak
            best = peaks[np.argmax(props["prominences"])]
            return float(angles[best])
 
        theta_peak_bare     = _peak_angle(I_bare)
        theta_peak_particle = _peak_angle(I_particle)
        delta_theta = (
            (theta_peak_particle - theta_peak_bare)
            if (theta_peak_bare is not None and theta_peak_particle is not None)
            else None
        )
 
        # ── MAAPS ratio perturbation ────────────────────────────────────────
        def _interp(I_arr: np.ndarray, theta_target: float) -> float:
            return float(np.interp(theta_target, angles, I_arr))
 
        th1, th2 = theta_maaps
        I_bare_1     = _interp(I_bare,     th1)
        I_bare_2     = _interp(I_bare,     th2)
        I_particle_1 = _interp(I_particle, th1)
        I_particle_2 = _interp(I_particle, th2)
 
        ratio_bare     = I_bare_2     / I_bare_1     if I_bare_1     > 0 else np.nan
        ratio_particle = I_particle_2 / I_particle_1 if I_particle_1 > 0 else np.nan
        delta_ratio    = (ratio_particle / ratio_bare - 1.0) if (
            np.isfinite(ratio_bare) and ratio_bare > 0
        ) else np.nan
 
        I_ratio_theta1 = I_particle_1 / I_bare_1 if I_bare_1 > 0 else np.nan
        I_ratio_theta2 = I_particle_2 / I_bare_2 if I_bare_2 > 0 else np.nan
 
        # ── Console summary ─────────────────────────────────────────────────
        if print_summary:
            pol_lbl = self._pol_label(s_weight, p_weight)
            sep = "─" * 60
            print(f"\n{sep}")
            print(f"  virus_on_bsw — {name}  ({pol_lbl}-pol, λ={self.wavelength} nm)")
            print(f"  {geometry_label}")
            print(f"  Effective slab:  d_eff = {d_eff:.1f} nm   n_eff = {n_eff:.5f}")
            print(sep)
            print(f"  BSW peak (bare stack)     : "
                  f"{'N/A' if theta_peak_bare is None else f'{theta_peak_bare:.3f}°'}")
            print(f"  BSW peak (with particle)  : "
                  f"{'N/A' if theta_peak_particle is None else f'{theta_peak_particle:.3f}°'}")
            print(f"  Δθ_resonance              : "
                  f"{'N/A' if delta_theta is None else f'{delta_theta:+.4f}°'}")
            print(sep)
            print(f"  MAAPS angles: θ₁={th1}°   θ₂={th2}°")
            print(f"  I(θ₁) bare / particle     : {I_bare_1:.4f}  /  {I_particle_1:.4f}   "
                  f"→ ratio {I_ratio_theta1:.4f}")
            print(f"  I(θ₂) bare / particle     : {I_bare_2:.4f}  /  {I_particle_2:.4f}   "
                  f"→ ratio {I_ratio_theta2:.4f}")
            print(f"  MAAPS ratio bare / particle: {ratio_bare:.4f}  /  {ratio_particle:.4f}")
            print(f"  Δ(MAAPS ratio) / ratio_bare: {delta_ratio:+.4f}  "
                  f"({delta_ratio * 100:+.2f} %)")
            print(sep + "\n")
 
        # ── Figure ──────────────────────────────────────────────────────────
        pol_lbl  = self._pol_label(s_weight, p_weight)
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 9),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        fig.suptitle(
            f"Particle on BSW — {name}  [{pol_lbl}-pol, λ={self.wavelength} nm]\n"
            f"{geometry_label}",
            fontsize=11, fontweight="bold",
        )
 
        # — Top panel: intensity curves —
        ax0 = axes[0]
        ax0.plot(angles, I_bare,     lw=2,   color="#1f77b4", label="Bare stack")
        ax0.plot(angles, I_particle, lw=2,   color="#d62728",
                 linestyle="--", label="With particle")
 
        # Mark MAAPS operating angles
        for th, ls in zip(theta_maaps, [":", "-."]):
            ax0.axvline(th, color="grey", lw=0.9, ls=ls, alpha=0.7,
                        label=f"θ_MAAPS = {th}°")
 
        # Mark resonance peaks
        for th_pk, col, lbl in [
            (theta_peak_bare,     "#1f77b4", f"Peak bare {theta_peak_bare:.2f}°"
             if theta_peak_bare else ""),
            (theta_peak_particle, "#d62728", f"Peak +particle {theta_peak_particle:.2f}°"
             if theta_peak_particle else ""),
        ]:
            if th_pk is not None:
                ax0.axvline(th_pk, color=col, lw=1.2, ls="-", alpha=0.5)
                ax0.annotate(
                    f"{th_pk:.2f}°",
                    xy=(th_pk, ax0.get_ylim()[1] if ax0.get_ylim()[1] > 0 else 1),
                    xytext=(4, -20), textcoords="offset points",
                    fontsize=8, color=col,
                )
 
        ax0.set_ylabel(r"$|E|^2 / |E_{\rm inc}|^2$  (a.u.)")
        ax0.set_xlabel("Incidence angle (°)")
        ax0.set_xlim(angle_range)
        ax0.legend(fontsize=8, loc="upper left")
        ax0.set_title("Intensity at stack surface vs angle", fontsize=10)
 
        # — Bottom panel: relative intensity change —
        ax1 = axes[1]
        ax1.axhline(0, color="k", lw=0.8, ls="--")
        ax1.fill_between(
            angles, delta_I_rel * 100,
            where=delta_I_rel > 0, alpha=0.35, color="#2ca02c", label="Enhancement"
        )
        ax1.fill_between(
            angles, delta_I_rel * 100,
            where=delta_I_rel < 0, alpha=0.35, color="#d62728", label="Suppression"
        )
        ax1.plot(angles, delta_I_rel * 100, lw=1.5, color="k")
        for th in theta_maaps:
            ax1.axvline(th, color="grey", lw=0.9, ls=":", alpha=0.7)
        ax1.set_ylabel(r"$\Delta I / I_{\rm bare}$  (%)")
        ax1.set_xlabel("Incidence angle (°)")
        ax1.set_xlim(angle_range)
        ax1.legend(fontsize=8, loc="upper left")
        ax1.set_title("Relative intensity change due to particle", fontsize=10)
 
        # Annotate MAAPS ratio shift
        ax1.text(
            0.98, 0.05,
            f"Δ(MAAPS ratio) = {delta_ratio * 100:+.2f} %\n"
            f"Δθ_BSW = {delta_theta:+.4f}°" if delta_theta is not None
            else f"Δ(MAAPS ratio) = {delta_ratio * 100:+.2f} %\nΔθ_BSW = N/A",
            transform=ax1.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8),
        )
 
        plt.tight_layout(rect=[0, 0, 1, 0.93])
 
        if savefig:
            fmt = savefig.rsplit(".", 1)[-1] if "." in savefig else "png"
            plt.savefig(savefig, format=fmt, dpi=180, bbox_inches="tight")
            print(f"Figure saved → '{savefig}'.")
 
        # ── Optional Excel export ────────────────────────────────────────────
        if export_excel:
            outname = f"{name}_virus_on_bsw_{R_shell_nm:.0f}nm_{pol_lbl}.xlsx"
            pd.DataFrame({
                "Angle (deg)":          angles,
                "I_bare (a.u.)":        I_bare,
                "I_particle (a.u.)":    I_particle,
                "Delta_I_rel (frac)":   delta_I_rel,
                "Delta_I_rel (%)":      delta_I_rel * 100,
            }).to_excel(outname, index=False)
            print(f"Data exported → '{outname}'.")
 
        plt.show()
 
        return {
            "angles":               angles,
            "I_bare":               I_bare,
            "I_particle":           I_particle,
            "delta_I_rel":          delta_I_rel,
            "theta_peak_bare":      theta_peak_bare,
            "theta_peak_particle":  theta_peak_particle,
            "delta_theta":          delta_theta,
            "d_eff_nm":             d_eff,
            "n_eff":                n_eff,
            "I_ratio_theta1":       I_ratio_theta1,
            "I_ratio_theta2":       I_ratio_theta2,
            "ratio_bare":           ratio_bare,
            "ratio_particle":       ratio_particle,
            "delta_ratio":          delta_ratio,
        }

