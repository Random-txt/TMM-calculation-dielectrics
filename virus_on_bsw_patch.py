"""
virus_on_bsw_patch.py
=====================
Drop-in addition for ``CoverslipRegistry``.

Paste the ``virus_on_bsw`` method body (everything between the dashed
lines) into ``coverslip_registry.py``, just before the line::

    # -----------------------------------------------------------------------
    # Polarisation helpers — unified compute engines
    # -----------------------------------------------------------------------

The method requires no new imports — everything it uses (numpy, matplotlib,
pandas, scipy.signal.find_peaks, tmm.coh_tmm) is already imported at the
top of coverslip_registry.py.

Usage example (mirrors the style of existing callers in run.py)
---------------------------------------------------------------
    reg = build_registry(wavelength=488, divergence=2, n_super=1.529)

    # HIV Gag VLP — RNA/protein core + lipid bilayer shell
    results = reg.virus_on_bsw(
        name        = "MAAPS488nm",
        angle_range = (60.0, 82.0),
        num_angles  = 300,
        # --- particle geometry ---
        R_core_nm   = 60.0,    # RNA+protein core radius
        n_core      = 1.50,    # ~protein/RNA mix
        R_shell_nm  = 70.0,    # outer VLP radius (core + lipid bilayer ~4 nm each side)
        n_shell     = 1.46,    # lipid bilayer
        # --- aqueous medium ---
        n_medium    = 1.35,
        # --- polarisation ---
        s_weight    = 0.2,
        p_weight    = 0.8,
        # --- output ---
        savefig     = "vlp_on_bsw.png",
        export_excel= True,
    )

    # Polystyrene bead — homogeneous sphere (set R_core_nm = 0 or equal to R_shell_nm)
    results = reg.virus_on_bsw(
        name        = "MAAPS488nm",
        R_core_nm   = 0.0,     # no core → purely shell (Maxwell-Garnett with one shell)
        n_core      = 1.59,    # unused when R_core_nm = 0
        R_shell_nm  = 50.0,    # bead radius
        n_shell     = 1.59,    # polystyrene
        n_medium    = 1.35,
    )

The method returns a dict with keys:
    "angles"            – angular axis (degrees)
    "I_bare"            – intensity vs angle for bare coverslip
    "I_particle"        – intensity vs angle with particle layer added
    "delta_I_rel"       – (I_particle − I_bare) / I_bare   (relative change)
    "theta_peak_bare"   – BSW resonance angle, bare stack (degrees)
    "theta_peak_particle" – BSW resonance angle, with particle (degrees)
    "delta_theta"       – resonance angle shift (degrees)
    "d_eff_nm"          – effective slab thickness used (nm)
    "n_eff"             – effective slab refractive index
    "I_ratio_theta1"    – I_particle / I_bare evaluated at theta_MAAPS[0]
    "I_ratio_theta2"    – I_particle / I_bare evaluated at theta_MAAPS[1]
    "delta_ratio"       – change in MAAPS ratio  R_particle/R_bare − 1
"""
import typing
from typing import Optional
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

# ── end of paste ─────────────────────────────────────────────────────────────
