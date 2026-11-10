"""
run_simulation.py
-----------------
Entry-point script for MAAPS / Fresnel multilayer simulations.

Adjust the sections below to enable or disable specific analyses.
All heavy computation lives in `coverslip_registry.py`; this file
only wires together coverslip definitions and analysis calls.

Polarisation is now controlled per-call via ``s_weight`` and ``p_weight``
arguments rather than a single ``pol`` string.  The weights are normalised
internally so that any non-negative pair is valid:

    s_weight=1.0, p_weight=0.0   →  pure s-pol  (TE)
    s_weight=0.0, p_weight=1.0   →  pure p-pol  (TM)
    s_weight=1.0, p_weight=1.0   →  equal 50/50 blend
    s_weight=0.7, p_weight=0.3   →  70 % s, 30 % p

Layout
------
1.  Shared coverslip definitions (helper function)
2.  Wavelength-sweep loop  – intensity vs angle for multiple λ
3.  One-off analyses       – single-wavelength, single coverslip calls

Run with:
    python run_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt

from coverslip_registry import CoverslipRegistry

# ---------------------------------------------------------------------------
# Shared layer stacks
# ---------------------------------------------------------------------------
# Thickness and material lists for each coverslip design.
# Ordered from superstrate (objective side) to substrate (sample side).

STACKS = {
    # --- MAAPS dual-angle designs ---
    "MAAPS488nm": dict(
        thicknesses=[307.860, 189.670, 62.031, 287.145, 145.724, 550.688, 20.0, 20.0],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    "MAAPS532nm": dict(
        thicknesses=[346.633, 211.267, 69.762, 322.679, 162.538, 593.278, 25.0, 20.0],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    "MAAPS532nm_lipid_mgf2": dict(
        thicknesses=[346.633, 211.267, 69.762, 322.679, 162.538, 593.278, 25.0, 20.0, 5.0, 25.0],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "lipid", "mgf2"],
    ),
    "MAAPS532nm_air": dict(
    thicknesses=[1000, 346.633, 211.267, 69.762, 322.679, 162.538, 593.278, 25.0, 20.0],
    materials=["air", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    # MAAPS488 with a 170 µm borosilicate substrate prepended
    "MAAPS488nm_glass": dict(
        thicknesses=[170_000, 307.860, 189.670, 62.031, 287.145, 145.724, 550.688, 20.0, 20.0],
        materials=["borosilicate", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    # Inverted MAAPS stacks (substrate → superstrate layer order)
    "MAAPS488nm_inverted": dict(
        thicknesses=[20.0, 20.0, 550.688, 145.724, 287.145, 62.031, 189.670, 307.860],
        materials=["sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5"],
    ),
    "MAAPS532nm_inverted": dict(
        thicknesses=[25.0, 20.0, 593.278, 162.538, 322.679, 69.762, 211.267, 346.633],
        materials=["sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5"],
    ),
    # MAAPS488 with alternative ('c') material indices for control comparison
    "MAAPS488nm_bead": dict(
        thicknesses=[307.860, 189.670, 62.031, 287.145, 145.724, 550.688, 20.0, 20.0, 20],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "polystyrene"],
    ),
    
    "MAAPS488nm_lipid": dict(
        thicknesses=[307.860, 189.670, 62.031, 287.145, 145.724, 550.688, 20.0, 20.0, 5],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "lipid"],
    ),
    "MAAPS532nm_at_ctrl": dict(
        thicknesses=[346.633, 211.267, 69.762, 322.679, 162.538, 593.278, 25.0, 20.0],
        materials=["nb2o5_622", "sio2_622", "nb2o5_622", "sio2_622", "nb2o5_622", "sio2_622", "nb2o5_622", "sio2_622"],
    ),

    # --- Experimental / reference designs ---
    "str535": dict(
        thicknesses=[154.053, 289.877, 157.803, 289.159, 155.483, 456.314, 15.452, 19.956],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),

    "str510": dict(
        thicknesses=[137.82, 278.003, 134.042, 266.99, 135.049, 309.774, 15.229, 19.933],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    )
    ,

    # --- SiO2-SiOx series (26-chamber deposition) ---
    "SiO2-SiOx-8-layer": dict(
        thicknesses=[108.670, 315.326, 99.265, 316.258, 82.570, 347.019, 5.602, 36.517],
        materials=["nb2o5-26", "sio2-26", "nb2o5-26", "sio2-26",
                   "nb2o5-26", "sio2-26", "nb2o5-26", "sio2-26"],
    ),
    "SiO2-SiOx-7-layer": dict(
        thicknesses=[333.537, 80.872, 326.192, 84.315, 372.863, 5.602, 36.517],
        materials=["sio2-26", "nb2o5-26", "sio2-26", "nb2o5-26",
                   "sio2-26", "nb2o5-26", "sio2-26"],
    ),
    "SiO2-SiOx-6-layer": dict(
        thicknesses=[79.901, 323.553, 84.139, 373.278, 5.602, 36.517],
        materials=["nb2o5-26", "sio2-26", "nb2o5-26", "sio2-26", "nb2o5-26", "sio2-26"],
    ),

    # --- Simple reference coverslips ---
    "Polystyrene": dict(
        thicknesses=[100],
        materials=["polystyrene"],
    ),
    "Glass_thin": dict(
        thicknesses=[1000, 0],
        materials=["air", "borosilicate"],
    ),
    "Glass_bead": dict(
        thicknesses=[1700, 100],
        materials=["polystyrene", "borosilicate"],
    ),
    "DR": dict(
        thicknesses=[97.619, 530.639, 20, 83.151, 29.088, 273.473, 20, 30.237, 60.254, 242.699, 101.807, 91.612, 26.489, 20],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    "NIS-SA240416-045L13CS151": dict(
        thicknesses=[155.3, 79.5, 155.3, 79.5, 155.3, 79.5, 155.3, 79.5, 155.3, 79.5, 155.3, 141.0, 98.1],
        materials=["sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"],
    ),
    "Fabien_optimized_P_1": dict(
        thicknesses=[117.233, 257.782, 73.471, 270.606, 74.599, 273.131, 74.788, 273.824, 74.808, 247.042, 74.801, 247.120, 74.794, 274.150, 17.013, 24.060],
        materials=["nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2"]

    ),
    "Fabien_optimized_P_1_inv": dict(
        thicknesses=[24.060, 17.013, 274.150, 74.794, 247.120, 74.801, 247.042, 74.808, 273.824, 74.788, 273.131, 74.599, 270.606, 73.471, 257.782, 117.233],
        materials=["sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5", "sio2", "nb2o5"]
    ),
    "SiO2-SiOx_og": dict(
        thicknesses = [409.704, 61.204, 406.791, 60.565, 394.443, 5.583, 36.558],
        materials = ['sio2','nb2o5','sio2','nb2o5','sio2','nb2o5','sio2']
    ),
    "SiO2-SiOx_og_inv": dict(
        thicknesses = [36.558, 5.583, 394.443, 60.565, 406.791, 61.204, 409.704],
        materials = ['sio2','nb2o5','sio2','nb2o5','sio2','nb2o5','sio2']
    ),

}


def build_registry(wavelength: float, pol: str = "s", divergence: float = 0.0, n_super: float = 1.518, n_subs: float = 1.0) -> CoverslipRegistry:
    """Instantiate a :class:`CoverslipRegistry` and register all standard stacks.

    Parameters
    ----------
    wavelength : float
        Excitation wavelength in nm.
    pol : str
        Default polarisation stored on the registry ('s' or 'p').
        Note: individual analysis calls now accept ``s_weight`` / ``p_weight``
        to control the polarisation blend per call.  This argument only sets
        the registry-level fallback used by ``compute_electric_field``.
    divergence : float
        Beam divergence FWHM in degrees (0 = plane wave).

    Returns
    -------
    CoverslipRegistry
        Fully populated registry ready for analysis.
    """
    registry = CoverslipRegistry(
        wavelength=wavelength,
        n_super=n_super,
        n_subs=n_subs,
        div_at_zero=divergence,
        max_div=divergence,
        exponent=2,
        skew=0.0,
        pol=pol,
    )
    for name, kwargs in STACKS.items():
        registry.add_coverslip(name, **kwargs)
    return registry


# ---------------------------------------------------------------------------
# Convenience helper: convert a 's'/'p'/'both' string to weight kwargs
# ---------------------------------------------------------------------------

def pol_weights(pol: str, tm_fraction: float = 0.5) -> dict:
    """Return ``{"s_weight": ..., "p_weight": ...}`` from a polarisation string.

    Parameters
    ----------
    pol : {'s', 'p', 'both'}
        'both' produces an equal-weight blend.
    tm_fraction : float
        Only used when *pol* is ``'both'``.  Sets the p-weight fraction
        (e.g. 0.5 → equal blend, 0.3 → 30 % p / 70 % s).

    Examples
    --------
    >>> reg.plot_coverslips(["DR"], **pol_weights("s"))
    >>> reg.plot_coverslips(["DR"], **pol_weights("both", tm_fraction=0.3))
    """
    if pol == "s":
        return {"s_weight": 1.0, "p_weight": 0.0}
    if pol == "p":
        return {"s_weight": 0.0, "p_weight": 1.0}
    if pol == "both":
        return {"s_weight": 1.0 - tm_fraction, "p_weight": tm_fraction}
    raise ValueError(f"pol must be 's', 'p', or 'both'; got {pol!r}")


# ===========================================================================
# 1.  Wavelength sweep — intensity vs angle at each λ
# ===========================================================================
# Uncomment and customise this block to sweep over a wavelength range.
# Pass s_weight / p_weight to plot_coverslips to select polarisation.

# S_WEIGHT, P_WEIGHT = 1.0, 0.0   # pure s-pol
# lambda_wavelength = np.round(np.linspace(488, 648, 50)).astype(int)
# for wave in lambda_wavelength:
#     reg = build_registry(wavelength=wave)
#     reg.plot_coverslips(["MAAPS488nm"], wavelength=wave,
#                         s_weight=S_WEIGHT, p_weight=P_WEIGHT)
# plt.show()


# ===========================================================================
# 2.  Single-wavelength analyses
# ===========================================================================   

if __name__ == "__main__":

    # ── Polarisation weights ─────────────────────────────────────────────────
    # Set s_weight / p_weight here and pass them to every call below.
    # Examples:
    #   pure s-pol  →  s_weight=1.0, p_weight=0.0
    #   pure p-pol  →  s_weight=0.0, p_weight=1.0
    #   50/50 blend →  s_weight=1.0, p_weight=1.0
    #   70/30 blend →  s_weight=0.7, p_weight=0.3
    # S_WEIGHT = 0.2
    # P_WEIGHT = 0.80

    # Loop over different S and P weight combinations from pure S to pure P. 1.0,   0.75, 0.5, 0.25, 0.0 and 0.0, 0.25, 0.5, 0.75, 1.0
    for wavelength in [488]:
        reg = build_registry(wavelength=wavelength, divergence=3, n_super=1.523, n_subs=1.35)
        for s in np.linspace(1.0, 0.0, 6):
            p = 1.0 - s
            S_WEIGHT, P_WEIGHT = s, p
            print(f"Running with S_WEIGHT={S_WEIGHT}, P_WEIGHT={P_WEIGHT}")
            # S_WEIGHT, P_WEIGHT = 0.2, 0.8
            reg.plot_coverslips(["MAAPS488nm_bead"], wavelength=wavelength, s_weight=S_WEIGHT, p_weight=P_WEIGHT, angles=np.linspace(50, 80, 1000))
            plt.show()

    reg = build_registry(wavelength=561, divergence=2, n_super=1.522, n_subs=1.335)

    # reg.plot_transmission_vs_wavelength("Glass_thin", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.1, p_weight=0.9)
    # reg.plot_transmission_vs_wavelength("Glass_thin", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.5, p_weight=0.5)

    # reg.plot_rta_vs_wavelength("MAAPS488nm", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.1, p_weight=0.9)  

    # reg.plot_transmission_vs_wavelength("MAAPS488nm", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.1, p_weight=0.9)
    # reg.plot_reflectance_transmission_absorbance_vs_angle("MAAPS488nm_inverted", angle_range=(0, 80), s_weight=0.5, p_weight=0.5)
    # reg.plot_transmission_vs_wavelength("MAAPS488nm_inverted", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.5, p_weight=0.5)

    # reg.plot_transmission_vs_wavelength("MAAPS532nm", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.1, p_weight=0.9)
    # reg.plot_transmission_vs_wavelength("MAAPS532nm_inverted", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.5, p_weight=0.5)
    
    # reg.plot_transmission_vs_wavelength("MAAPS488nm", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.01, p_weight=0.99)
    reg.plot_transmission_vs_wavelength("MAAPS488nm_inverted", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.5, p_weight=0.5)

    reg.plot_transmission_vs_wavelength("MAAPS532nm", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.01, p_weight=0.99)
    reg.plot_transmission_vs_wavelength("MAAPS532nm_inverted", wavelength_range=(400, 800), angle_range=(0, 80), s_weight=0.5, p_weight=0.5)
    