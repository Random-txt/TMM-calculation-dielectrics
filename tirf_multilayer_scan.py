"""
tirf_multilayer_scan.py
=======================
Config-driven entry-point for the dual-wavelength R/T/A + E-field
depth-profile analysis.

All physics lives in :class:`coverslip_registry.CoverslipRegistry`
(method :meth:`plot_rta_efield_dual_wavelength`).  This file only
holds the CONFIG block and the lines that wire it up.

Stack definitions shared with ``run.py`` are re-used via
``build_registry`` — there is no duplication of the STACKS dict.

Layout
------
1.  CONFIG  — edit wavelengths, angle range, TIRF angle, coverslip name
2.  Custom stack definition (optional — skip if using a STACKS entry)
3.  Entry point — builds registry, runs analysis

Run with::

    python tirf_multilayer_scan.py
"""

import matplotlib.pyplot as plt

# Re-use the shared stack definitions and registry factory from run.py
from run import build_registry, STACKS


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ── edit everything here ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Coverslip to analyse — must be a key in STACKS (run.py) or "custom" (below)
COVERSLIP_NAME = "Glass_thin"

# Wavelengths (nm)
LAM_EXC  = 488.0
LAM_CTRL = 488.0

# Angular sweep limits and resolution (degrees, in the glass superstrate)
THETA_MIN =  0.0
THETA_MAX = 80.0
N_THETA   = 500

# Angle used for the E-field depth profile
THETA_EFIELD = 64.0

# Sampling density inside each layer for the E-field profile
N_PTS_PER_LAYER = 300

# Print optical-thickness table to stdout?
PRINT_TABLE = True

# Output figure path — None to skip saving.
# Extension controls format: .png  .svg  .pdf
SAVEFIG = "tirf_multilayer_scan.png"

# # ── Optional custom stack ────────────────────────────────────────────────────
# # Set COVERSLIP_NAME = "custom" and fill in the lists below to analyse a
# # stack not already registered in run.py.
# # Layers ordered superstrate → substrate.
# CUSTOM_STACK = dict(
#     thicknesses=[100.0, 150.0, 100.0],
#     materials=["nb2o5", "sio2", "nb2o5"],
# )

# ─────────────────────────────────────────────────────────────────────────────
# END CONFIG
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    # Build the shared registry.  n/k are initialised at LAM_EXC here,
    # but plot_rta_efield_dual_wavelength re-fetches n/k independently at
    # both wavelengths via materials.py, so the init wavelength only matters
    # for methods that cache refractive indices at construction time.
    reg = build_registry(wavelength=LAM_EXC, pol="s")

    # # Register the custom stack if that option was selected
    # if COVERSLIP_NAME == "custom":
    #     reg.add_coverslip(COVERSLIP_NAME, **CUSTOM_STACK)

    reg.plot_rta_efield_dual_wavelength(
        name=COVERSLIP_NAME,
        lambda_exc=LAM_EXC,
        lambda_ctrl=LAM_CTRL,
        theta_efield=THETA_EFIELD,
        angle_range=(THETA_MIN, THETA_MAX),
        num_angles=N_THETA,
        n_pts_per_layer=N_PTS_PER_LAYER,
        print_optical_table=PRINT_TABLE,
        savefig=SAVEFIG,
    )

    plt.show()
