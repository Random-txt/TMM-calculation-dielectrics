"""
materials.py
------------
Refractive index (n) and extinction coefficient (k) look-up functions
for optical thin-film materials used in Fresnel/TMM simulations.

Data are read from a shared Excel file containing wavelength-indexed
n, k columns for each material.  Missing materials fall back to
hard-coded constant values (polystyrene, custom medium).

Supported material keys
-----------------------
nb2o5, sio2, nb2o5_H, sio2_H, ta2o5_H, x_H, borosilicate,
nb2o5-26, sio2-26, sio2c, nb2o5c, polystyrene, custom
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default path to the shared n-k data workbook.
# Override by passing `filepath` explicitly to any function.
# ---------------------------------------------------------------------------
_DEFAULT_NK_FILE = (
    r"data"
    r"\nk_layers_measured_Fev2022_modif_ravi.xlsx"
)

# ---------------------------------------------------------------------------
# Column layout inside the Excel file (0-indexed after skiprows=2).
# Each entry maps a material key -> (wavelength_col, n_col) or
# (wavelength_col, n_col, k_col).
# ---------------------------------------------------------------------------
_NK_COLUMNS = {
    "nb2o5":       (1,  2,  3),
    "sio2":        (5,  6,  7),
    "nb2o5_H":     (9,  10, 11),
    "sio2_H":      (13, 14, 15),
    "ta2o5_H":     (17, 18, 19),
    "x_H":         (21, 22, 23),
    "borosilicate":(25, 26, 27),
    "nb2o5-26":    (29, 30, 31),
    "sio2-26":     (33, 34, 35),
    "sio2_602":       (37, 38, 39),
    "nb2o5_602":      (41, 42, 43),
    "sio2_622":       (45, 46, 47),
    "nb2o5_622":      (49, 50, 51),
}

# Cache the raw data array so the file is only read once per process.
_DATA_CACHE: dict[str, np.ndarray] = {}


def _load_data(filepath: str) -> np.ndarray:
    """Load (and cache) the raw n-k Excel table as a NumPy array."""
    if filepath not in _DATA_CACHE:
        _DATA_CACHE[filepath] = np.array(
            pd.read_excel(filepath, skiprows=2)
        )
    return _DATA_CACHE[filepath]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nk_measured(
    wavelength_nm: float,
    mat: str,
    filepath: str = _DEFAULT_NK_FILE,
) -> float:
    """Return the real refractive index *n* for *mat* at *wavelength_nm*.

    Parameters
    ----------
    wavelength_nm : float
        Excitation wavelength in nanometres.
    mat : str
        Material key (see module docstring for full list).
    filepath : str, optional
        Path to the n-k Excel data file.

    Returns
    -------
    float
        Refractive index n(λ).
    """
    # Hard-coded constants for materials not in the Excel file
    if mat == "custom":
        return 1.376458
    if mat == "polystyrene":
        return 1.59
    if mat == "lipid":
        return 1.41
    if mat == "solution":
        return 1.35
    if mat == "air":
        return 1.0
    if mat == "mgf2":
        return 1.37

    data = _load_data(filepath)
    lam_col, n_col, _ = _NK_COLUMNS[mat]

    lam_data = data[:, lam_col]
    n_data   = data[:, n_col]

    # print(f"{wavelength_nm} nm: material {mat} {float(np.interp(wavelength_nm, lam_data, n_data))}")
    return float(np.interp(wavelength_nm, lam_data, n_data))


def excitation_coeff(
    wavelength_nm: float,
    mat: str,
    filepath: str = _DEFAULT_NK_FILE,
) -> float:
    """Return the extinction coefficient *k* for *mat* at *wavelength_nm*.

    Parameters
    ----------
    wavelength_nm : float
        Excitation wavelength in nanometres.
    mat : str
        Material key (see module docstring for full list).
    filepath : str, optional
        Path to the n-k Excel data file.

    Returns
    -------
    float
        Extinction coefficient k(λ).
    """
    if mat == "custom":
        return 0.0
    if mat == "polystyrene":
        return 0.0
    if mat == "lipid":
        return 0.0
    if mat == "solution":
        return 0.0
    if mat == "air":
        return 0.0
    if mat == "mgf2":
        return 0.0
    data = _load_data(filepath)
    lam_col, _, k_col = _NK_COLUMNS[mat]
    lam_data = data[:, lam_col]
    k_data   = data[:, k_col]
    return float(np.interp(wavelength_nm, lam_data, k_data))