import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------- User settings ------------------------------
COVERSLIP_NAME = "MAAPS488nm" # e.g. "MAAPS488nm", "Glass_thin", etc.
WAVELENGTH_NM = 488
CASE = "divergence"  # "nominal", "divergence", or "both"
S_WEIGHT = 0.05
P_WEIGHT = 0.95


FILENAME_PATTERN = re.compile(
	r"^coverslip_(?P<name>.+?)_wavelength_(?P<wavelength>\d+)"
	r"_intensity_(?P<case>nominal|divergence_\d+(?:\.\d+)?deg)"
	r"_s(?P<s>\d+(?:\.\d+)?)p(?P<p>\d+(?:\.\d+)?)"
	r"_(?P<n_super>\d+\.\d+)_(?P<n_subs>\d+\.\d+)\.xlsx$"
)


def candidate_directories() -> list[Path]:
	"""Return plausible folders where exported Excel files may be located."""
	here = Path(__file__).resolve()
	cwd = Path.cwd().resolve()
	return [cwd, here.parent, here.parents[1], here.parents[2]]


def parse_file_metadata(file_path: Path) -> dict | None:
	"""Parse export metadata from the standard Excel filename format."""
	match = FILENAME_PATTERN.match(file_path.name)
	if not match:
		return None
	data = match.groupdict()
	return {
		"file": file_path,
		"name": data["name"],
		"wavelength": int(data["wavelength"]),
		"case": data["case"],
		"s": float(data["s"]),
		"p": float(data["p"]),
		"n_super": float(data["n_super"]),
		"n_subs": float(data["n_subs"]),
	}


def find_matching_files() -> tuple[list[dict], Path]:
	"""Find all Excel exports matching the selected settings."""
	selected_case = CASE.lower().strip()
	if selected_case not in {"nominal", "divergence", "both"}:
		raise ValueError("CASE must be one of: 'nominal', 'divergence', 'both'")

	found_dir = None
	metadata: list[dict] = []

	for directory in candidate_directories():
		if not directory.exists():
			continue
		files = list(directory.glob("*.xlsx"))
		if not files:
			continue

		current_batch: list[dict] = []
		for file in files:
			parsed = parse_file_metadata(file)
			if parsed is None:
				continue

			if parsed["name"] != COVERSLIP_NAME:
				continue
			if parsed["wavelength"] != WAVELENGTH_NM:
				continue
			if not np.isclose(parsed["s"], S_WEIGHT) or not np.isclose(parsed["p"], P_WEIGHT):
				continue

			is_nominal = parsed["case"] == "nominal"
			is_divergence = parsed["case"].startswith("divergence_")
			if selected_case == "nominal" and not is_nominal:
				continue
			if selected_case == "divergence" and not is_divergence:
				continue

			current_batch.append(parsed)

		if current_batch:
			found_dir = directory
			metadata = current_batch
			break

	if found_dir is None:
		raise FileNotFoundError(
			"No matching .xlsx files found. Check COVERSLIP_NAME, WAVELENGTH_NM, "
			"S_WEIGHT/P_WEIGHT, and CASE settings."
		)

	metadata.sort(key=lambda x: (x["case"], x["n_super"]))
	return metadata, found_dir


def extract_angle_intensity_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Return angle/intensity series, using named columns when available."""
	if {"Angle (deg)", "Intensity (a.u.)"}.issubset(df.columns):
		return df["Angle (deg)"], df["Intensity (a.u.)"]

	numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	if len(numeric_cols) >= 2:
		return df[numeric_cols[0]], df[numeric_cols[1]]

	raise ValueError(
		"Could not infer angle/intensity columns. Expected 'Angle (deg)' and "
		"'Intensity (a.u.)' or at least two numeric columns."
	)


def plot_case(ax: plt.Axes, rows: list[dict], title: str) -> None:
	"""Plot all n_super curves for a single case on one axis."""
	rows = sorted(rows, key=lambda x: x["n_super"])
	colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(rows)))

	for color, row in zip(colors, rows):
		df = pd.read_excel(row["file"])
		angle, intensity = extract_angle_intensity_columns(df)
		label = f"n_super={row['n_super']:.6f}"
		ax.plot(angle, intensity, lw=1.6, color=color, label=label)

	ax.set_title(title, fontsize=12)
	ax.set_xlabel("Angle (deg)", fontsize=11)
	ax.set_ylabel("Intensity (a.u.)", fontsize=11)
	ax.grid(True, linestyle=":", alpha=0.5)
	ax.tick_params(labelsize=10)
	ax.legend(fontsize=8, ncol=2, framealpha=0.9)


def main() -> None:
	rows, source_dir = find_matching_files()

	nominal_rows = [r for r in rows if r["case"] == "nominal"]
	divergence_rows = [r for r in rows if r["case"].startswith("divergence_")]

	if CASE == "both":
		fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
		if nominal_rows:
			plot_case(axes[0], nominal_rows, "Nominal")
		else:
			axes[0].set_title("Nominal (no files found)")
			axes[0].axis("off")

		if divergence_rows:
			div_label = divergence_rows[0]["case"].replace("_", " ")
			plot_case(axes[1], divergence_rows, div_label)
		else:
			axes[1].set_title("Divergence (no files found)")
			axes[1].axis("off")

		fig.suptitle(
			f"{COVERSLIP_NAME} | {WAVELENGTH_NM} nm | S={S_WEIGHT:.2f}, P={P_WEIGHT:.2f}",
			fontsize=13,
		)
		plt.tight_layout()
		out_file = source_dir / (
			f"{COVERSLIP_NAME}_n_super_scan_nominal_vs_divergence_"
			f"{WAVELENGTH_NM}nm_s{S_WEIGHT:.2f}p{P_WEIGHT:.2f}.svg"
		)
	elif CASE == "nominal":
		if not nominal_rows:
			raise FileNotFoundError("No nominal files found for current settings.")
		fig, ax = plt.subplots(figsize=(10, 6))
		plot_case(ax, nominal_rows, "Nominal")
		fig.suptitle(
			f"{COVERSLIP_NAME} | Nominal | {WAVELENGTH_NM} nm | "
			f"S={S_WEIGHT:.2f}, P={P_WEIGHT:.2f}",
			fontsize=13,
		)
		plt.tight_layout()
		out_file = source_dir / (
			f"{COVERSLIP_NAME}_n_super_scan_nominal_"
			f"{WAVELENGTH_NM}nm_s{S_WEIGHT:.2f}p{P_WEIGHT:.2f}.svg"
		)
	else:
		if not divergence_rows:
			raise FileNotFoundError("No divergence files found for current settings.")
		fig, ax = plt.subplots(figsize=(10, 6))
		div_label = divergence_rows[0]["case"].replace("_", " ")
		plot_case(ax, divergence_rows, div_label)
		fig.suptitle(
			f"{COVERSLIP_NAME} | {div_label} | {WAVELENGTH_NM} nm | "
			f"S={S_WEIGHT:.2f}, P={P_WEIGHT:.2f}",
			fontsize=13,
		)
		plt.tight_layout()
		out_file = source_dir / (
			f"{COVERSLIP_NAME}_n_super_scan_divergence_"
			f"{WAVELENGTH_NM}nm_s{S_WEIGHT:.2f}p{P_WEIGHT:.2f}.svg"
		)

	plt.savefig(out_file, bbox_inches="tight")
	print(f"Saved figure: {out_file}")
	plt.show()


if __name__ == "__main__":
	main()
