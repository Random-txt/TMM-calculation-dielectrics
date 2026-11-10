import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy.integrate import trapezoid

data_dir = os.getcwd()
divergence = 1.5
coverslip_name = f"MAAPS532"


def parse_weights(fname):
    # Accept pure polarization names like ..._s.xlsx and ..._s_<n1>_<n2>.xlsx
    if re.search(r'_s(?:_[0-9.]+_[0-9.]+)?\.xlsx$', fname):
        return 1.0, 0.0
    if re.search(r'_p(?:_[0-9.]+_[0-9.]+)?\.xlsx$', fname):
        return 0.0, 1.0
    # Match both s0_50p0_50 (underscore) and s0.50p0.50 (dot) formats
    m = re.search(r's0[._](\d+)p0[._](\d+)', fname)
    if m:
        return int(m.group(1)) / 100, int(m.group(2)) / 100
    return None, None


plt.figure(figsize=(8, 6))
ax = plt.gca()
pure_p_data = {}


for wavelength in [488, 532, 561, 640]:

    files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx") and coverslip_name in f and f"{divergence}deg" in f and f"wavelength_{wavelength}" in f]

    plot_data = []
    for f in files:
        s, p = parse_weights(f)
        if p == 1.0:
            df = pd.read_excel(os.path.join(data_dir, f))
            plot_data.append({'s': s, 'p': p, 'df': df})
    if plot_data:
        df = plot_data[0]['df']
        pure_p_data[wavelength] = df
        ax.plot(df['Angle (deg)'], df['Intensity (a.u.)'], label=f'{wavelength} nm')


# plot in one figure, only pure p polar condition with all wavelengths in it. the intensity vs angles plot
ax.set_xlabel('Angle (degrees)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title(f'{coverslip_name} - Pure p polarization', fontsize=14)
ax.legend(title='Wavelength', bbox_to_anchor=(1.02, 1), loc='upper left',
          fontsize=9, title_fontsize=10, framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_xlim(55, 75)  # Focus on 55-75°
# ax.set_xlim(60, 80)  # Focus on 60-80°
plt.tight_layout()
plt.savefig(rf'Z:\Gurram Raviteja\For cyril\cyril fabien p pol coverslips\SiO2SiOx/{coverslip_name}_pure_p_all_wavelengths_{divergence}deg.png', bbox_inches='tight')


integration_windows = {
    640: (62.5, 67.5),
    561: (65.0, 69.5),
    532: (66.0, 70.0),
    488: (68, 71.5),
}

for wavelength, (angle_min, angle_max) in integration_windows.items():
    df = pure_p_data.get(wavelength)
    if df is None:
        print(f'{wavelength} nm: no pure p data found')
        continue

    angle_mask = (df['Angle (deg)'] >= angle_min) & (df['Angle (deg)'] <= angle_max)
    selected = df.loc[angle_mask]
    if selected.empty:
        print(f'{wavelength} nm: no samples in {angle_min}-{angle_max}°')
        continue

    area = trapezoid(selected['Intensity (a.u.)'], selected['Angle (deg)'])
    print(f'{wavelength} nm: area under curve from {angle_min}° to {angle_max}° = {area:.6g}')




