import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import seaborn as sns

sns.set_context("talk")
for wavelength in [488]:
    data_dir = os.getcwd()
    divergence = 2
    coverslip_name = f"MAAPS488nm"
    wavelength = wavelength
    files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx") and coverslip_name in f and f"{divergence}deg" in f and f"wavelength_{wavelength}" in f
             and f"1.523000_1.35" in f]
    # files = [f for f in os.listdir(data_dir) if f.endswith(".xlsx") and coverslip_name in f and f"nominal" in f and f"wavelength_{wavelength}" in f]

    def parse_weights(fname):
        # Accept pure polarization names like ..._s.xlsx and ..._s_<n1>_<n2>.xlsx
        if re.search(r'_s(?:_[0-9.]+_[0-9.]+)?\.xlsx$', fname):
            return 1.0, 0.0
        if re.search(r'_p(?:_[0-9.]+_[0-9.]+)?\.xlsx$', fname):
            return 0.0, 1.0
        # Match both s0_50p0_50 (underscore) and s0.50p0.50 (dot) formats
        m = re.search(r's0[._](\d+)p0[._](\d+)', fname)
        if m: return int(m.group(1))/100, int(m.group(2))/100
        return None, None

    plot_data = []
    for f in files:
        s, p = parse_weights(f)
        print(f"{f}  →  s={s}, p={p}")  # debug: remove once confirmed working
        if s is not None:
            df = pd.read_excel(os.path.join(data_dir, f))
            plot_data.append({'s': s, 'p': p, 'df': df})
    plot_data.sort(key=lambda x: x['s'], reverse=True)

    cmap = plt.cm.coolwarm
    fig, ax = plt.subplots(figsize=(11, 6))

    for d in plot_data:
        color = cmap(1 - d['s'])  # s=1 → blue, s=0 → red
        label = f"S={d['s']:.0%}, P={d['p']:.0%}"
        ax.plot(d['df']['Angle (deg)'], d['df']['Intensity (a.u.)'],
                color=color, lw=1.8, label=label)

    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Intensity (a.u.)')
    # ax.legend(title='Polarization mix', bbox_to_anchor=(1.02, 1), loc='upper left',
    #         fontsize=9, title_fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(60, 80)
    # ax.tick_params(labelsize=11)

    plt.tight_layout()
    # plt.savefig(rf'Z:\Gurram Raviteja\For cyril\cyril fabien p pol coverslips\SiO2SiOx/{coverslip_name}_{wavelength}nm_divergence_polarization_mix_{divergence}.png', bbox_inches='tight')
    plt.show()
    plt.close()