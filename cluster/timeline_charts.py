"""Charts of the life islands' evolution from life/islands/<k>/metrics.csv and census.csv.
usage: python cluster/timeline_charts.py <life/islands> <out.png>"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root, out = Path(sys.argv[1]), Path(sys.argv[2])


def read(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows, name, scale=1.0):
    v = []
    for r in rows:
        try:
            v.append(float(r[name]) * scale)
        except (KeyError, TypeError, ValueError):
            v.append(float("nan"))
    return v


islands = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
plt.style.use("dark_background")
fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
fig.suptitle("Life on the home planet (seed 59): all islands", fontsize=15)
for isl in islands:
    m, c = read(isl / "metrics.csv"), read(isl / "census.csv")
    if not m:
        continue
    y = col(m, "year")
    lbl = f"island {isl.name}"
    ax[0][0].semilogy(y, col(m, "live"), label=lbl)
    ax[0][1].semilogy(y, [max(v, 1e-3) for v in col(m, "C_land", 1e-3)], label=lbl)
    ax[0][2].plot(y, col(m, "land_cells_veg"), label=lbl)
    ax[1][0].semilogy(y, [max(v, 1e-3) for v in col(m, "OXY")], label=f"{lbl} OXY")
    ax[1][1].plot(y, col(m, "g_struct"), label=f"{lbl}")
    if c:
        ax[1][2].plot(col(c, "year"), col(c, "tallest_m"), label=lbl)
ax[0][0].set_title("organisms (agents)")
ax[0][1].set_title("living carbon on land (mol C)")
ax[0][2].set_title("vegetated land cells")
ax[1][0].set_title("oxygenic photosynthesis (mmol/day)")
ax[1][1].set_title("mean structure gene (wood/height investment, 0..255)")
ax[1][2].set_title("tallest plant (m)")
for a in ax.flat:
    a.set_xlabel("simulated years")
    a.grid(alpha=0.2)
    if a.lines:
        a.legend(fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(out, dpi=75)
print("wrote", out)
