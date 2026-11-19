"""APOKASC-3 giants around J0008 and the selections used for its seismic-mass comparison.

(a) Teff-log g, (b) [Fe/H]-[C/N], (c) mass distributions, (d) median and 16-84% mass per selection.
Selections:
  broad: Teff from 4700, 4800 or 4900 K to 5050 K, log g 3.0-3.3, [Fe/H] and [C/N] -0.3 to 0
  survey-centred: +/-80 K, +/-0.07 dex log g, +/-0.1 dex [Fe/H] around each survey's corrected values in
  Zhao et al. (2026) Table S2, and +/-0.1 dex around the DR17 [C/N] (as in j0008_analysis.py)
Not in the note. Usage: python j0008_apokasc3_selections.py [path/to/apokasc3_table4.csv] [outstem]
Writes fig_apokasc3_selections.png and .pdf next to this script by default.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "apokasc3_table4.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig_apokasc3_selections")

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
MASS_CMAP = LinearSegmentedColormap.from_list("mass", ["#d8d3f6", "#8f81e6", "#4b3bb8", "#1f1466"])
MLO, MHI = 0.9, 1.9

rows = list(csv.DictReader(open(CSV)))
col = lambda c: np.array([float(r[c]) if r[c].strip() else np.nan for r in rows])
te, lg, fe, mass = col("Teff"), col("loggSpec"), col("[Fe/H]"), col("Mass")
cn = col("[C/Fe]") - col("[N/Fe]")
ok = np.isfinite(mass) & (mass > 0) & np.isfinite(cn) & np.isfinite(te) & np.isfinite(lg) & np.isfinite(fe)

CN_J = -0.013652 - 0.16286                     # J0008 APOGEE DR17 [C/N]
SURVEYS = [("APOGEE DR17", 4964, 3.14, -0.24, "o"), ("APOGEE SDSS-V", 4984, 3.19, -0.22, "s"),
           ("LAMOST LRS (LASP)", 4993, 3.27, -0.22, "^"), ("LAMOST LRS (DD-Payne)", 4986, 3.30, -0.22, "D"),
           ("LAMOST MRS (LASP)", 4998, 3.20, -0.28, "v")]
BROAD_T = (4700, 4800, 4900)

def broad(tlo):
    return ok & (te > tlo) & (te < 5050) & (lg > 3.0) & (lg < 3.3) & (fe > -0.3) & (fe < 0) & (cn > -0.3) & (cn < 0)

def centred(t0, g0, f0):
    return (ok & (np.abs(te - t0) <= 80) & (np.abs(lg - g0) <= 0.07) & (np.abs(fe - f0) <= 0.1)
            & (np.abs(cn - CN_J) <= 0.1))

sel = [(f"Broad, Teff {t}–5050 K", broad(t)) for t in BROAD_T] + \
      [(f"Centred on {s[0]}", centred(*s[1:4])) for s in SURVEYS]
for lab, s in sel:
    q = np.percentile(mass[s], [16, 50, 84])
    print(f"{lab:40s} N = {s.sum():4d}  M = {q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f})  P(>=1.67) = {np.mean(mass[s] >= 1.67):.2f}")

plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), dpi=300)
(ax, bx), (cx, dx) = axs

# ---------------------------------------------------------------- (a) Teff-log g
win = ok & (te > 4450) & (te < 5350) & (lg > 2.75) & (lg < 3.55)
chem = win & (fe > -0.3) & (fe < 0) & (cn > -0.3) & (cn < 0)
ax.scatter(te[win & ~chem], lg[win & ~chem], s=2, c="#cfcdc8", lw=0, zorder=1, rasterized=True)
sc = ax.scatter(te[chem], lg[chem], s=5, c=mass[chem], cmap=MASS_CMAP, vmin=MLO, vmax=MHI, lw=0, zorder=2,
                rasterized=True)
ax.add_patch(Rectangle((min(BROAD_T), 3.0), 5050 - min(BROAD_T), 0.3, fill=False, ec=INK, lw=0.7,
                       ls=(0, (4, 2)), zorder=3))                   # nested broad boxes: outer edge ...
for t in BROAD_T[1:]:
    ax.plot([t, t], [3.0, 3.3], color=INK, lw=0.7, ls=(0, (4, 2)), zorder=3)   # ... and their inner edges
for t in BROAD_T:
    ax.text(t, 2.995, f"{t}", fontsize=5.5, color=INK, ha="center", va="bottom")
for name, t0, g0, f0, mk in SURVEYS:
    ax.add_patch(Rectangle((t0 - 80, g0 - 0.07), 160, 0.14, fill=False, ec=BLUE, lw=0.6, alpha=0.8, zorder=3))
    ax.plot(t0, g0, mk, ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=4, label=name)
ax.text(0.03, 0.03, "dashed: broad selections (lower Teff edge labelled)\nblue: survey-centred boxes",
        transform=ax.transAxes, fontsize=6, color=MUTED, va="bottom")
ax.set_xlim(5350, 4450); ax.set_ylim(3.55, 2.75)
ax.set_xlabel(r"$T_{\rm eff}$ (K, APOGEE DR17)"); ax.set_ylabel(r"$\log g$ (spectroscopic, APOGEE DR17)")
ax.set_title("(a) Coloured: [Fe/H] and [C/N] from −0.3 to 0", fontsize=7.5, loc="left", color=INK)

# ---------------------------------------------------------------- (b) [Fe/H]-[C/N]
kiel = ok & (te > 4700) & (te < 5050) & (lg > 3.0) & (lg < 3.3)
bx.scatter(fe[ok & ~kiel], cn[ok & ~kiel], s=1.5, c="#e2e0db", lw=0, zorder=1, rasterized=True)
bx.scatter(fe[kiel], cn[kiel], s=5, c=mass[kiel], cmap=MASS_CMAP, vmin=MLO, vmax=MHI, lw=0, zorder=2,
           rasterized=True)
bx.add_patch(Rectangle((-0.3, -0.3), 0.3, 0.3, fill=False, ec=INK, lw=0.7, ls=(0, (4, 2)), zorder=3))
for name, t0, g0, f0, mk in SURVEYS:
    bx.add_patch(Rectangle((f0 - 0.1, CN_J - 0.1), 0.2, 0.2, fill=False, ec=BLUE, lw=0.6, alpha=0.8, zorder=3))
    bx.plot(f0, CN_J, mk, ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=4)
bx.set_xlim(-0.75, 0.45); bx.set_ylim(-0.75, 0.3)
bx.set_xlabel("[Fe/H] (APOGEE DR17)"); bx.set_ylabel("[C/N] (APOGEE DR17)")
bx.set_title(r"(b) Coloured: $T_{\rm eff}$ 4700–5050 K, $\log g$ 3.0–3.3", fontsize=7.5, loc="left",
             color=INK)
cb = fig.colorbar(sc, ax=bx, fraction=0.05, pad=0.02)
cb.set_label(r"seismic mass (M$_\odot$)"); cb.outline.set_linewidth(0.5)
handles, labels = ax.get_legend_handles_labels()
leg = bx.legend(handles, labels, loc="lower left", fontsize=5.8, frameon=True, framealpha=0.8, edgecolor="none",
                handletextpad=0.3, borderaxespad=0.2, title="J0008 (Table S2;\nDR17 [C/N] for all)",
                title_fontsize=5.8)
leg._legend_box.align = "left"

# ---------------------------------------------------------------- (c) mass distributions
bins = np.arange(0.7, 2.61, 0.05)
for a_ in (cx, dx):
    a_.axvspan(1.10, 1.40, color=BLUE, alpha=0.12, lw=0, zorder=0)
    a_.axvspan(1.52, 1.82, color=ORANGE, alpha=0.12, lw=0, zorder=0)
s_b, s_c = sel[0][1], sel[3][1]
cx.hist(mass[s_b], bins=bins, histtype="stepfilled", color="#cfcdc8", ec=MUTED, lw=0.6, zorder=1,
        label=f"Broad, Teff 4700–5050 K (N = {s_b.sum()})")
cx.hist(mass[s_c], bins=bins, histtype="step", color=INK, lw=1.1, zorder=2,
        label=f"Centred on APOGEE DR17 (N = {s_c.sum()})")
ymax = cx.get_ylim()[1]
cx.text(1.25, ymax * 1.02, "this note\n1.25±0.15", ha="center", va="bottom", fontsize=6, color=BLUE)
cx.text(1.67, ymax * 1.02, "Zhao+26\n1.67±0.15", ha="center", va="bottom", fontsize=6, color=ORANGE)
cx.set_ylim(0, ymax * 1.22)
cx.set_xlim(0.7, 2.6); cx.set_xlabel(r"APOKASC-3 seismic mass (M$_\odot$)"); cx.set_ylabel("stars per 0.05 M$_\\odot$")
cx.legend(loc="center right", fontsize=6, frameon=False, bbox_to_anchor=(1.0, 0.55))
cx.set_title("(c) Mass distributions", fontsize=7.5, loc="left", color=INK)

# ---------------------------------------------------------------- (d) per-selection summary
ys, labs = [], []
for k, (lab, s) in enumerate(sel):
    y = len(sel) - 1 - k + (0.6 if k < 3 else 0)
    q = np.percentile(mass[s], [16, 50, 84])
    dx.plot([q[0], q[2]], [y, y], color=INK, lw=1.0, zorder=2, solid_capstyle="round")
    dx.plot(q[1], y, "o", ms=4, color=INK, mec="white", mew=0.6, zorder=3)
    dx.text(2.58, y, f"N = {s.sum()},  {100 * np.mean(mass[s] >= 1.67):.0f}% ≥ 1.67", va="center", ha="right",
            fontsize=5.8, color=MUTED)
    ys.append(y)
    labs.append(lab.replace("Broad, Teff ", r"$T_{\rm eff}$ ").replace("Centred on ", ""))
dx.set_yticks(ys, labs, fontsize=6.2, color=INK)
dx.tick_params(axis="y", length=0)
tr = dx.get_yaxis_transform()
dx.text(-0.02, ys[0] + 0.75, "Broad selections", transform=tr, ha="right", fontsize=6.2, color=MUTED, style="italic")
dx.text(-0.02, ys[3] + 0.75, "Survey-centred boxes", transform=tr, ha="right", fontsize=6.2, color=MUTED,
        style="italic")
dx.set_ylim(-0.7, len(sel) + 0.6)
dx.set_xlim(0.7, 2.6); dx.set_xlabel(r"APOKASC-3 seismic mass (M$_\odot$): median and 16–84%")
dx.set_title("(d) Each selection", fontsize=7.5, loc="left", color=INK)

for a_ in axs.flat:
    a_.tick_params(length=2.5)
    for side in ("top", "right"):
        a_.spines[side].set_visible(False)
fig.tight_layout(h_pad=1.6, w_pad=1.2)
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png/.pdf")
