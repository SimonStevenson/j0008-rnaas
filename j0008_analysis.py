"""Reproduce the numbers and Figure 1 of the RNAAS comment on Zhao et al. (2026),
J0008+0128 (Gaia DR3 2546630699873483008).

Usage:  python j0008_analysis.py        (writes fig_m1m2.pdf/.png, prints numbers)
Inputs: inputs.csv (all values with their sources)
Optional: `pip install gaiadr3-zeropoint` to recompute the Lindegren et al. (2021)
parallax zero point; otherwise the value stored in inputs.csv is used.
"""
import csv
import os

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
G_CGS, RSUN, MSUN, TSUN = 6.674e-8, 6.957e10, 1.989e33, 5772.0
N_MC, SEED = 100_000, 3
D_PAPER = 1464.56  # pc, distance used by Zhao et al. for the SED radius

# ---------------------------------------------------------------- inputs
inp = {}
with open(os.path.join(HERE, "inputs.csv")) as fh:
    for row in csv.DictReader(fh):
        f = lambda k: float(row[k]) if row[k] else np.nan
        inp[row["quantity"]] = (f("value"), f("lower"), f("upper"))
val = lambda k: inp[k][0]
half_width = lambda k: (inp[k][2] - inp[k][1]) / 2

FM, SIG_FM = val("mass_function"), half_width("mass_function")


def m2_from(m1, inc_deg, fm=FM):
    """Companion mass from the mass function f = (M2 sin i)^3 / (M1 + M2)^2."""
    s = np.sin(np.radians(inc_deg))
    return brentq(lambda m2: (m2 * s) ** 3 / (m1 + m2) ** 2 - fm, 1e-3, 50.0)


def m2_vec(m1, inc_deg, fm=FM):
    """Vectorised version of m2_from (bisection; the left side rises monotonically in M2)."""
    m1, s, fm = np.broadcast_arrays(m1, np.sin(np.radians(inc_deg)), fm)
    lo, hi = np.full(m1.shape, 1e-3), np.full(m1.shape, 50.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        low = (mid * s) ** 3 / (m1 + mid) ** 2 < fm
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return 0.5 * (lo + hi)


def grav_mass(logg, r_rsun):
    return 10 ** logg * (r_rsun * RSUN) ** 2 / G_CGS / MSUN


rng = np.random.default_rng(SEED)


def draw(m1_mu, m1_sig, inc_mu=73.0, inc_sig=11.0, n=N_MC):
    """Monte Carlo draws of (M1, i, M2), with Gaussian M1, i and f(M); i folded at 90 deg."""
    m1 = rng.normal(m1_mu, m1_sig, n)
    inc = rng.normal(inc_mu, inc_sig, n)
    inc = np.where(inc > 90, 180 - inc, inc)
    fm = rng.normal(FM, SIG_FM, n)
    return m1, inc, m2_vec(m1, inc, fm)


def describe(m2):
    p5, p16, p50, p84, p95 = np.percentile(m2, [5, 16, 50, 84, 95])
    return (f"M2 = {p50:.2f} +{p84 - p50:.2f} -{p50 - p16:.2f} Msun (5-95%: {p5:.2f}-{p95:.2f}); "
            f"P(M2>2.0) = {(m2 > 2.0).mean():.2f}, P(M2>2.3) = {(m2 > 2.3).mean():.2f}")


# ------------------------------------------------- 1. what sets M2
print("== Section 1: sensitivity ==")
dm2_dm1 = (m2_from(1.67, 73) - m2_from(1.20, 73)) / (1.67 - 1.20)
print(f"dM2/dM1 at i=73 deg: {dm2_dm1:.2f}")
print("M2(1.20, i=62/73/84/90) = "
      + ", ".join(f"{m2_from(1.20, i):.2f}" for i in (62, 73, 84, 90)))
print(f"M2(1.67, i=73) = {m2_from(1.67, 73):.2f}")

# Zhao et al. combine five masses as if they were independent. Their adopted values
# match the inverse-variance weighted means (symmetrised errors).
print("\nZhao et al. combination of five estimates:")
for pre in ("M1", "M2"):
    keys = [k for k in inp if k.startswith(pre + "_Zhao_") and k not in
            (pre + "_Zhao_adopted", pre + "_Zhao_MIST")]
    mu = np.array([val(k) for k in keys])
    sig = np.array([half_width(k) for k in keys])
    w = 1 / sig ** 2
    print(f"  {pre}: inverse-variance mean of {len(keys)} = {np.sum(w * mu) / w.sum():.2f} "
          f"+/- {w.sum() ** -0.5:.2f} Msun (Zhao et al. adopt {val(pre + '_Zhao_adopted'):.2f} "
          f"+{inp[pre + '_Zhao_adopted'][2] - val(pre + '_Zhao_adopted'):.2f} "
          f"-{val(pre + '_Zhao_adopted') - inp[pre + '_Zhao_adopted'][1]:.2f})")
_, _, m2_zhao = draw(1.67, 0.15)
print(f"  Propagating M1 = 1.67 +/- 0.15 and i = 73 +/- 11 directly: {describe(m2_zhao)}")

# ------------------------------------------- 2. gR^2/G versus distance
print("\n== Section 2: gR^2/G versus distance ==")
logg, sig_logg = val("logg_APOGEE"), val("logg_err_calibrated")
r_paper = val("R1_SED")  # R scales linearly with d
for d in (1296.0, 1323.6, 1403.1, 1416.0, D_PAPER):
    print(f"d = {d:7.1f} pc -> R = {r_paper * d / D_PAPER:.2f} Rsun, "
          f"M1 = {grav_mass(logg, r_paper * d / D_PAPER):.2f} Msun")
print(f"distance needed for M1 = 1.67 with the APOGEE logg: "
      f"{D_PAPER * np.sqrt(1.67 / grav_mass(logg, r_paper)):.0f} pc")

# ---------------------------------------------- parallax zero point
plx, sig_plx = val("parallax_DR3"), half_width("parallax_DR3")
zp = val("parallax_zero_point_L21")
print(f"\n1/parallax = {1e3 / plx:.1f} pc (Zhao et al. use {D_PAPER} pc)")
try:
    from zero_point import zpt
    zpt.load_tables()
    zp_new = zpt.get_zpt(np.array([val("G_DR3")]), np.array([val("nu_eff_DR3")]),
                         np.array([np.nan]), np.array([val("ecl_lat_DR3")]),
                         np.array([int(val("astrometric_params_solved"))]))[0]
    print(f"gaiadr3-zeropoint: {zp_new:.4f} mas (inputs.csv: {zp:.4f} mas)")
    zp = zp_new
except ImportError:
    print(f"gaiadr3-zeropoint not installed; using {zp:.4f} mas from inputs.csv")
d_corr = 1e3 / (plx - zp)
print(f"L21 zero point = {zp:.4f} mas -> d = {d_corr:.1f} +/- "
      f"{1e3 * sig_plx / (plx - zp) ** 2:.0f} pc")

# gR^2/G at the zero-point-corrected distance, with the calibrated logg error
# (Zhao et al. Table S2) and the parallax error; the SED radius scales with d.
lg_s = rng.normal(logg, sig_logg, N_MC)
d_s = 1e3 / rng.normal(plx - zp, sig_plx, N_MC)
m1_plx_s = grav_mass(lg_s, r_paper * d_s / D_PAPER)
m1_plx = np.percentile(m1_plx_s, [16, 50, 84])
m1_plx_mu, m1_plx_sig = grav_mass(logg, r_paper * d_corr / D_PAPER), m1_plx_s.std()
print(f"gR^2/G at {d_corr:.0f} pc: M1 = {m1_plx_mu:.2f} +/- {m1_plx_sig:.2f} Msun "
      f"(16/50/84%: {m1_plx[0]:.2f}/{m1_plx[1]:.2f}/{m1_plx[2]:.2f})")

# ---------------------------------------------- Gaia FLAME
r_fl, l_fl = val("radius_FLAME"), val("lum_FLAME")
print(f"\nFLAME radius {r_fl:.2f} Rsun + APOGEE logg -> {grav_mass(logg, r_fl):.2f} Msun")
print(f"FLAME luminosity and radius imply Teff = {TSUN * (l_fl / r_fl ** 2) ** 0.25:.0f} K "
      f"(APOGEE {val('Teff_APOGEE'):.0f} K)")

# ------------------------------------- Martig et al. (2016) [C/N] masses
U4 = np.array([  # Table 4: [1,[M/H],[C/M],[N/M],[(C+N)/M],Teff/4000,logg]
    [95.87, -10.40, 41.36, 15.05, -67.61, -144.18, -9.42],
    [0, -0.73, -5.32, -0.93, 7.05, 5.12, 1.52],
    [0, 0, -46.78, -30.52, 133.58, -73.77, 16.04],
    [0, 0, 0, -1.61, 38.94, -15.29, 1.35],
    [0, 0, 0, 0, -88.99, 101.75, -18.65],
    [0, 0, 0, 0, 0, 27.77, 28.80],
    [0, 0, 0, 0, 0, 0, -4.10]])
U3 = np.array([  # Table 3: [1,[M/H],[C/M],[N/M],[(C+N)/M]]
    [1.08, -0.18, 4.30, 1.43, -7.55],
    [0, -1.05, -1.12, -0.67, -1.30],
    [0, 0, -49.92, -41.04, 139.92],
    [0, 0, 0, -0.63, 47.33],
    [0, 0, 0, 0, -86.62]])


def cnm(c, n, eps_c=8.39, eps_n=7.78):
    return np.log10((10 ** (c + eps_c) + 10 ** (n + eps_n)) / (10 ** eps_c + 10 ** eps_n))


def martig(mh, cm, nm, teff, lg, table=4):
    x = np.array([1, mh, cm, nm, cnm(cm, nm), teff / 4000, lg])
    return x[:5] @ U3 @ x[:5] if table == 3 else x @ U4 @ x


# [C/M] ~ [C/Fe] and [N/M] ~ [N/Fe] (assumes [Fe/H] ~ [M/H])
args = (val("M_H_APOGEE"), val("C_Fe_APOGEE"), val("N_Fe_APOGEE"), val("Teff_APOGEE"), logg)
m4 = martig(*args)
print(f"\nMartig Table 3: {martig(*args, table=3):.2f} Msun; Table 4: {m4:.2f} Msun")
print(f"  Table 4, Teff+100 K: {martig(*args[:3], args[3] + 100, logg) - m4:+.2f} Msun")
print(f"  Table 4, [N/M]+0.05: {martig(args[0], args[1], args[2] + 0.05, args[3], logg) - m4:+.2f} Msun")

# ------------------------------------------- 3. Monte Carlo companion mass
print("\n== Section 3: companion mass ==")
m1_s, inc_s, m2_s = draw(val("M1_adopted_this_note"), half_width("M1_adopted_this_note"))
print(f"Adopted M1 = 1.20 +/- 0.15, i = 73 +/- 11: {describe(m2_s)}")
tail = m2_s > 2.3
print(f"  draws with M2 > 2.3: {np.mean(inc_s[tail] < 62):.2f} have i < 62 deg")
print(f"  std(M2): all {m2_s.std():.2f}; M1 varied alone {m2_vec(m1_s, 73.0).std():.2f}; "
      f"i varied alone {m2_vec(1.20, inc_s).std():.2f} Msun")
print(f"Minimum mass at i=90 deg for M1=1.20: {m2_from(1.20, 90):.2f} Msun")
print("Robustness to the adopted M1:")
for mu, sig, lab in ((1.30, 0.20, "MIST track masses, wider uncertainty"),
                     (round(m1_plx_mu, 2), round(m1_plx_sig, 2), "gR^2/G, ZP-corrected parallax"),
                     (1.67, 0.15, "Zhao et al. adopted M1")):
    print(f"  M1 = {mu:.2f} +/- {sig:.2f} ({lab}): {describe(draw(mu, sig)[2])}")
for m1_par, lab in ((1.38, "ZP-corrected parallax"), (1.48, "uncorrected parallax")):
    print(f"Parallax-based M1 = {m1_par} ({lab}) -> M2(i=73) = {m2_from(m1_par, 73):.2f} Msun")

# ------------------------------------------------------------ Figure 1
BLUE, ORANGE, VIOLET, INK, MUTED = "#2a78d6", "#eb6834", "#6250d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
grid = np.linspace(0.95, 2.0, 200)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 4.0), dpi=300, sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.08})
ax.axhspan(2.15, 2.30, color="#e4e3df", lw=0, zorder=0)
ax.text(0.97, 2.225, r"max. NS mass (2.15–2.3 M$_\odot$)", va="center", fontsize=6.5, color=MUTED)
ax.axvspan(1.05, 1.35, color=BLUE, alpha=0.12, lw=0, zorder=0)
ax.axvspan(1.52, 1.82, color=ORANGE, alpha=0.12, lw=0, zorder=0)
ax.fill_between(grid, m2_vec(grid, 84.0), m2_vec(grid, 62.0), color=VIOLET, alpha=0.25,
                lw=0, zorder=1)
ax.plot(grid, m2_vec(grid, 73.0), color=INK, lw=1.2, zorder=2)
ax.plot(grid, m2_vec(grid, 90.0), color=INK, lw=0.8, ls="--", zorder=2)
ax.text(1.99, m2_from(1.99, 73) + 0.03, r"$i=73^\circ\pm11^\circ$", ha="right", va="bottom",
        fontsize=6.5, color=INK)
ax.text(1.40, m2_from(1.40, 90) - 0.05, r"$i=90^\circ$ (minimum)", ha="left", va="top",
        fontsize=6.5, color=INK)
for m, col in ((1.20, BLUE), (1.67, ORANGE)):
    y = m2_from(m, 73)
    ax.plot(m, y, "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=4)
ax.set_ylim(1.55, 2.6)
ax.set_ylabel(r"Companion mass $M_2$ (M$_\odot$)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (label, (value, lower, upper), colour, show error bar, label side)
est = [("Zhao+26 adopted (gR²/G, d=1.46 kpc)", inp["M1_Zhao_adopted"], ORANGE, True, "left"),
       ("Zhao+26 MIST isochrones", inp["M1_Zhao_MIST"], ORANGE, True, "left"),
       (f"gR²/G at d={d_corr / 1e3:.2f} kpc (parallax)", (m1_plx_mu, m1_plx[0], m1_plx[2]),
        INK, True, "right"),
       ("APOGEE DistMass ([C/N])", inp["M1_DistMass"], BLUE, True, "right"),
       ("APOGEE StarHorse", inp["M1_StarHorse"], BLUE, False, "right"),
       ("Gaia FLAME (solar-Z prior)", inp["M1_FLAME"], MUTED, True, "left")]
for k, (lab, (m, lo, hi), col, bar, side) in enumerate(est):
    y = len(est) - 1 - k
    xerr = [[m - lo], [hi - m]] if bar else None
    bx.errorbar(m, y, xerr=xerr, fmt="o", ms=4, color=col, mec="white",
                mew=0.8, elinewidth=1.3, capsize=0)
    right_edge, left_edge = (hi, lo) if bar else (m + 0.01, m - 0.01)
    bx.text(right_edge + 0.02 if side == "right" else left_edge - 0.02, y, lab, va="center",
            ha="left" if side == "right" else "right", fontsize=6, color=INK)
bx.set_yticks([])
bx.set_ylim(-0.7, len(est) - 0.3)
for s in ("top", "right", "left"):
    bx.spines[s].set_visible(False)
bx.set_xlim(0.95, 2.0)
bx.set_xlabel(r"Giant mass $M_1$ (M$_\odot$)")
for a in (ax, bx):
    a.tick_params(length=3, width=0.6)
fig.savefig(os.path.join(HERE, "fig_m1m2.pdf"), bbox_inches="tight",
            metadata={"CreationDate": None})  # no timestamp, so reruns give identical files
fig.savefig(os.path.join(HERE, "fig_m1m2.png"), bbox_inches="tight")
print("\nWrote fig_m1m2.pdf and fig_m1m2.png")
