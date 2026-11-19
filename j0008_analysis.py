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
M1A, SIG_M1A = val("M1_adopted_this_note"), half_width("M1_adopted_this_note")  # this note's M1


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
dm2_dm1 = (m2_from(1.67, 73) - m2_from(M1A, 73)) / (1.67 - M1A)
print(f"dM2/dM1 at i=73 deg: {dm2_dm1:.2f}")
print(f"M2({M1A:.2f}, i=62/73/84/90) = "
      + ", ".join(f"{m2_from(M1A, i):.2f}" for i in (62, 73, 84, 90)))
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
# The same with the SDSS-V (DR19) corrected log g (Zhao et al. Table S2). A separate random
# stream keeps every other number in this script unchanged.
rng_s5 = np.random.default_rng(SEED + 1)
lg5, sig_lg5 = val("logg_SDSSV_corr"), half_width("logg_SDSSV_corr")
m1_s5_s = grav_mass(rng_s5.normal(lg5, sig_lg5, N_MC),
                    r_paper * (1e3 / rng_s5.normal(plx - zp, sig_plx, N_MC)) / D_PAPER)
m1_s5 = np.percentile(m1_s5_s, [16, 50, 84])
m1_s5_mu = grav_mass(lg5, r_paper * d_corr / D_PAPER)
print(f"gR^2/G at {d_corr:.0f} pc with the SDSS-V log g ({lg5}): M1 = {m1_s5_mu:.2f} +/- {m1_s5_s.std():.2f} "
      f"Msun (16/50/84%: {m1_s5[0]:.2f}/{m1_s5[1]:.2f}/{m1_s5[2]:.2f}); x{10 ** (lg5 - logg):.3f} of the DR17 value")

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

# ----------------------------- APOKASC-3 seismic masses of giants like J0008
# Pinsonneault et al. (2025, ApJS 276, 69), VizieR J/ApJS/276/69 table 4 (selected columns in
# data/apokasc3_table4.csv): APOGEE DR17 Teff, log g, [Fe/H], [C/Fe], [N/Fe] and seismic masses.
# (a) three broad selections: Teff from 4700, 4800 or 4900 K to 5050 K, log g 3.0-3.3,
#     [Fe/H] and [C/N] from -0.3 to 0;
# (b) boxes centred on each survey's corrected parameters in Zhao et al. Table S2:
#     +/-80 K, +/-0.07 dex in log g, +/-0.1 dex in [Fe/H] and in [C/N] (DR17 [C/N] = -0.18).
print("\n== APOKASC-3 seismic masses of similar giants ==")
ak = {}
with open(os.path.join(HERE, "data", "apokasc3_table4.csv")) as fh:
    rows = list(csv.DictReader(fh))
for c in ("Teff", "loggSpec", "[Fe/H]", "[C/Fe]", "[N/Fe]", "Mass"):
    ak[c] = np.array([float(r[c]) if r[c].strip() else np.nan for r in rows])
ak_cn = ak["[C/Fe]"] - ak["[N/Fe]"]
ak_ok = np.isfinite(ak["Mass"]) & (ak["Mass"] > 0) & np.isfinite(ak_cn)


def apokasc(sel):
    m = ak["Mass"][sel & ak_ok]
    return len(m), np.percentile(m, [16, 50, 84]), np.mean(m >= 1.52), np.mean(m >= 1.67)


box = ((ak["loggSpec"] > 3.0) & (ak["loggSpec"] < 3.3) & (ak["[Fe/H]"] > -0.3) & (ak["[Fe/H]"] < 0)
       & (ak_cn > -0.3) & (ak_cn < 0))
for tlo in (4700, 4800, 4900):
    n, q, f152, f167 = apokasc(box & (ak["Teff"] > tlo) & (ak["Teff"] < 5050))
    print(f"  broad selection, Teff {tlo}-5050 K: N = {n}, M = {q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f}); "
          f"P(M>=1.52) = {f152:.2f}, P(M>=1.67) = {f167:.2f}")
cn_j = val("C_Fe_APOGEE") - val("N_Fe_APOGEE")
apokasc_res = {}
for lab, t0, g0, fe0 in (("APOGEE DR17", 4964, 3.14, -0.24), ("APOGEE SDSS-V", 4984, 3.19, -0.22),
                         ("LAMOST LRS LASP", 4993, 3.27, -0.22), ("LAMOST LRS DD-Payne", 4986, 3.30, -0.22),
                         ("LAMOST MRS LASP", 4998, 3.20, -0.28)):
    sel = ((np.abs(ak["Teff"] - t0) <= 80) & (np.abs(ak["loggSpec"] - g0) <= 0.07)
           & (np.abs(ak["[Fe/H]"] - fe0) <= 0.1) & (np.abs(ak_cn - cn_j) <= 0.1))
    n, q, f152, f167 = apokasc(sel)
    apokasc_res[lab] = q
    print(f"  centred on {lab:20s} ({t0} K, {g0}): N = {n:3d}, M = {q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f}); "
          f"P(M>=1.52) = {f152:.2f}, P(M>=1.67) = {f167:.2f}")
M1_APOKASC = apokasc_res["APOGEE DR17"]  # shown in Figure 1

# ------------------------------------------- 3. Monte Carlo companion mass
print("\n== Section 3: companion mass ==")
m1_s, inc_s, m2_s = draw(val("M1_adopted_this_note"), half_width("M1_adopted_this_note"))
print(f"Adopted M1 = {M1A:.2f} +/- {SIG_M1A:.2f}, i = 73 +/- 11: {describe(m2_s)}")
tail = m2_s > 2.3
print(f"  draws with M2 > 2.3: {np.mean(inc_s[tail] < 62):.2f} have i < 62 deg")
print(f"  std(M2): all {m2_s.std():.2f}; M1 varied alone {m2_vec(m1_s, 73.0).std():.2f}; "
      f"i varied alone {m2_vec(M1A, inc_s).std():.2f} Msun")
print(f"Minimum mass at i=90 deg for M1={M1A:.2f}: {m2_from(M1A, 90):.2f} Msun")
print("Robustness to the adopted M1:")
for mu, sig, lab in ((1.30, 0.20, "MIST track masses, wider uncertainty"),
                     (round(m1_plx_mu, 2), round(m1_plx_sig, 2), "gR^2/G, ZP-corrected parallax"),
                     (1.67, 0.15, "Zhao et al. adopted M1"),
                     (round(m1_s5_mu, 2), round(m1_s5_s.std(), 2), "gR^2/G, ZP-corrected parallax, SDSS-V log g")):
    print(f"  M1 = {mu:.2f} +/- {sig:.2f} ({lab}): {describe(draw(mu, sig)[2])}")
for m1_par, lab in ((1.38, "ZP-corrected parallax"), (1.48, "uncorrected parallax")):
    print(f"Parallax-based M1 = {m1_par} ({lab}) -> M2(i=73) = {m2_from(m1_par, 73):.2f} Msun")

# Population prior x J0008's own gR^2/G data: draw M1 from the adopted APOKASC-3 prior, weight each
# draw by the likelihood of the observed log g given M1 and the radius at a distance drawn from the
# zero-point-corrected parallax (the SED radius scales with d), and resample. As in the gR^2/G
# estimates above, the SED radius is fixed at a given distance unless sig_r is set. Separate random
# stream, so the numbers above are unchanged.
rng_c = np.random.default_rng(SEED + 2)


def combine(lg_obs, sig_lg, zp_=zp, sig_r=0.0, n=4 * N_MC):
    m1 = rng_c.normal(M1A, SIG_M1A, n)
    d = 1e3 / rng_c.normal(plx - zp_, sig_plx, n)
    r = rng_c.normal(r_paper, sig_r, n) * d / D_PAPER
    ok = (m1 > 0) & (d > 0)
    lg_model = np.log10(G_CGS * np.where(ok, m1, 1.0) * MSUN / (r * RSUN) ** 2)
    w = np.where(ok, np.exp(-0.5 * ((lg_obs - lg_model) / sig_lg) ** 2), 0.0)
    idx = rng_c.choice(n, N_MC, p=w / w.sum())
    inc = rng_c.normal(73.0, 11.0, N_MC)
    inc = np.where(inc > 90, 180 - inc, inc)
    return m1[idx], m2_vec(m1[idx], inc, rng_c.normal(FM, SIG_FM, N_MC))


print("APOKASC-3 prior (M1 = 1.25 +/- 0.15) combined with J0008's log g and parallax:")
for lg_obs, sig_lg, zp_, sig_r, lab in (
        (logg, sig_logg, zp, 0.0, "DR17 log g, ZP-corrected parallax"),
        (lg5, sig_lg5, zp, 0.0, "SDSS-V log g, ZP-corrected parallax"),
        (logg, sig_logg, zp, half_width("R1_SED"), "DR17 log g, ZP-corrected parallax, SED radius error"),
        (logg, sig_logg, 0.0, 0.0, "DR17 log g, uncorrected parallax")):
    m1c, m2c = combine(lg_obs, sig_lg, zp_, sig_r)
    q = np.percentile(m1c, [16, 50, 84])
    print(f"  {lab}: M1 = {q[1]:.2f} +{q[2] - q[1]:.2f} -{q[1] - q[0]:.2f}; {describe(m2c)}")

# ------------------------------------------------------------ Figure 1
BLUE, ORANGE, VIOLET, INK, MUTED = "#2a78d6", "#eb6834", "#6250d6", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
grid = np.linspace(0.95, 2.0, 200)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 4.6), dpi=300, sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.5], "hspace": 0.08})
ax.axhspan(2.15, 2.30, color="#e4e3df", lw=0, zorder=0)
ax.text(0.97, 2.225, r"max. NS mass (2.15–2.3 M$_\odot$)", va="center", fontsize=6.5, color=MUTED)
ax.axvspan(M1A - SIG_M1A, M1A + SIG_M1A, color=BLUE, alpha=0.12, lw=0, zorder=0)
ax.axvspan(1.52, 1.82, color=ORANGE, alpha=0.12, lw=0, zorder=0)
ax.fill_between(grid, m2_vec(grid, 84.0), m2_vec(grid, 62.0), color=VIOLET, alpha=0.25,
                lw=0, zorder=1)
ax.plot(grid, m2_vec(grid, 73.0), color=INK, lw=1.2, zorder=2)
ax.plot(grid, m2_vec(grid, 90.0), color=INK, lw=0.8, ls="--", zorder=2)
ax.text(1.99, m2_from(1.99, 73) + 0.03, r"$i=73^\circ\pm11^\circ$", ha="right", va="bottom",
        fontsize=6.5, color=INK)
ax.text(1.40, m2_from(1.40, 90) - 0.05, r"$i=90^\circ$ (minimum)", ha="left", va="top",
        fontsize=6.5, color=INK)
for m, col in ((M1A, BLUE), (1.67, ORANGE)):
    y = m2_from(m, 73)
    ax.plot(m, y, "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=4)
ax.set_ylim(1.55, 2.6)
ax.set_ylabel(r"Companion mass $M_2$ (M$_\odot$)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# (label, (value, lower, upper), colour, show error bar, label side)
est = [("Zhao+26 adopted (gR²/G, d=1.46 kpc)", inp["M1_Zhao_adopted"], ORANGE, True, "left"),
       ("Zhao+26 MIST isochrones", inp["M1_Zhao_MIST"], ORANGE, True, "left"),
       (f"gR²/G, {d_corr / 1e3:.2f} kpc, SDSS-V log g", (m1_s5_mu, m1_s5[0], m1_s5[2]), INK, True, "left"),
       ("MIST tracks (Teff, log g)", inp["M1_MIST_tracks"], INK, True, "right"),
       (f"gR²/G, {d_corr / 1e3:.2f} kpc, DR17 log g", (m1_plx_mu, m1_plx[0], m1_plx[2]),
        INK, True, "right"),
       ("APOKASC-3 seismic masses", (M1_APOKASC[1], M1_APOKASC[0], M1_APOKASC[2]), BLUE, True, "right"),
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

# ------------------------------- Implied prior on M2 (a check; not quoted in the note)
# Drawing (M1, i, f) and solving for M2 is equivalent to a posterior with priors on M1 and i,
# a Gaussian likelihood on f, and an implied conditional prior on M2 proportional to
#   df/dM2 = sin^3 i * M2^2 (M2 + 3 M1) / (M1 + M2)^3.
# Importance weights pi_new(M2) / (df/dM2) give the posterior under other M2 priors. This
# note's case reuses the Section 3 draws; Zhao et al.'s M1 = 1.67 +/- 0.15 takes new draws
# here, after every number above, so nothing earlier changes.
print("\n== Check: implied prior on M2 (not in the note) ==")


def ns_pop(m, mmax=2.12):
    """Neutron star mass distribution of Alsing, Silva & Berti (2018, MNRAS 478, 1377): two
    Gaussians (1.34 +/- 0.07, 1.80 +/- 0.21; weight 0.65 in the first) with a sharp maximum mass."""
    gauss = lambda x, mu, sig: np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    return np.where(m < mmax, 0.65 * gauss(m, 1.34, 0.07) + 0.35 * gauss(m, 1.80, 0.21), 0.0)


def prior_weights(m1, inc, m2):
    dfdm2 = np.sin(np.radians(inc)) ** 3 * m2 ** 2 * (m2 + 3 * m1) / (m1 + m2) ** 3
    return [("implied (as in the note)", np.ones_like(m2), BLUE),
            ("flat in M2", 1 / dfdm2, MUTED),
            ("NS population, Alsing+18", ns_pop(m2) / dfdm2, ORANGE)]


def describe_w(m2, w):
    w = w / w.sum()
    order = np.argsort(m2)
    p16, p50, p84 = np.interp([0.16, 0.5, 0.84], np.cumsum(w[order]), m2[order])
    return (f"M2 = {p50:.2f} +{p84 - p50:.2f} -{p50 - p16:.2f} Msun; P(M2>2.0) = {w[m2 > 2.0].sum():.2f}, "
            f"P(M2>2.3) = {w[m2 > 2.3].sum():.2f} (effective samples {1 / np.sum(w ** 2):,.0f})")


m1_z, inc_z, m2_z = draw(1.67, 0.15)
cases_m2 = [("this note", rf"this note: $M_1={M1A:.2f}\pm{SIG_M1A:.2f}$", m1_s, inc_s, m2_s),
            ("Zhao et al. M1", r"Zhao et al.: $M_1=1.67\pm0.15$", m1_z, inc_z, m2_z)]
for name, _, m1c, incc, m2c in cases_m2:
    print(f" {name} (M1 = {m1c.mean():.2f}, i = 73 +/- 11 deg):")
    for lab, w, _ in prior_weights(m1c, incc, m2c):
        print(f"  {lab:26s} {describe_w(m2c, w)}")

fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.7), dpi=300, gridspec_kw={"wspace": 0.3})
ax = axs[0]
grid_m2 = np.linspace(1.2, 3.0, 300)
for m1v, ls in ((M1A, "-"), (1.67, "--")):
    g = grid_m2 ** 2 * (grid_m2 + 3 * m1v) / (m1v + grid_m2) ** 3
    ax.plot(grid_m2, g / np.interp(2.0, grid_m2, g), ls, color=BLUE, lw=1.2,
            label=rf"implied, $M_1={m1v:.2f}$, $i=73^\circ$")
ax.axhline(1, color=MUTED, lw=0.8, label="flat")
pop_g = ns_pop(grid_m2)
ax.plot(grid_m2, pop_g / pop_g.max(), color=ORANGE, lw=1.2, label="NS population (peak = 1)")
ax.set(xlabel=r"$M_2$ (M$_\odot$)", ylabel="prior density (relative)", ylim=(0, 1.6), title="priors on $M_2$")
ax.title.set_fontsize(7)
ax.legend(fontsize=5.5, frameon=False)
bins = np.linspace(1.4, 3.2, 73)
for bx, (_, title, m1c, incc, m2c) in zip(axs[1:], cases_m2):
    bx.axvspan(2.15, 2.30, color="#e4e3df", lw=0, zorder=0)
    for lab, w, col in prior_weights(m1c, incc, m2c):
        bx.hist(m2c, bins=bins, weights=w / w.sum(), density=True, histtype="step", color=col, lw=1.2,
                label=lab)
    bx.set(xlabel=r"$M_2$ (M$_\odot$)", ylabel="posterior density", xlim=(1.4, 3.2))
    bx.set_title(title + r", $i=73^\circ\pm11^\circ$", fontsize=7)
xq = np.linspace(1.4, 3.2, 400)
quoted = np.where(xq < 2.20, np.exp(-0.5 * ((xq - 2.20) / 0.13) ** 2), np.exp(-0.5 * ((xq - 2.20) / 0.14) ** 2))
axs[2].plot(xq, quoted / np.trapz(quoted, xq), ":", color=INK, lw=1.0, label=r"quoted $2.20^{+0.14}_{-0.13}$")
axs[1].legend(fontsize=5.5, frameon=False)
axs[2].legend(fontsize=5.5, frameon=False)
for a in axs:
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.tick_params(length=3, width=0.6)
fig.savefig(os.path.join(HERE, "fig_m2_prior.png"), bbox_inches="tight")
print("Wrote fig_m2_prior.png")
