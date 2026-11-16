"""HR and Kiel diagrams for the giant in J0008+0128 with MIST tracks.

Shows where the giant sits for the three distances discussed in the note
(spectroscopic 1.30 kpc, zero-point-corrected parallax 1.42 kpc, uncorrected
parallax 1.46 kpc) against MIST v1.2 tracks at [Fe/H] = -0.25.

Usage:
  python j0008_hrd.py                 # downloads the MIST EEP tarball (~110 MB)
  python j0008_hrd.py /path/to/EEPS   # use an already unpacked EEPS directory

Requires: numpy, matplotlib.
"""
import glob
import os
import sys
import tarfile
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEH_TAG = "m0.25"  # MIST grid point closest to [Fe/H] = -0.25
TARBALL = f"MIST_v1.2_feh_{FEH_TAG}_afe_p0.0_vvcrit0.4_EEPS.txz"
URL = f"https://mist.science/data/tarballs_v1.2/{TARBALL}"  # old waps.cfa.harvard.edu URL redirects here
MASSES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# Observed star (APOGEE DR17 calibrated values from Zhao et al. Table S2;
# luminosity from the paper's SED radius rescaled to each distance).
TEFF, TEFF_ERR = 4964.0, 40.0
LOGG, LOGG_ERR = 3.14, 0.07
TSUN = 5772.0
R_PAPER, D_PAPER = 5.39, 1464.56  # Rsun at pc
TEFF_SED = 4929.0  # Teff of the SED fit that gave R_PAPER; L = R^2 Teff^4 uses this pair
TEFF_FLAME = 5063.0  # Teff implied by the Gaia DR3 FLAME luminosity and radius (15.244 Lsun, 5.0747 Rsun)
DISTANCES = [  # (pc, 1-sigma pc, label, colour)
    (1296.0, 86.0, "spectroscopic 1.30 kpc", "#2a78d6"),
    (1416.0, 60.0, "ZP-corrected parallax 1.42 kpc", "#1baf7a"),
    (1464.6, 64.0, "uncorrected parallax 1.46 kpc", "#eb6834"),
]


def get_eep_dir(arg=None):
    if arg:
        return arg
    if not os.path.exists(TARBALL):
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, TARBALL)
    with tarfile.open(TARBALL) as tf:
        top = tf.getnames()[0].split("/")[0]
        if not os.path.isdir(top):
            try:
                tf.extractall(filter="data")
            except TypeError:  # Python without tarfile extraction filters
                tf.extractall()
    return top


def read_eep(path):
    """Read a MIST .track.eep file into a dict of columns."""
    header = None
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith("#") and "star_age" in line:
            header = line.lstrip("#").split()
    data = np.loadtxt([l for l in lines if not l.startswith("#") and l.strip()])
    return {name: data[:, i] for i, name in enumerate(header)}


def track_file(eep_dir, mass):
    name = f"{int(round(mass * 100)):05d}M.track.eep"
    hits = glob.glob(os.path.join(eep_dir, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def lum_at(d):
    """Luminosity (Lsun) from the SED radius rescaled to distance d."""
    return (R_PAPER * d / D_PAPER) ** 2 * (TEFF_SED / TSUN) ** 4


def teff_on_rgb(t, col, target):
    """Teff where the post-main-sequence track first reaches `target` in column `col`
    (log_g falling, or log_L rising on the RGB). NaN if the track never gets there."""
    rgb = t["phase"] == 2
    x, teff = t[col][rgb], 10 ** t["log_Teff"][rgb]
    if col == "log_L":  # skip the subgiant dip in L for the more massive tracks
        start = np.argmin(x)
        x, teff = x[start:], teff[start:]
        cross = np.nonzero((x[:-1] < target) & (x[1:] >= target))[0]
    else:
        cross = np.nonzero((x[:-1] > target) & (x[1:] <= target))[0]
    if not len(cross):
        return np.nan
    j = cross[0]
    return np.interp(target, sorted([x[j], x[j + 1]]),
                     [teff[j], teff[j + 1]] if x[j] < x[j + 1] else [teff[j + 1], teff[j]])


def track_mass(tracks, col, target, teff_obs):
    """Mass interpolated between tracks at the observed Teff, at fixed log g or log L."""
    ms = np.array([m for m in tracks])
    tt = np.array([teff_on_rgb(tracks[m], col, target) for m in ms])
    ok = np.isfinite(tt)
    order = np.argsort(tt[ok])
    return np.interp(teff_obs, tt[ok][order], ms[ok][order], left=np.nan, right=np.nan)


def main():
    eep_dir = get_eep_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), dpi=200)
    cmap = plt.get_cmap("Greys")
    tracks = {}
    for k, m in enumerate(MASSES):
        t = tracks[m] = read_eep(track_file(eep_dir, m))
        # main sequence turn-off to tip of the RGB: MIST phase 0 (MS) and 2 (RGB)
        sel = (t["phase"] >= 0) & (t["phase"] <= 2)
        col = cmap(0.35 + 0.6 * k / (len(MASSES) - 1))
        ax1.plot(10 ** t["log_Teff"][sel], t["log_L"][sel], color=col, lw=1)
        ax2.plot(10 ** t["log_Teff"][sel], t["log_g"][sel], color=col, lw=1)
        # label each track on the RGB near logL = 1.35
        rgb = sel & (t["phase"] == 2)
        if rgb.any():
            i = np.argmin(np.abs(t["log_L"][rgb] - 1.35))
            ax1.text(10 ** t["log_Teff"][rgb][i] + 25, t["log_L"][rgb][i],
                     f"{m:.1f}", fontsize=7, color=col, ha="right", va="center")
            # Kiel labels on the subgiant branch, where the tracks are well separated
            j = np.argmin(np.abs(10 ** t["log_Teff"][rgb] - 5500))
            ax2.text(5500, t["log_g"][rgb][j] - 0.015, f"{m:.1f}", fontsize=7, color=col,
                     ha="center", va="bottom")

    # Track masses at the observed Teff (Teff +/- 1 sigma gives the range)
    print("MIST [Fe/H]=-0.25 track mass at the observed Teff "
          f"({TEFF:.0f} +/- {TEFF_ERR:.0f} K):")
    lo_hi = lambda col, x: [track_mass(tracks, col, x, TEFF + s * TEFF_ERR) for s in (1, 0, -1)]
    hi_m, mid, lo_m = lo_hi("log_g", LOGG)
    print(f"  Kiel, log g = {LOGG}: M = {mid:.2f} ({lo_m:.2f}-{hi_m:.2f}) Msun")
    print(f"  Kiel, log g = {LOGG}, at the FLAME Teff of {TEFF_FLAME:.0f} K: "
          f"M = {track_mass(tracks, 'log_g', LOGG, TEFF_FLAME):.2f} Msun")
    for d, _, lab, _ in DISTANCES:
        hi_m, mid, lo_m = lo_hi("log_L", np.log10(lum_at(d)))
        print(f"  HR, {lab} (log L = {np.log10(lum_at(d)):.2f}): "
              f"M = {mid:.2f} ({lo_m:.2f}-{hi_m:.2f}) Msun")

    for d, sd, lab, c in DISTANCES:
        logl = np.log10(lum_at(d))
        dlog = 2 * sd / d / np.log(10)  # L scales as d^2
        ax1.errorbar(TEFF, logl, xerr=TEFF_ERR, yerr=dlog, fmt="o", color=c,
                     mec="white", ms=6, label=lab, zorder=5)
    ax2.errorbar(TEFF, LOGG, xerr=TEFF_ERR, yerr=LOGG_ERR, fmt="o", color="k",
                 mec="white", ms=6, zorder=5, label="APOGEE DR17 (distance-free)")

    for ax in (ax1, ax2):
        ax.set_xlim(5600, 4500)
        ax.set_xlabel(r"$T_{\rm eff}$ (K)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_ylim(0.6, 1.8)
    ax1.set_ylabel(r"$\log L/L_\odot$")
    ax1.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax1.set_title(f"HR diagram, MIST [Fe/H] = -0.25 (masses in M$_\\odot$)", fontsize=8)
    ax2.set_ylim(3.9, 2.4)
    ax2.set_ylabel(r"$\log g$")
    ax2.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax2.set_title("Kiel diagram", fontsize=8)
    fig.tight_layout()
    fig.savefig("j0008_hrd.png", bbox_inches="tight")
    print("Wrote j0008_hrd.png")


if __name__ == "__main__":
    main()
